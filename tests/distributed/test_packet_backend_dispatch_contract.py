"""CPU-only adapter dispatch contract; no vLLM/CUDA runtime import needed.

Compile the actual dispatch methods in isolation so unknown plugin names can be
exercised without creating GPUs or importing optional vLLM platform modules.
Full integration tests remain in test_ce_a2a.py.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def adapter(caps):
    source = (
        Path(__file__).parents[2] / "vllm/distributed/device_communicators/ce_a2a.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "CeA2AAll2AllManager"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_launch_fixed_exchange", "supports_async")
    ]
    isolated = ast.parse(
        "from __future__ import annotations\nclass Adapter:\n    pass\n"
    )
    isolated.body[1].body = methods
    namespace = {}
    exec(compile(ast.fix_missing_locations(isolated), str(source), "exec"), namespace)
    obj = namespace["Adapter"]()
    obj.control_kind = "external_plugin"
    obj.backend_caps = SimpleNamespace(
        async_packets=True,
        variable_widths=True,
        early_geometry=True,
        adaptive_schedule=False,
    )
    for name, value in caps.items():
        setattr(obj.backend_caps, name, value)
    obj.async_split = True
    obj._lane_id = 1
    calls = []

    def submit(*args, **kwargs):
        calls.append((args, kwargs))
        return lambda: calls.append("dependency")

    obj.control = SimpleNamespace(submit_async=submit)
    return obj, calls


def test_plugin_dispatch_preserves_buffers_lane_and_deferred_completion():
    obj, calls = adapter({})
    payload, counts, widths, stage = (object() for _ in range(4))
    hook = obj._launch_fixed_exchange(
        1, payload, counts, payload, counts, widths, widths, geometry_stage=stage
    )
    assert obj.supports_async()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("combine", payload, counts, payload, counts, widths, widths)
    assert kwargs == {"geometry_stage": stage, "lane": 1}
    hook()
    assert calls[-1] == "dependency"


@pytest.mark.parametrize(
    "cap,options",
    [
        ("early_geometry", {"geometry_stage": object()}),
        ("adaptive_schedule", {"global_count_matrix": object()}),
    ],
)
def test_unsupported_optional_contract_rejected_before_submit(cap, options):
    obj, calls = adapter({cap: False})
    with pytest.raises(RuntimeError):
        obj._launch_fixed_exchange(0, None, None, None, None, **options)
    assert calls == []


def test_unsupported_variable_width_rejected_before_submit():
    obj, calls = adapter({"variable_widths": False})
    with pytest.raises(RuntimeError):
        obj._launch_fixed_exchange(0, None, None, None, None, object(), object())
    assert calls == []
