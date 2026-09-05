# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.device_communicators import ce_a2a as ce_a2a_module
from vllm.distributed.device_communicators.ce_a2a import (
    CeA2AAll2AllManager,
    _attention_phase,
    _codec_group_count,
    _forward_phase,
    _from_ce_wire,
    _ladder_views,
    _pace_policy_name,
    _prefill_policy_fallback_reason,
    _proxy_cpu_for_rank,
    _solve_max_edge_dispatch_plan,
    _to_ce_wire,
    _value_row_bytes,
    _worker_cpu_for_rank,
)


def test_gptoss_group64_codec_layout() -> None:
    from ce_a2a_moe import DispatchPacketSpec, lowbit_block_payload_bytes

    assert _codec_group_count(2880, 64) == 45
    assert _value_row_bytes(2880, 6, 64) == 2160
    spec = DispatchPacketSpec(
        hidden_size=2880,
        top_k=4,
        activation_bits=6,
        group_size=64,
    )
    assert spec.metadata_offset == 2252
    assert spec.metadata_padding_bytes == 2
    assert spec.packet_bytes == 2284
    assert lowbit_block_payload_bytes(2880, 6, 64) == 2250


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PACE codec requires CUDA")
def test_uncompressed_bf16_compute_uses_fp16_wire_round_trip() -> None:
    from ce_a2a_moe import (
        DispatchPacketSpec,
        FusedBlockDispatchBuilder,
        unpack_fixed_dispatch_blocks,
    )

    rows, hidden_size, top_k = 11, 128, 2
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(19)
    hidden = torch.randn(
        (rows, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    topk_ids = torch.randint(
        4,
        (rows, top_k),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    topk_weights = torch.rand(
        (rows, top_k),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    spec = DispatchPacketSpec(
        hidden_size=hidden_size,
        top_k=top_k,
        activation_bits=16,
    )
    wire_hidden = _to_ce_wire(hidden)
    builder = FusedBlockDispatchBuilder(
        spec=spec,
        experts_per_rank=4,
        world_size=1,
        max_edge_rows=rows,
        device=device,
    )
    built = builder.build(wire_hidden, topk_ids, topk_weights)
    recv_hidden, recv_ids, recv_weights = unpack_fixed_dispatch_blocks(
        built.packets,
        built.send_counts,
        spec=spec,
        expert_rank=0,
        experts_per_rank=4,
        output_hidden=torch.empty(
            (rows + 1, hidden_size), dtype=torch.float16, device=device
        ),
    )

    restored = _from_ce_wire(recv_hidden[:rows], torch.bfloat16)
    torch.testing.assert_close(restored, hidden, rtol=0, atol=0)
    torch.testing.assert_close(recv_ids[:rows], topk_ids, rtol=0, atol=0)
    torch.testing.assert_close(recv_weights[:rows], topk_weights, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PACE codec requires CUDA")
@pytest.mark.parametrize("bits", [5, 6])
def test_gptoss_group64_fused_codes_are_bit_exact(bits: int) -> None:
    from ce_a2a_moe import DispatchPacketSpec, FusedBlockDispatchBuilder
    from ce_a2a_moe.lowbit import quantize_pack_lowbit_blocks
    from ce_a2a_moe.packet import dispatch_packet_views, dispatch_scale_view

    rows, hidden_size, top_k = 17, 2880, 4
    world_size, experts_per_rank, group_size = 4, 8, 64
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(bits)
    hidden = torch.randn(
        (rows, hidden_size),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    topk_ids = torch.randint(
        world_size * experts_per_rank,
        (rows, top_k),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    topk_weights = torch.rand(
        (rows, top_k),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    spec = DispatchPacketSpec(
        hidden_size=hidden_size,
        top_k=top_k,
        activation_bits=bits,
        group_size=group_size,
    )
    builder = FusedBlockDispatchBuilder(
        spec=spec,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
        max_edge_rows=rows,
        device=device,
    )
    fused = builder.build(hidden, topk_ids, topk_weights)
    flat = fused.packets.reshape(-1, spec.packet_bytes)
    values, _ids, _weights = dispatch_packet_views(flat, spec)
    scales = dispatch_scale_view(flat, spec)
    block_rows = rows + 1

    for destination, count in enumerate(fused.send_counts.cpu().tolist()):
        if not count:
            continue
        gathered = hidden[fused.token_ids[destination, :count]]
        expected_values = torch.empty(
            (count, _value_row_bytes(hidden_size, bits, group_size)),
            dtype=torch.uint8,
            device=device,
        )
        expected_scales = torch.empty(
            (count, _codec_group_count(hidden_size, group_size)),
            dtype=torch.float16,
            device=device,
        )
        quantize_pack_lowbit_blocks(
            gathered.contiguous(),
            expected_values,
            expected_scales,
            bits=bits,
            group_size=group_size,
        )
        start = destination * block_rows
        torch.testing.assert_close(
            values[start : start + count], expected_values, rtol=0, atol=0
        )
        torch.testing.assert_close(
            scales[start : start + count], expected_scales, rtol=0, atol=0
        )


@pytest.mark.parametrize(
    ("hidden_size", "top_k", "group_size"),
    [(2048, 8, 128), (2880, 4, 64)],
)
def test_max_edge_plan_keeps_cross_rank_ties_int5_and_respects_budget(
    hidden_size: int, top_k: int, group_size: int
) -> None:
    from ce_a2a_moe.packet import DispatchPacketSpec, dispatch_ladder_packet_bytes

    spec = DispatchPacketSpec(
        hidden_size=hidden_size,
        top_k=top_k,
        activation_bits=5,
        group_size=group_size,
        carry_token_id=True,
    )
    peak = 128
    rank_rows = ([128, 73, 29, 0], [41, 128, 64, 7])
    floor = dispatch_ladder_packet_bytes(spec, 0)
    for counts in rank_rows:
        promoted, widths = _solve_max_edge_dispatch_plan(counts, spec, peak)
        assert promoted[counts.index(peak)] == 0
        assert any(
            groups > 0
            for count, groups in zip(counts, promoted)
            if count < peak
        )
        for count, groups, width in zip(counts, promoted, widths):
            assert width == dispatch_ladder_packet_bytes(spec, groups)
            assert count * width <= peak * floor


def test_gptoss_plan_uses_exact_aligned_widths() -> None:
    from ce_a2a_moe.packet import DispatchPacketSpec

    spec = DispatchPacketSpec(
        hidden_size=2880,
        top_k=4,
        activation_bits=5,
        group_size=64,
        carry_token_id=True,
    )
    promoted, widths = _solve_max_edge_dispatch_plan([100, 99], spec, 100)
    assert promoted == [0, 2]
    assert widths == [1928, 1944]
    assert all(width % 4 == 0 for width in widths)


def test_device_max_edge_plan_matches_reference_without_host_copies() -> None:
    from ce_a2a_moe.packet import DispatchPacketSpec, dispatch_ladder_packet_bytes

    spec = DispatchPacketSpec(
        hidden_size=2048,
        top_k=8,
        activation_bits=5,
        group_size=128,
        carry_token_id=True,
    )
    counts = torch.tensor([128, 73, 29, 0], dtype=torch.int32)
    peak = torch.tensor(128, dtype=torch.int32)
    expected_groups, expected_widths = _solve_max_edge_dispatch_plan(
        counts.tolist(), spec, int(peak)
    )
    manager = CeA2AAll2AllManager.__new__(CeA2AAll2AllManager)
    manager.dispatch_floor_row = dispatch_ladder_packet_bytes(spec, 0)
    manager.dispatch_groups = spec.groups
    wide_row = dispatch_ladder_packet_bytes(spec, spec.groups)
    manager.dispatch_ladder_step = (
        wide_row - manager.dispatch_floor_row
    ) // manager.dispatch_groups
    actual_groups = torch.empty_like(counts)

    manager._solve_max_edge_delta(counts, peak, actual_groups)
    actual_widths = actual_groups * manager.dispatch_ladder_step
    actual_widths.add_(manager.dispatch_floor_row)

    assert actual_groups.tolist() == expected_groups
    assert actual_widths.tolist() == expected_widths


def test_locked_five_arm_policies_select_distinct_paths() -> None:
    policies = {
        # Historical policy label; the raw 16-bit path now retains FP16/BF16.
        "fp16": (0, 0, False, False, False),
        "uniform_int6": (6, 6, False, False, False),
        "maxedge_delta_5to6": (5, 6, True, True, False),
        "lightedge_fill_6to7": (6, 6, False, False, True),
        "maxedge_delta_plus_fill": (5, 6, True, True, True),
    }
    selected = {_pace_policy_name(*policy) for policy in policies.values()}
    assert selected == set(policies)


def test_proxy_cpu_map_is_topology_aware_unique_and_exact() -> None:
    mapping = "20,21,22,23,48,49,50,51"

    assert [_proxy_cpu_for_rank(20, rank, 8, mapping) for rank in range(8)] == [
        20,
        21,
        22,
        23,
        48,
        49,
        50,
        51,
    ]
    assert _proxy_cpu_for_rank(20, 3, 8, "") == 23
    assert _proxy_cpu_for_rank(-1, 0, 8, "") is None
    with pytest.raises(ValueError, match="exactly one CPU per rank"):
        _proxy_cpu_for_rank(20, 0, 8, "20,21")
    with pytest.raises(ValueError, match="unique and nonnegative"):
        _proxy_cpu_for_rank(20, 0, 4, "20,20,21,22")


def test_worker_cpu_map_is_optional_unique_and_exact() -> None:
    mapping = "8,9,10,11,36,37,38,39"

    assert [_worker_cpu_for_rank(rank, 8, mapping) for rank in range(8)] == [
        8,
        9,
        10,
        11,
        36,
        37,
        38,
        39,
    ]
    assert _worker_cpu_for_rank(0, 8, "") is None
    with pytest.raises(ValueError, match="exactly one CPU per rank"):
        _worker_cpu_for_rank(0, 8, "8,9")
    with pytest.raises(ValueError, match="unique and nonnegative"):
        _worker_cpu_for_rank(0, 4, "8,8,9,10")


def test_ce_observability_environment_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm import envs

    monkeypatch.setenv("VLLM_CE_A2A_PHASE_TIMING", "1")
    monkeypatch.setenv("VLLM_CE_A2A_ENQUEUE_ONLY_WAIT", "1")
    monkeypatch.setenv("VLLM_CE_A2A_WORKER_CPU_MAP", "8,9,10,11")
    monkeypatch.setenv(
        "VLLM_CE_A2A_EDGE_SCHEDULE", "ep8_dual_numa_adaptive_v1"
    )

    assert envs.environment_variables["VLLM_CE_A2A_PHASE_TIMING"]() is True
    assert envs.environment_variables["VLLM_CE_A2A_ENQUEUE_ONLY_WAIT"]() is True
    assert (
        envs.environment_variables["VLLM_CE_A2A_WORKER_CPU_MAP"]()
        == "8,9,10,11"
    )
    assert (
        envs.environment_variables["VLLM_CE_A2A_EDGE_SCHEDULE"]()
        == "ep8_dual_numa_adaptive_v1"
    )


def test_gptoss_checkpoint_group_is_not_a_valid_codec_group() -> None:
    with pytest.raises(ValueError, match="divide hidden size"):
        _codec_group_count(2880, 128)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (SimpleNamespace(num_prefill_tokens=128, num_decode_tokens=0), "prefill"),
        (SimpleNamespace(num_prefill_tokens=1, num_decode_tokens=0), "prefill"),
        (SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=4), "decode"),
        (
            [
                {"a": SimpleNamespace(num_prefill_tokens=16, num_decode_tokens=0)},
                {"b": SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=2)},
            ],
            "mixed",
        ),
        (None, "unknown"),
    ],
)
def test_attention_phase(metadata: object, expected: str) -> None:
    assert _attention_phase(metadata) == expected


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("prefill", "prefill"),
        ("decode", "decode"),
        ("mixed", "mixed"),
        ("idle", "idle"),
        (None, "idle"),
    ],
)
def test_forward_phase_prefers_authoritative_runner_signal(
    monkeypatch: pytest.MonkeyPatch,
    phase: str | None,
    expected: str,
) -> None:
    additional_kwargs = (
        {} if phase is None else {"moe_attention_phase": phase}
    )
    context = SimpleNamespace(
        additional_kwargs=additional_kwargs,
        attn_metadata=None,
    )
    monkeypatch.setattr(ce_a2a_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(ce_a2a_module, "get_forward_context", lambda: context)
    assert _forward_phase() == expected


def test_forward_phase_classifies_missing_context_as_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ce_a2a_module, "is_forward_context_available", lambda: False
    )

    assert _forward_phase() == "idle"


def test_forward_phase_keeps_malformed_active_context_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        additional_kwargs={},
        attn_metadata={"layer": SimpleNamespace()},
    )
    monkeypatch.setattr(ce_a2a_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(ce_a2a_module, "get_forward_context", lambda: context)

    assert _forward_phase() == "unknown"


@pytest.mark.parametrize(
    ("phase", "reason"),
    [
        ("prefill", None),
        ("decode", "policy_decode"),
        ("idle", "policy_idle"),
        ("mixed", "policy_mixed_batch"),
        ("unknown", "policy_unknown"),
    ],
)
def test_prefill_policy_has_explicit_fallback_reason(
    phase: str, reason: str | None
) -> None:
    assert _prefill_policy_fallback_reason(phase) == reason


def test_fp16_codec_boundary_preserves_qwen_tensor() -> None:
    tensor = torch.tensor([[1.0, -2.0]], dtype=torch.float16)
    wire = _to_ce_wire(tensor)
    assert wire.data_ptr() == tensor.data_ptr()
    assert _from_ce_wire(wire, torch.float16).data_ptr() == tensor.data_ptr()


def test_bf16_codec_boundary_restores_compute_dtype() -> None:
    tensor = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    wire = _to_ce_wire(tensor)
    restored = _from_ce_wire(wire, torch.bfloat16)
    assert wire.dtype == torch.float16
    assert wire.data_ptr() != tensor.data_ptr()
    assert restored.dtype == torch.bfloat16
    assert restored.data_ptr() != wire.data_ptr()
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)


def test_bf16_compressed_codec_boundary_preserves_exponent_range() -> None:
    tensor = torch.tensor([[1.0e20, -1.0e20]], dtype=torch.bfloat16)
    wire = _to_ce_wire(tensor, compressed=True)
    assert wire.dtype == torch.bfloat16
    assert wire.data_ptr() == tensor.data_ptr()
    assert torch.isfinite(wire).all()
    assert wire.abs().max() > torch.finfo(torch.float16).max
    assert _from_ce_wire(wire, torch.bfloat16).data_ptr() == tensor.data_ptr()


def test_fp16_compressed_codec_boundary_is_byte_identical() -> None:
    tensor = torch.tensor([[1.0, -2.0]], dtype=torch.float16)
    before = tensor.view(torch.uint8).clone()
    wire = _to_ce_wire(tensor, compressed=True)
    assert wire.data_ptr() == tensor.data_ptr()
    assert torch.equal(wire.view(torch.uint8), before)


def test_ladder_views_selects_bf16_scale_storage() -> None:
    arena = torch.zeros((2, 3, 12), dtype=torch.uint8)
    values, scales = _ladder_views(arena, 8, torch.bfloat16)
    assert values.shape == (2, 36)
    assert scales.shape == (2, 3, 2)
    assert scales.dtype == torch.bfloat16


def test_wire_accounting_separates_phase_payload_and_self_edge() -> None:
    manager = CeA2AAll2AllManager.__new__(CeA2AAll2AllManager)
    manager.rank = 1
    manager.world_size = 4
    manager.ce_dispatch_wire_bytes = 0
    manager.ce_combine_wire_bytes = 0
    manager.ce_dispatch_payload_bytes = 0
    manager.ce_combine_payload_bytes = 0
    manager.ce_dispatch_packets = 0
    manager.ce_combine_packets = 0
    manager.ce_dispatch_messages = 0
    manager.ce_combine_messages = 0
    manager._ce_wire_bytes_device = torch.zeros(2, dtype=torch.int64)
    manager._ce_payload_bytes_device = torch.zeros(2, dtype=torch.int64)
    manager._ce_packet_counts_device = torch.zeros(2, dtype=torch.int64)
    manager._ce_message_counts_device = torch.zeros(2, dtype=torch.int64)

    counts = torch.tensor([3, 4, 5, 6], dtype=torch.int32)
    widths = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    manager._record_wire_bytes(0, counts, 10)
    manager._record_wire_bytes(0, counts, widths, payload=True)
    manager._record_wire_bytes(1, (2, 3, 4, 5), 7)
    manager._record_wire_bytes(1, (2, 3, 4, 5), widths, payload=True)

    assert manager._ce_wire_bytes_device.tolist() == [140, 0]
    assert manager._ce_payload_bytes_device.tolist() == [420, 0]
    assert manager._ce_packet_counts_device.tolist() == [14, 0]
    assert manager._ce_message_counts_device.tolist() == [3, 0]
    assert manager.ce_combine_wire_bytes == 77
    assert manager.ce_combine_payload_bytes == 340
    assert manager.ce_combine_packets == 11
    assert manager.ce_combine_messages == 3
