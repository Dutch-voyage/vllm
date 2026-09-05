from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu_ubatch_wrapper as gpu_ubatch_wrapper
from vllm.v1.worker.ubatch_utils import UBatchSlice


def test_ubatch_forward_context_preserves_additional_kwargs(monkeypatch):
    captured: list[dict[str, str] | None] = []

    def fake_create_forward_context(*args, additional_kwargs=None, **kwargs):
        captured.append(additional_kwargs)
        return object()

    def fake_make_ubatch_contexts(*, num_micro_batches, **kwargs):
        return [SimpleNamespace(id=index) for index in range(num_micro_batches)]

    monkeypatch.setattr(
        gpu_ubatch_wrapper,
        "create_forward_context",
        fake_create_forward_context,
    )
    monkeypatch.setattr(
        gpu_ubatch_wrapper,
        "make_ubatch_contexts",
        fake_make_ubatch_contexts,
    )

    wrapper = object.__new__(gpu_ubatch_wrapper.UBatchWrapper)
    wrapper.vllm_config = object()
    wrapper.ready_barrier = object()
    wrapper.comm_stream = object()
    phase = {"moe_attention_phase": "prefill"}
    slices = [
        UBatchSlice(slice(0, 1), slice(0, 2)),
        UBatchSlice(slice(1, 2), slice(2, 4)),
    ]

    metadata = wrapper._make_ubatch_metadata(
        ubatch_slices=slices,
        attn_metadata=[object(), object()],
        slot_mapping=None,
        input_ids=torch.arange(4),
        positions=torch.arange(4),
        inputs_embeds=None,
        intermediate_tensors=None,
        compute_stream=object(),
        dp_metadata=[object(), object()],
        batch_descriptor=object(),
        cudagraph_runtime_mode=object(),
        additional_kwargs=phase,
    )

    assert len(metadata) == 2
    assert captured == [phase, phase]
