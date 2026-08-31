# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.device_communicators.ce_a2a import (
    CeA2AAll2AllManager,
    _attention_phase,
    _codec_group_count,
    _from_ce_wire,
    _prefill_policy_fallback_reason,
    _to_ce_wire,
    _value_row_bytes,
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
    ("phase", "reason"),
    [
        ("prefill", None),
        ("decode", "policy_decode"),
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
    assert restored.dtype == torch.bfloat16
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)


def test_wire_accounting_separates_phase_payload_and_self_edge() -> None:
    manager = CeA2AAll2AllManager.__new__(CeA2AAll2AllManager)
    manager.rank = 1
    manager.world_size = 4
    manager.ce_dispatch_wire_bytes = 0
    manager.ce_combine_wire_bytes = 0
    manager.ce_dispatch_payload_bytes = 0
    manager.ce_combine_payload_bytes = 0
    manager._ce_wire_bytes_device = torch.zeros(2, dtype=torch.int64)
    manager._ce_payload_bytes_device = torch.zeros(2, dtype=torch.int64)

    counts = torch.tensor([3, 4, 5, 6], dtype=torch.int32)
    widths = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    manager._record_wire_bytes(0, counts, 10)
    manager._record_wire_bytes(0, counts, widths, payload=True)
    manager._record_wire_bytes(1, (2, 3, 4, 5), 7)
    manager._record_wire_bytes(1, (2, 3, 4, 5), widths, payload=True)

    assert manager._ce_wire_bytes_device.tolist() == [140, 0]
    assert manager._ce_payload_bytes_device.tolist() == [420, 0]
    assert manager.ce_combine_wire_bytes == 77
    assert manager.ce_combine_payload_bytes == 340
