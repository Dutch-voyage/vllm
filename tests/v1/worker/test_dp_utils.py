from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.dp_utils as dp_utils


@pytest.mark.parametrize(
    ("phase_codes", "expected"),
    [
        ([4, 4, 4, 4], "prefill"),
        ([4, 1, 2, 4], "mixed"),
    ],
)
def test_coordinate_moe_attention_phase_is_rank_consistent(
    monkeypatch, phase_codes, expected
):
    config = SimpleNamespace(
        data_parallel_size=4,
        data_parallel_rank=0,
        num_ubatches=2,
    )
    monkeypatch.setattr(
        dp_utils,
        "_get_device_and_group",
        lambda _config: ("cpu", object()),
    )

    def fake_all_reduce(tensor, group):
        tensor[0].fill_(8)
        tensor[1].fill_(8)
        tensor[2].zero_()
        tensor[3].zero_()
        tensor[4].copy_(torch.tensor(phase_codes, dtype=torch.int32))

    monkeypatch.setattr(dp_utils.dist, "all_reduce", fake_all_reduce)

    _, _, _, phase = dp_utils.coordinate_batch_and_moe_phase_across_dp(
        num_tokens_unpadded=8,
        allow_microbatching=False,
        parallel_config=config,
        moe_attention_phase="prefill",
    )
    assert phase == expected


def test_coordinate_moe_attention_phase_rejects_unknown_phase():
    config = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
    with pytest.raises(ValueError, match="unknown MoE attention phase"):
        dp_utils.coordinate_batch_and_moe_phase_across_dp(
            num_tokens_unpadded=8,
            allow_microbatching=False,
            parallel_config=config,
            moe_attention_phase="unknown",
        )
