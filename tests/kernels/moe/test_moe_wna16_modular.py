# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tests.kernels.moe.utils import (
    fused_moe,
    make_dummy_moe_config,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import MoEActivation, fused_topk
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.quantization.moe_wna16 import (
    _MoeWNA16ExpertsAdapter,
)


def test_moe_wna16_modular_matches_legacy(workspace_init):
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 17
    hidden_size = 256
    intermediate_size = 256
    num_experts = 8
    topk = 4
    group_size = 64

    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype
    )
    router_logits = torch.randn(
        num_tokens, num_experts, device=device, dtype=dtype
    )
    w1 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w1_scale = torch.rand(
        num_experts,
        2 * intermediate_size,
        hidden_size // group_size,
        device=device,
        dtype=dtype,
    ) / 100
    w2_scale = torch.rand(
        num_experts,
        hidden_size,
        intermediate_size // group_size,
        device=device,
        dtype=dtype,
    ) / 100
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=[0, group_size],
    )

    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        in_dtype=dtype,
        activation=MoEActivation.SILU,
    )
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=quant_config,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None
    modular_kernel = FusedMoEKernel(
        prepare_finalize,
        _MoeWNA16ExpertsAdapter(moe_config, quant_config),
    )
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, router_logits.float(), topk, False
    )

    with set_current_vllm_config(VllmConfig()):
        legacy_output = fused_moe(
            hidden_states,
            w1,
            w2,
            router_logits,
            topk,
            quant_config=quant_config,
            global_num_experts=num_experts,
        )
        modular_output = modular_kernel.apply(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    torch.testing.assert_close(modular_output, legacy_output, rtol=0, atol=0)
