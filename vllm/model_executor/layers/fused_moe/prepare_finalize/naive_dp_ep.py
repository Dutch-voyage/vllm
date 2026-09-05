# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave
from vllm.v1.worker.ubatching import dbo_maybe_run_recv_hook


def _no_wait(result):
    return (lambda: None), (lambda: result)


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
    # Defer input quantization to the MoE kernel.
    if defer_input_quant:
        a1q = a1
        a1q_scale = None
    else:
        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )

        # NOTE: swizzling pads the scales to multiple of 128
        # which makes the scales tensor different shape than
        # the hidden states, breaking the A2A kernel. So, we
        # delay the swizzling until after the A2A.
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            input_sf,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
            is_scale_swizzled=False,
            mx_alignment=quant_config.mx_alignment,
        )

    # Skip gathering scales if we have static quantization
    # (the scale is a scalar, replicated on all ranks) or
    # if quantization is deferred.
    skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
    scales = None if skip_gather_scales else [a1q_scale]

    return a1q, scales, a1q_scale


def _unwrap_scale_and_prepare_for_moe(
    scales: list[torch.Tensor] | None,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    assert scales is not None and len(scales) == 1
    a1q_scale = scales[0]
    # Apply swizzling after a2a if the MoE kernel needs it.
    if quant_config.quant_dtype == "nvfp4" and quant_config.is_scale_swizzled:
        assert a1q_scale is not None
        if a1q_scale.element_size() == 1:
            a1q_scale = a1q_scale.view(torch.uint8)
        a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

    return a1q_scale


class MoEPrepareAndFinalizeNaiveDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the topk weights and ids.
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers
        self._route_after_gather = envs.VLLM_MOE_AG_RS_ROUTE_AFTER_GATHER
        if self._route_after_gather and self.is_sequence_parallel:
            raise ValueError(
                "VLLM_MOE_AG_RS_ROUTE_AFTER_GATHER currently supports only "
                "TP=1, non-sequence-parallel DP/EP"
            )
        # Set by FusedMoEWithLoRA.set_mapping() when LoRA is active. When
        # present, prepare() dispatches the per-token LoRA mapping alongside
        # hidden_states and writes the gathered result back to the context so
        # experts can use the per-rank-local mapping.
        self._lora_context = None

    def set_lora_context(self, ctx) -> None:
        self._lora_context = ctx

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    @staticmethod
    def _dp_layout() -> tuple[object, list[int]]:
        group = get_dp_group()
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        return group, sizes

    def dispatch_before_routing(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        if not self._route_after_gather:
            return None
        if self._lora_context is not None:
            raise NotImplementedError(
                "route-after-gather ablation does not support MoE LoRA"
            )
        group, sizes = self._dp_layout()
        assert sizes[group.rank_in_group] == hidden_states.shape[0]
        return group.all_gatherv(hidden_states, dim=0, sizes=sizes)

    def allocate_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._route_after_gather:
            return super().allocate_output(hidden_states)
        group, sizes = self._dp_layout()
        local_rows = sizes[group.rank_in_group]
        return torch.empty_like(hidden_states[:local_rows])

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """Quantize and Dispatch Topk Weights and Topk Ids."""

        hook, receiver = self._prepare(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=defer_input_quant,
            do_async=False,
        )
        hook()
        return receiver()

    def supports_async(self) -> bool:
        return (
            not self._route_after_gather
            and get_ep_group().supports_dispatch_combine_async()
        )

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable, mk.ReceiverType]:
        return self._prepare(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=defer_input_quant,
            do_async=True,
        )

    def _prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
        do_async: bool,
    ) -> tuple[Callable, mk.ReceiverType]:
        """Quantize and start dispatch, optionally deferring its receive."""

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        # When LoRA is active, dispatch the per-token LoRA id along with
        # hidden_states so every rank receives the correct mapping for the
        # tokens it ends up processing. The punica_wrapper stores indices as
        # int64 but the moe_lora_align_block_size kernel expects int32, so
        # pull the pre-cast view from token_mapping_meta.
        lora_ctx = self._lora_context
        local_token_lora_mapping = None
        if lora_ctx is not None:
            local_token_lora_mapping = (
                lora_ctx.punica_wrapper.token_mapping_meta.token_lora_mapping[
                    : a1.shape[0]
                ]
            )

        if self._route_after_gather:
            if local_token_lora_mapping is not None:
                raise NotImplementedError(
                    "route-after-gather ablation does not support MoE LoRA"
                )
            # The runner already gathered activations, then ran the replicated
            # router on those global rows. Dynamic activation scales, when
            # present, were therefore also computed from global rows here.
            return _no_wait(
                (a1q, a1q_scale_orig, None, topk_ids, topk_weights)
            )

        extra_tensors: list[torch.Tensor] | None = None
        if scales is not None:
            extra_tensors = list(scales)
        if local_token_lora_mapping is not None:
            if extra_tensors is None:
                extra_tensors = []
            extra_tensors.append(local_token_lora_mapping)

        group = get_ep_group()
        dispatch_args = (
            a1q,
            topk_weights,
            topk_ids,
        )
        dispatch_kwargs = {
            "is_sequence_parallel": self.is_sequence_parallel,
            "extra_tensors": extra_tensors,
        }
        if do_async:
            hook, receive_dispatch = group.dispatch_async(
                *dispatch_args, **dispatch_kwargs
            )
        else:
            hook, receive_dispatch = _no_wait(
                group.dispatch(*dispatch_args, **dispatch_kwargs)
            )

        def receiver() -> mk.PrepareResultType:
            res = receive_dispatch()
            if extra_tensors is None:
                assert len(res) == 3
                recv_a1q, recv_topk_weights, recv_topk_ids = res
                recv_a1q_scale = a1q_scale_orig
            else:
                assert len(res) == 4
                recv_a1q, recv_topk_weights, recv_topk_ids, gathered_extras = res
                gathered_extras = list(gathered_extras)
                if local_token_lora_mapping is not None:
                    dispatched_lora_mapping = gathered_extras.pop()
                    assert lora_ctx is not None
                    lora_ctx.local_token_lora_mapping = dispatched_lora_mapping
                if scales is not None:
                    recv_a1q_scale = _unwrap_scale_and_prepare_for_moe(
                        gathered_extras, quant_config
                    )
                else:
                    recv_a1q_scale = a1q_scale_orig
            return (
                recv_a1q,
                recv_a1q_scale,
                None,
                recv_topk_ids,
                recv_topk_weights,
            )

        return hook, receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        hook, receiver = self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False,
        )
        hook()
        receiver()

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable]:
        return self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=True,
        )

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> tuple[Callable, Callable]:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        # Drain the dispatch wait before starting combine. Under DBO this hook
        # belongs to the other microbatch; omitting it can silently overwrite
        # the only dependency protecting the receive buffer.
        dbo_maybe_run_recv_hook()
        group = get_ep_group()
        if do_async:
            hook, receive_combine = group.combine_async(
                out, is_sequence_parallel=self.is_sequence_parallel
            )
        else:
            hook, receive_combine = _no_wait(
                group.combine(out, is_sequence_parallel=self.is_sequence_parallel)
            )
        return hook, lambda: output.copy_(receive_combine())


class MoEPrepareAndFinalizeNaiveDPEPMonolithic(mk.FusedMoEPrepareAndFinalizeMonolithic):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the router logits (the MoE kernel runs the router internally).
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType:
        """Quantize and Dispatch Router Logits."""

        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        res = get_ep_group().dispatch_router_logits(
            a1q,
            router_logits,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            assert len(res) == 2
            a1q, router_logits = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 3
            a1q, router_logits, scales = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(scales, quant_config)

        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        out = get_ep_group().combine(
            fused_expert_output, is_sequence_parallel=self.is_sequence_parallel
        )
        return out


def make_moe_prepare_and_finalize_naive_dp_ep(
    use_monolithic: bool,
    is_sequence_parallel: bool = False,
    num_dispatchers: int = 1,
) -> MoEPrepareAndFinalizeNaiveDPEPModular | MoEPrepareAndFinalizeNaiveDPEPMonolithic:
    return (
        MoEPrepareAndFinalizeNaiveDPEPMonolithic(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
        if use_monolithic
        else MoEPrepareAndFinalizeNaiveDPEPModular(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
    )
