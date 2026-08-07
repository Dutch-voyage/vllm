# SPDX-License-Identifier: Apache-2.0
"""Experimental exact-size copy-engine all-to-all for single-host DP/EP MoE.

The data plane is intentionally narrow: TP1, linear EP placement, FP16
activations, and FP32 router weights.  Calls below the admission threshold (as
well as unsupported calls) use vLLM's existing NCCL AG/RS manager.  Long
prefill chunks may use either eager native-proxy CE control or a captured,
device-only symmetric-memory P2P path.
"""

from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import record_function

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .all2all import AgRsAll2AllManager
from .base_device_communicator import All2AllManagerBase


logger = init_logger(__name__)


# The integer handle is embedded as a constant in the compiled graph.  The
# manager owns the scheduler lifetime; this registry only lets the opaque
# custom op recover it when torch.compile or CUDAGraphWrapper invokes the op.
_GRAPH_CONTROLS: dict[int, Any] = {}


def _ce_a2a_exchange(
    send: torch.Tensor,
    send_counts: torch.Tensor,
    recv: torch.Tensor,
    recv_counts: torch.Tensor,
    control_handle: int,
    phase: int,
) -> None:
    control = _GRAPH_CONTROLS.get(control_handle)
    if control is None:
        raise RuntimeError("CE A2A graph control handle is no longer registered")
    control.submit(
        "dispatch" if phase == 0 else "combine",
        send,
        send_counts,
        recv,
        recv_counts,
    )


def _ce_a2a_exchange_fake(
    send: torch.Tensor,
    send_counts: torch.Tensor,
    recv: torch.Tensor,
    recv_counts: torch.Tensor,
    control_handle: int,
    phase: int,
) -> None:
    del send, send_counts, recv, recv_counts, control_handle, phase


direct_register_custom_op(
    op_name="ce_a2a_exchange_",
    op_func=_ce_a2a_exchange,
    mutates_args=["recv", "recv_counts"],
    fake_impl=_ce_a2a_exchange_fake,
)


def _positive_value(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _codec_bits(name: str, value: int) -> int:
    """Return a supported activation width, where zero means uncompressed."""

    value = int(value)
    if value and value not in (4, 5, 6, 8):
        raise ValueError(f"{name} must be 0 (FP16) or one of 4, 5, 6, 8; got {value}")
    return value


def _global_num_experts(hf_config: Any) -> int:
    for name in ("num_experts", "n_routed_experts", "num_local_experts"):
        value = getattr(hf_config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("CE A2A could not determine the model's global expert count")


class _SkewRecorder:
    """Log per-call destination row counts without adding a device sync.

    The counts already live on device, so recording one is a small
    device-to-device copy into a preallocated ring. Nothing crosses to the host
    until teardown, which keeps the instrument off the critical path it is
    meant to characterize.
    """

    def __init__(self, path: str, *, rank: int, world_size: int, capacity: int) -> None:
        self.path = path
        self.rank = rank
        self.capacity = capacity
        self.counts = torch.zeros(
            (capacity, world_size),
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.calls = 0
        self.dropped = 0
        atexit.register(self.flush)

    def record(self, send_counts: torch.Tensor) -> None:
        if self.calls < self.capacity:
            self.counts[self.calls].copy_(send_counts, non_blocking=True)
            self.calls += 1
        else:
            self.dropped += 1

    def summary(self) -> dict[str, Any]:
        """Reduce the recorded counts to per-call skew statistics.

        This rides out through ``diagnostics()``, which the harness actively
        collects while the process is alive. The file dump below is a fallback
        only: engine cores are frequently hard-killed at shutdown, and
        ``atexit`` does not survive that.
        """

        if self.calls == 0:
            return {"recorded_calls": 0}
        counts = self.counts[: self.calls].float()
        totals = counts.sum(dim=1)
        alive = totals > 0
        if not bool(alive.any()):
            return {"recorded_calls": self.calls, "nonempty_calls": 0}
        counts, totals = counts[alive], totals[alive]
        ratio = counts.max(dim=1).values / (totals / counts.shape[1])
        order = torch.argsort(ratio)
        percentile = ratio[order[int(0.95 * (len(order) - 1))]]
        return {
            "recorded_calls": self.calls,
            "nonempty_calls": int(alive.sum()),
            "dropped_calls": self.dropped,
            "skew_mean": float(ratio.mean()),
            "skew_p50": float(ratio.median()),
            "skew_p95": float(percentile),
            "skew_max": float(ratio.max()),
            "rows_per_call_mean": float(totals.mean()),
            "heaviest_destination_share": [
                float(share)
                for share in (
                    counts.argmax(dim=1).bincount(minlength=counts.shape[1]).float()
                    / counts.shape[0]
                )
            ],
        }

    def flush(self) -> None:
        if self.calls == 0:
            return
        recorded, self.calls = self.calls, 0
        try:
            torch.save(
                {
                    "rank": self.rank,
                    "calls": recorded,
                    "dropped": self.dropped,
                    "counts": self.counts[:recorded].cpu(),
                },
                f"{self.path}.rank{self.rank}.pt",
            )
        except Exception:  # teardown races CUDA shutdown; a lost log is not fatal
            logger.warning("CE A2A skew log could not be written", exc_info=True)


def _resample_topk_ids(
    topk_ids: torch.Tensor,
    *,
    global_num_experts: int,
    experts_per_rank: int,
    world_size: int,
    bias: float,
    rolled: bool,
) -> torch.Tensor:
    """Redraw each token's expert set with controllable destination skew.

    Gumbel top-k over uniform noise gives a uniform random k-subset, which is
    destination-balanced by construction while preserving the per-token fan-out.
    Concentrating a token's experts onto one rank raises the skew, but it also
    lowers how many ranks that token reaches at all — so skew and wire volume
    move together and a naive sweep cannot tell them apart.

    ``rolled`` is the matched-volume control. It applies the identical
    concentration but favours a *random* rank per token, so per-token fan-out and
    therefore total bytes are unchanged, while the destination distribution comes
    out balanced by symmetry. A ``fixed`` and a ``rolled`` arm at the same bias
    differ in skew and in nothing else.

    Every arm pays the resampling cost, including the balanced ones, so it
    cancels out of the comparison rather than confounding it.
    """

    rows, top_k = topk_ids.shape
    device = topk_ids.device
    uniform = torch.rand((rows, global_num_experts), device=device, dtype=torch.float32)
    exponential = -torch.log(uniform.clamp_min(1e-20))
    scores = -torch.log(exponential.clamp_min(1e-20))
    if bias:
        destination = (
            torch.arange(global_num_experts, device=device) // experts_per_rank
        )
        favoured = (
            torch.randint(world_size, (rows, 1), device=device)
            if rolled
            else torch.zeros((rows, 1), dtype=torch.long, device=device)
        )
        shed = -bias / max(1, world_size - 1)
        scores = scores + torch.where(
            destination.unsqueeze(0) == favoured,
            bias,
            shed,
        )
    return torch.topk(scores, top_k, dim=1).indices.to(topk_ids.dtype)


@dataclass
class _CeExchange:
    send_counts: tuple[int, ...] | torch.Tensor
    recv_counts: tuple[int, ...] | torch.Tensor
    owner_token_ids: torch.Tensor | None
    token_positions: torch.Tensor | None
    local_rows: int
    fixed_control: bool = False


class CeA2AAll2AllManager(All2AllManagerBase):
    """Select CE-A2A for large prefill and NCCL AG/RS for small calls."""

    def __init__(
        self,
        cpu_group: Any,
        tcp_store_group: Any = None,
        *,
        device_group: Any,
    ) -> None:
        super().__init__(cpu_group, tcp_store_group)
        if device_group is None:
            raise ValueError("CE A2A requires an EP CUDA process group")
        self.device_group = device_group
        self.fallback = AgRsAll2AllManager(cpu_group, tcp_store_group)

        config = get_current_vllm_config()
        parallel = config.parallel_config
        if self.tp_group.world_size != 1:
            raise ValueError("CE A2A v1 supports TP1 only")
        if self.dp_world_size != self.world_size:
            raise ValueError("CE A2A v1 requires the EP group to equal the DP group")
        if parallel.enable_eplb or parallel.expert_placement_strategy != "linear":
            raise ValueError("CE A2A v1 requires linear expert placement without EPLB")

        self.global_num_experts = _global_num_experts(config.model_config.hf_config)
        if self.global_num_experts % self.world_size:
            raise ValueError("global experts must divide evenly across CE A2A ranks")
        self.experts_per_rank = self.global_num_experts // self.world_size
        self.min_global_rows = _positive_value(
            "VLLM_CE_A2A_MIN_ROWS", envs.VLLM_CE_A2A_MIN_ROWS
        )
        self.max_edge_rows = _positive_value(
            "VLLM_CE_A2A_MAX_EDGE_ROWS",
            envs.VLLM_CE_A2A_MAX_EDGE_ROWS,
        )
        self.packet_builder_kind = envs.VLLM_CE_A2A_PACKET_BUILDER
        if self.packet_builder_kind not in ("reference", "fixed", "fused"):
            raise ValueError(
                "VLLM_CE_A2A_PACKET_BUILDER must be reference, fixed, or fused"
            )
        self.control_kind = envs.VLLM_CE_A2A_CONTROL
        if self.control_kind not in (
            "host_sync",
            "device_proxy",
            "native_proxy",
            "graph_proxy",
        ):
            raise ValueError(
                "VLLM_CE_A2A_CONTROL must be host_sync, device_proxy, or "
                "native_proxy, or graph_proxy"
            )
        if (
            self.control_kind != "host_sync"
            and self.packet_builder_kind != "fused"
        ):
            raise ValueError("proxy control requires the fused packet builder")
        self.dispatch_bits = _codec_bits(
            "VLLM_CE_A2A_DISPATCH_BITS", envs.VLLM_CE_A2A_DISPATCH_BITS
        )
        self.combine_bits = _codec_bits(
            "VLLM_CE_A2A_COMBINE_BITS", envs.VLLM_CE_A2A_COMBINE_BITS
        )
        self.codec_group_size = _positive_value(
            "VLLM_CE_A2A_CODEC_GROUP", envs.VLLM_CE_A2A_CODEC_GROUP
        )
        if (self.dispatch_bits or self.combine_bits) and (
            self.packet_builder_kind != "fused"
        ):
            raise ValueError("activation compression requires the fused packet builder")
        self.scheduler = envs.VLLM_CE_A2A_SCHEDULER
        if self.scheduler not in (
            "edge_credits",
            "edge_stream_memops",
            "cyclic_barrier",
        ):
            raise ValueError(
                "VLLM_CE_A2A_SCHEDULER must be edge_credits, "
                "edge_stream_memops, or cyclic_barrier"
            )

        # Experiment-only knobs, read straight from the environment so the
        # sweep does not need a config surface it will never ship with.
        self.skew_bias = float(os.environ.get("VLLM_CE_A2A_SKEW_BIAS", "0") or 0.0)
        self.skew_resample = bool(
            int(os.environ.get("VLLM_CE_A2A_SKEW_RESAMPLE", "0") or 0)
        )
        skew_mode = os.environ.get("VLLM_CE_A2A_SKEW_MODE", "fixed").strip() or "fixed"
        if skew_mode not in ("fixed", "rolled"):
            raise ValueError("VLLM_CE_A2A_SKEW_MODE must be fixed or rolled")
        self.skew_rolled = skew_mode == "rolled"
        skew_log = os.environ.get("VLLM_CE_A2A_SKEW_LOG", "").strip()
        self.skew_recorder = (
            _SkewRecorder(
                skew_log,
                rank=self.rank,
                world_size=self.world_size,
                capacity=int(
                    os.environ.get("VLLM_CE_A2A_SKEW_LOG_CAPACITY", "65536") or 65536
                ),
            )
            if skew_log
            else None
        )

        self.transport: Any | None = None
        self.packet_spec: Any | None = None
        self.packet_builder: Any | None = None
        self.control: Any | None = None
        self._control_handle: int | None = None
        self.recv_counts_device: torch.Tensor | None = None
        self.dispatch_recv_blocks: torch.Tensor | None = None
        self.fixed_recv_hidden: torch.Tensor | None = None
        self.fixed_recv_ids: torch.Tensor | None = None
        self.fixed_recv_weights: torch.Tensor | None = None
        self.combine_recv_blocks: torch.Tensor | None = None
        self.combine_send_blocks: torch.Tensor | None = None
        self._active: _CeExchange | str | None = None
        self.ce_dispatch_calls = 0
        self.ce_combine_calls = 0
        self.nccl_dispatch_calls = 0
        self.nccl_combine_calls = 0
        self.capacity_fallbacks = 0
        self.unsupported_fallbacks = 0
        logger.info_once(
            "CE A2A enabled for global rows >= %d "
            "(edge capacity=%d, scheduler=%s, packet_builder=%s, control=%s, "
            "dispatch=%s, combine=%s); smaller calls use NCCL AG/RS",
            self.min_global_rows,
            self.max_edge_rows,
            self.scheduler,
            self.packet_builder_kind,
            self.control_kind,
            f"int{self.dispatch_bits}" if self.dispatch_bits else "fp16",
            f"int{self.combine_bits}" if self.combine_bits else "fp16",
        )
        if self.skew_resample or self.skew_recorder is not None:
            logger.info_once(
                "CE A2A skew instrumentation: resample=%s mode=%s bias=%.3f log=%s",
                self.skew_resample,
                skew_mode,
                self.skew_bias,
                skew_log or "off",
            )

    def _get_comm_group(self, is_sequence_parallel: bool) -> Any:
        return self.fallback._get_comm_group(is_sequence_parallel)

    def _get_sizes(self, num_local_tokens: int, comm_group: Any) -> list[int]:
        return self.fallback._get_sizes(num_local_tokens, comm_group)

    def _fallback_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool,
        extra_tensors: list[torch.Tensor] | None,
        *,
        unsupported: bool = False,
    ):
        self._active = "nccl"
        self.nccl_dispatch_calls += 1
        self.unsupported_fallbacks += int(unsupported)
        return self.fallback.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors,
        )

    def _ensure_transport(self, hidden_size: int, top_k: int) -> None:
        from ce_a2a_moe import (
            CoalescedCeTransport,
            DeviceCountCeScheduler,
            DispatchPacketSpec,
            FixedBlockDispatchBuilder,
            FusedBlockDispatchBuilder,
            GraphP2PA2AScheduler,
            NativeDeviceCountCeScheduler,
            lowbit_block_payload_bytes,
        )

        if self.transport is not None:
            assert self.packet_spec is not None
            if (
                self.packet_spec.hidden_size != hidden_size
                or self.packet_spec.top_k != top_k
            ):
                raise ValueError("CE A2A v1 supports one MoE packet shape per model")
            return
        self.packet_spec = DispatchPacketSpec(
            hidden_size=hidden_size,
            top_k=top_k,
            activation_bits=self.dispatch_bits or 16,
            group_size=self.codec_group_size,
        )
        self.combine_row_bytes = (
            lowbit_block_payload_bytes(
                hidden_size, self.combine_bits, self.codec_group_size
            )
            if self.combine_bits
            else hidden_size * 2
        )
        if self.packet_builder_kind in ("fixed", "fused"):
            builder_type = (
                FusedBlockDispatchBuilder
                if self.packet_builder_kind == "fused"
                else FixedBlockDispatchBuilder
            )
            self.packet_builder = builder_type(
                spec=self.packet_spec,
                experts_per_rank=self.experts_per_rank,
                world_size=self.world_size,
                max_edge_rows=self.max_edge_rows,
                device=torch.cuda.current_device(),
            )
        self.transport = CoalescedCeTransport(
            max_edge_rows=self.max_edge_rows,
            dispatch_row_bytes=self.packet_spec.packet_bytes,
            hidden_size=hidden_size,
            combine_row_bytes=self.combine_row_bytes,
            scheduler=self.scheduler,
            process_group=self.device_group,
        )
        if self.control_kind != "host_sync":
            block_rows = self.max_edge_rows + 1
            total_rows = self.world_size * block_rows
            self.recv_counts_device = torch.empty(
                self.world_size,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            self.dispatch_recv_blocks = torch.empty(
                (self.world_size, block_rows, self.packet_spec.packet_bytes),
                dtype=torch.uint8,
                device=torch.cuda.current_device(),
            )
            self.fixed_recv_hidden = torch.empty(
                (total_rows, hidden_size),
                dtype=torch.float16,
                device=torch.cuda.current_device(),
            )
            self.fixed_recv_ids = torch.empty(
                (total_rows, top_k),
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            self.fixed_recv_weights = torch.empty(
                (total_rows, top_k),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            if self.combine_bits:
                # Compression needs a staging arena on the send side: the
                # uncompressed path hands the expert output tensor straight to
                # the transport, and that tensor is no longer the wire layout.
                self.combine_send_blocks = torch.empty(
                    (self.world_size, block_rows, self.combine_row_bytes),
                    dtype=torch.uint8,
                    device=torch.cuda.current_device(),
                )
                self.combine_recv_blocks = torch.empty_like(self.combine_send_blocks)
            else:
                self.combine_recv_blocks = torch.empty(
                    (self.world_size, block_rows, hidden_size),
                    dtype=torch.float16,
                    device=torch.cuda.current_device(),
                )
            proxy_cpu_base = envs.VLLM_CE_A2A_PROXY_CPU
            if self.control_kind == "graph_proxy":
                self.control = GraphP2PA2AScheduler(self.transport)
            else:
                control_type = (
                    NativeDeviceCountCeScheduler
                    if self.control_kind == "native_proxy"
                    else DeviceCountCeScheduler
                )
                self.control = control_type(
                    self.transport,
                    ring_depth=envs.VLLM_CE_A2A_CONTROL_RING_DEPTH,
                    submission_window=envs.VLLM_CE_A2A_PROXY_WINDOW,
                    proxy_cpu=(
                        None
                        if proxy_cpu_base < 0
                        else proxy_cpu_base + self.rank
                    ),
                )
            if self.control_kind == "graph_proxy":
                self._control_handle = id(self.control)
                _GRAPH_CONTROLS[self._control_handle] = self.control

    def _submit_fixed_exchange(
        self,
        phase: int,
        send: torch.Tensor,
        send_counts: torch.Tensor,
        recv: torch.Tensor,
        recv_counts: torch.Tensor,
    ) -> None:
        assert self.control is not None
        if self.control_kind == "graph_proxy":
            assert self._control_handle is not None
            torch.ops.vllm.ce_a2a_exchange_(
                send,
                send_counts,
                recv,
                recv_counts,
                self._control_handle,
                phase,
            )
            return
        self.control.submit(
            "dispatch" if phase == 0 else "combine",
            send,
            send_counts,
            recv,
            recv_counts,
        )

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        # The CE packet ABI consumes post-routing top-k metadata.  Monolithic
        # router-logit dispatch remains on the unmodified NCCL reference path.
        self._active = "nccl"
        self.nccl_dispatch_calls += 1
        self.unsupported_fallbacks += 1
        return self.fallback.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        if self._active is not None:
            raise RuntimeError("CE A2A dispatch called before the prior combine")

        comm_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(int(hidden_states.shape[0]), comm_group)
        global_rows = sum(int(value) for value in sizes)
        unsupported = bool(
            is_sequence_parallel
            or extra_tensors is not None
            or hidden_states.dtype != torch.float16
            or topk_weights.dtype != torch.float32
            or hidden_states.ndim != 2
            or topk_ids.ndim != 2
        )
        if unsupported or global_rows < self.min_global_rows:
            return self._fallback_dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
                unsupported=unsupported,
            )
        if max(sizes, default=0) > self.max_edge_rows:
            self.capacity_fallbacks += 1
            return self._fallback_dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
            )

        from ce_a2a_moe import (
            build_dispatch_packets,
            compact_fixed_block_rows,
            dispatch_packet_views,
            unpack_fixed_dispatch_blocks,
        )

        hidden_size = int(hidden_states.shape[1])
        top_k = int(topk_ids.shape[1])
        self._ensure_transport(hidden_size, top_k)
        assert self.transport is not None and self.packet_spec is not None

        if self.skew_resample:
            with record_function("moe.ce_a2a.skew_resample"):
                topk_ids = _resample_topk_ids(
                    topk_ids,
                    global_num_experts=self.global_num_experts,
                    experts_per_rank=self.experts_per_rank,
                    world_size=self.world_size,
                    bias=self.skew_bias,
                    rolled=self.skew_rolled,
                )

        with record_function("moe.ce_a2a.coalesce_pack"):
            if self.packet_builder_kind in ("fixed", "fused"):
                assert self.packet_builder is not None
                coalesced = self.packet_builder.build(
                    hidden_states,
                    topk_ids,
                    topk_weights,
                )
            else:
                coalesced = build_dispatch_packets(
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    experts_per_rank=self.experts_per_rank,
                    world_size=self.world_size,
                    spec=self.packet_spec,
                )
        if self.skew_recorder is not None:
            self.skew_recorder.record(coalesced.send_counts)

        with record_function("moe.ce_a2a.count_exchange_d2h"):
            recv_counts_device = (
                self.recv_counts_device
                if self.control_kind != "host_sync"
                else torch.empty_like(coalesced.send_counts)
            )
            assert recv_counts_device is not None
            if self.control_kind != "graph_proxy":
                with record_function("moe.ce_a2a.count_exchange_gpu"):
                    dist.all_to_all_single(
                        recv_counts_device,
                        coalesced.send_counts,
                        group=self.device_group,
                    )
            if self.control_kind != "host_sync":
                assert self.control is not None
                assert self.dispatch_recv_blocks is not None
                with record_function("moe.ce_a2a.dispatch_control"):
                    self._submit_fixed_exchange(
                        0,
                        coalesced.packets,
                        coalesced.send_counts,
                        self.dispatch_recv_blocks,
                        recv_counts_device,
                    )
                send_counts = coalesced.send_counts
                recv_counts = recv_counts_device
                owner_token_ids = None
            else:
                count_pair = torch.stack(
                    (coalesced.send_counts, recv_counts_device),
                    dim=0,
                ).cpu()
                send_counts = tuple(int(value) for value in count_pair[0].tolist())
                recv_counts = tuple(int(value) for value in count_pair[1].tolist())
                owner_token_ids = (
                    compact_fixed_block_rows(
                        coalesced.token_ids,
                        send_counts,
                    )
                    if self.packet_builder_kind in ("fixed", "fused")
                    else coalesced.token_ids
                )

        if self.control_kind != "host_sync":
            assert self.dispatch_recv_blocks is not None
            assert self.fixed_recv_hidden is not None
            assert self.fixed_recv_ids is not None
            assert self.fixed_recv_weights is not None
            assert coalesced.token_positions is not None
            with record_function("moe.ce_a2a.unpack"):
                recv_hidden, recv_topk_ids, recv_topk_weights = (
                    unpack_fixed_dispatch_blocks(
                        self.dispatch_recv_blocks,
                        recv_counts_device,
                        spec=self.packet_spec,
                        expert_rank=self.rank,
                        experts_per_rank=self.experts_per_rank,
                        output_hidden=self.fixed_recv_hidden,
                        output_ids=self.fixed_recv_ids,
                        output_weights=self.fixed_recv_weights,
                    )
                )
            self._active = _CeExchange(
                send_counts=send_counts,
                recv_counts=recv_counts,
                owner_token_ids=None,
                token_positions=coalesced.token_positions,
                local_rows=int(hidden_states.shape[0]),
                fixed_control=True,
            )
            self.ce_dispatch_calls += 1
            return recv_hidden, recv_topk_weights, recv_topk_ids

        recv_packets = torch.empty(
            (sum(recv_counts), self.packet_spec.packet_bytes),
            dtype=torch.uint8,
            device=hidden_states.device,
        )
        with record_function("moe.ce_a2a.dispatch_transport"):
            self.transport.dispatch(
                coalesced.packets,
                send_counts,
                recv_packets,
                recv_counts,
            )
        with record_function("moe.ce_a2a.unpack"):
            recv_hidden, local_topk_ids, recv_topk_weights = dispatch_packet_views(
                recv_packets, self.packet_spec
            )
            # The wire ABI is array-of-struct: packet metadata follows every
            # activation row.  Consequently these zero-copy field views have a
            # packet-sized leading stride.  Marlin requires dense row-major
            # hidden states, and the generic weight/reduce path expects dense
            # route rows.
            recv_hidden = recv_hidden.contiguous()
            recv_topk_weights = recv_topk_weights.contiguous()
            # vLLM's expert_map consumes global IDs and expects every ID to be
            # a valid global expert. Zero-weight routes that do not belong to
            # this rank are mapped outside this rank, following DeepEP.
            invalid_global_id = self.global_num_experts - 1 if self.rank == 0 else 0
            recv_topk_ids = torch.where(
                local_topk_ids == -1,
                invalid_global_id,
                local_topk_ids + self.rank * self.experts_per_rank,
            )
        self._active = _CeExchange(
            send_counts=send_counts,
            recv_counts=recv_counts,
            owner_token_ids=owner_token_ids,
            token_positions=None,
            local_rows=int(hidden_states.shape[0]),
        )
        self.ce_dispatch_calls += 1
        return recv_hidden, recv_topk_weights, recv_topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        active = self._active
        if active is None:
            raise RuntimeError(
                "CE A2A combine has no matching dispatch; route-after-gather is "
                "not supported with the ce_a2a backend"
            )
        if active == "nccl":
            self._active = None
            self.nccl_combine_calls += 1
            return self.fallback.combine(hidden_states, is_sequence_parallel)
        assert isinstance(active, _CeExchange)
        if is_sequence_parallel:
            raise RuntimeError("CE A2A state cannot be combined as sequence parallel")
        if hidden_states.dtype != torch.float16 or hidden_states.ndim != 2:
            raise ValueError("CE A2A combine requires a rank-2 FP16 tensor")

        from ce_a2a_moe import (
            quantize_pack_combine_blocks,
            reduce_fixed_owner_partials,
            reduce_owner_partials,
            reduce_packed_owner_partials,
        )

        assert self.transport is not None
        if active.fixed_control:
            assert self.control is not None
            assert self.combine_recv_blocks is not None
            assert isinstance(active.send_counts, torch.Tensor)
            assert isinstance(active.recv_counts, torch.Tensor)
            assert active.token_positions is not None
            block_rows = self.max_edge_rows + 1
            expected_rows = self.world_size * block_rows
            if int(hidden_states.shape[0]) != expected_rows:
                raise ValueError(
                    "device-proxy expert output does not match fixed row capacity"
                )
            hidden_size = int(hidden_states.shape[1])
            blocks = hidden_states.reshape(self.world_size, block_rows, hidden_size)
            if self.combine_bits:
                assert self.combine_send_blocks is not None
                with record_function("moe.ce_a2a.combine_pack"):
                    blocks = quantize_pack_combine_blocks(
                        blocks,
                        self.combine_send_blocks,
                        bits=self.combine_bits,
                        group_size=self.codec_group_size,
                    )
            with record_function("moe.ce_a2a.combine_transport"):
                with record_function("moe.ce_a2a.combine_control"):
                    self._submit_fixed_exchange(
                        1,
                        blocks,
                        active.recv_counts,
                        self.combine_recv_blocks,
                        active.send_counts,
                    )
            with record_function("moe.ce_a2a.owner_reduce"):
                if self.combine_bits:
                    output = reduce_packed_owner_partials(
                        self.combine_recv_blocks,
                        active.token_positions,
                        local_rows=active.local_rows,
                        hidden_size=hidden_size,
                        bits=self.combine_bits,
                        group_size=self.codec_group_size,
                    )
                else:
                    output = reduce_fixed_owner_partials(
                        self.combine_recv_blocks,
                        active.token_positions,
                        local_rows=active.local_rows,
                    )
            self._active = None
            self.ce_combine_calls += 1
            return output

        assert isinstance(active.send_counts, tuple)
        assert isinstance(active.recv_counts, tuple)
        assert active.owner_token_ids is not None
        owner_partials = torch.empty(
            (sum(active.send_counts), int(hidden_states.shape[1])),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        with record_function("moe.ce_a2a.combine_transport"):
            self.transport.combine(
                hidden_states,
                active.recv_counts,
                owner_partials,
                active.send_counts,
            )
        with record_function("moe.ce_a2a.owner_reduce"):
            output = reduce_owner_partials(
                owner_partials,
                active.owner_token_ids,
                local_rows=active.local_rows,
            )
        self._active = None
        self.ce_combine_calls += 1
        return output

    def diagnostics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "backend": "ce_a2a",
            "min_global_rows": self.min_global_rows,
            "max_edge_rows": self.max_edge_rows,
            "packet_builder": self.packet_builder_kind,
            "control": self.control_kind,
            "dispatch_bits": self.dispatch_bits or 16,
            "combine_bits": self.combine_bits or 16,
            "codec_group_size": self.codec_group_size,
            "ce_dispatch_calls": self.ce_dispatch_calls,
            "ce_combine_calls": self.ce_combine_calls,
            "nccl_dispatch_calls": self.nccl_dispatch_calls,
            "nccl_combine_calls": self.nccl_combine_calls,
            "capacity_fallbacks": self.capacity_fallbacks,
            "unsupported_fallbacks": self.unsupported_fallbacks,
            "transport_initialized": self.transport is not None,
        }
        if self.transport is not None:
            result["transport"] = self.transport.diagnostics()
        if self.control is not None:
            result["device_control"] = self.control.diagnostics()
        result["skew"] = {
            "resample": self.skew_resample,
            "mode": "rolled" if self.skew_rolled else "fixed",
            "bias": self.skew_bias,
        }
        if self.skew_recorder is not None:
            result["skew"].update(self.skew_recorder.summary())
        return result

    def destroy(self) -> None:
        if self._control_handle is not None:
            _GRAPH_CONTROLS.pop(self._control_handle, None)
            self._control_handle = None
        if self.control is not None:
            self.control.close()
            self.control = None
        self.transport = None
        self.packet_spec = None
        self.packet_builder = None
        self.recv_counts_device = None
        self.dispatch_recv_blocks = None
        self.fixed_recv_hidden = None
        self.fixed_recv_ids = None
        self.fixed_recv_weights = None
        self.combine_recv_blocks = None
        self.combine_send_blocks = None
        self._active = None
