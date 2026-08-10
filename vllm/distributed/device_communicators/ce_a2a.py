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


def _ladder_views(arena: torch.Tensor, value_bytes: int):
    """Split one fixed-stride arena into the pair of tensors the ladder wants.

    The codec addresses codes by row pitch and scales as ``[world, rows,
    groups]``, but the transport moves exactly one buffer per phase, so both have
    to live inside it: each row is its codes followed by its scales. These are
    views, not copies -- the arena itself is what goes on the wire.
    """

    world, rows, row_bytes = (int(dim) for dim in arena.shape)
    return (
        arena.view(world, rows * row_bytes),
        arena[:, :, value_bytes:].view(torch.float16),
    )


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
        self.delta_dispatch = bool(
            int(os.environ.get("VLLM_CE_A2A_DELTA_DISPATCH", "0") or 0)
        )
        # How many dispatches make one forward pass. Token indices only name the
        # same token within a pass, so the references are dropped when one ends.
        # Getting this wrong costs compression on the first layer of a pass, not
        # correctness: both sides read the same stale reference and stay in step.
        self.delta_period = int(
            os.environ.get("VLLM_CE_A2A_DELTA_PERIOD", "0") or 0
        )
        # How much flatter the residual has to be before the sender prefers it.
        # One takes delta whenever it is better at this layer; zero never takes
        # it. Below one the sender declines marginal wins, on the theory that a
        # residual buys little accuracy but inherits the reference's structure.
        self.delta_margin = float(
            os.environ.get("VLLM_CE_A2A_DELTA_MARGIN", "1.0") or 1.0
        )
        if self.delta_dispatch and not 0.0 <= self.delta_margin <= 1.0:
            raise ValueError("VLLM_CE_A2A_DELTA_MARGIN scales a ratio: [0, 1]")
        if self.delta_dispatch and not self.dispatch_bits:
            raise ValueError("delta dispatch codes a residual, so it needs a width")
        if self.delta_dispatch and self.packet_builder_kind != "fused":
            raise ValueError("delta dispatch requires the fused packet builder")
        self.delta_probe = bool(
            int(os.environ.get("VLLM_CE_A2A_DELTA_PROBE", "0") or 0)
        )
        self.delta_span_ratio: list[float] = []
        self._probe_state: tuple = ()
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
        # Fractional combine width: promote the first g of each row's groups to
        # base+1 bits, giving an average of base + g/groups. The width is uniform
        # across destinations here, which is what lets the row pitch stay fixed
        # and the transport stay untouched; a per-destination g would need the
        # segmented exchange. Zero keeps the plain integer codec.
        self.combine_ladder_g = int(
            os.environ.get("VLLM_CE_A2A_COMBINE_LADDER_G", "0") or 0
        )
        # Whole-bit-per-layer allocation: run `low` layers out of every `period`
        # at the base width and the rest a bit above it. E34 found the per-row
        # ladder sits inside the frontier its two rails define, and predicted
        # that spending the bit per layer instead lands on that line -- this is
        # the knob that tests the prediction. Expressed through the ladder
        # because its kernels take a caller-supplied row pitch, so a narrow layer
        # can be packed into an arena sized for a wide one; the plain integer
        # packer derives its pitch from the width and cannot.
        self.combine_layer_period = int(
            os.environ.get("VLLM_CE_A2A_COMBINE_LAYER_PERIOD", "0") or 0
        )
        self.combine_layer_low = int(
            os.environ.get("VLLM_CE_A2A_COMBINE_LAYER_LOW", "0") or 0
        )
        if not 0 <= self.combine_layer_low <= self.combine_layer_period:
            raise ValueError("COMBINE_LAYER_LOW must lie within COMBINE_LAYER_PERIOD")
        # A modular schedule interleaves, which blurs exactly the structure E35
        # says is there: narrowing 36 layers cost a quarter of what the last 12
        # cost, so sensitivity is concentrated somewhere, and depth is the first
        # place to look. A contiguous range answers "where" in four runs.
        self.combine_layer_span: tuple[int, int] | None = None
        span = os.environ.get("VLLM_CE_A2A_COMBINE_LAYER_RANGE", "").strip()
        if span:
            start, _, end = span.partition(":")
            self.combine_layer_span = (int(start), int(end))
            self.combine_layer_period = 1  # selects the layer-mix path below
        # Needed only by the range form, which unlike the modular one cannot
        # rely on the layer count dividing the period.
        self.moe_layers = int(os.environ.get("VLLM_CE_A2A_NUM_MOE_LAYERS", "0") or 0)
        if self.combine_layer_span and self.moe_layers <= 0:
            raise ValueError("COMBINE_LAYER_RANGE needs NUM_MOE_LAYERS")
        # The fill: give every combine edge the widest ladder it can carry inside
        # the busiest edge's byte budget. The exchange waits on its busiest link
        # (E24), so the heavy edge stays at the base width and sets the step time
        # exactly as a uniform base exchange would, and every lighter edge's extra
        # bits ride in slack that was being discarded.
        #
        # Sender and receiver have to agree on the width without another round
        # trip. They do, because the width is a function of the *edge* row count
        # and one global budget: rank s sending to r uses recv_counts_s[r], rank r
        # decoding from s uses send_counts_r[s], and those are the same number.
        self.combine_fill = bool(
            int(os.environ.get("VLLM_CE_A2A_COMBINE_FILL", "0") or 0)
        )
        if self.combine_fill and self.combine_layer_period:
            raise ValueError("the fill and layer mixing are exclusive")
        if self.combine_fill and self.combine_ladder_g:
            raise ValueError("the fill solves the width; do not also fix it")
        if self.combine_fill and not self.combine_bits:
            raise ValueError("the fill needs a base width in COMBINE_BITS")
        if self.combine_layer_period and self.combine_ladder_g:
            raise ValueError("layer mixing and a fixed ladder width are exclusive")
        if (self.combine_ladder_g or self.combine_layer_period) and (
            not self.combine_bits
        ):
            raise ValueError("the combine ladder needs a base width in COMBINE_BITS")
        # Layers run in a fixed order, so counting entries into dispatch -- both
        # the admitted ones and the ones that fall back -- recovers the index.
        self.layer_calls = 0
        self.layer_index = 0

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
        self.dispatch_reference: torch.Tensor | None = None
        self.combine_recv_blocks: torch.Tensor | None = None
        self.combine_send_blocks: torch.Tensor | None = None
        self.combine_value_bytes = 0
        self.combine_ladder: torch.Tensor | None = None
        self.combine_ladder_recv: torch.Tensor | None = None
        self.combine_floor_row = 0
        self.combine_ladder_step = 0
        self.combine_groups = 0
        self.combine_peak: torch.Tensor | None = None
        self.combine_send_row_bytes: torch.Tensor | None = None
        self.combine_recv_row_bytes: torch.Tensor | None = None
        # Kept as device tensors and only read in diagnostics(), which runs off
        # the hot path, so recording the matrix costs no synchronization.
        self.last_combine_send_counts: torch.Tensor | None = None
        self.last_combine_recv_counts: torch.Tensor | None = None
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
            carry_token_id=self.delta_dispatch,
        )
        if self.combine_layer_period or self.combine_fill:
            # Sized for the wide rail, since some layers use it and the arena is
            # allocated once. The narrow layers therefore save no bytes here;
            # this configuration measures accuracy, and E34 already established
            # that the leg is linear in bytes at 457 ms per KB of row.
            #
            # The fill is in the same position for a different reason. Its edges
            # only ever get *wider* than the base, and its claim is that the peak
            # edge is unchanged -- which is a statement about bytes per edge, not
            # about the arena. Realizing it on the wire needs a per-edge copy
            # length in the native proxy's ABI; until then the widths are real and
            # the accuracy they buy is measurable, but the peak is conservative.
            self.combine_ladder_g = hidden_size // self.codec_group_size
        if self.combine_ladder_g:
            from ce_a2a_moe.ladder import ladder_row_bytes, ladder_value_bytes

            self.combine_value_bytes = ladder_value_bytes(
                hidden_size,
                self.combine_bits,
                self.combine_ladder_g,
                self.codec_group_size,
            )
            self.combine_row_bytes = ladder_row_bytes(
                hidden_size,
                self.combine_bits,
                self.combine_ladder_g,
                self.codec_group_size,
            )
        elif self.combine_bits:
            self.combine_row_bytes = lowbit_block_payload_bytes(
                hidden_size, self.combine_bits, self.codec_group_size
            )
        else:
            self.combine_row_bytes = hidden_size * 2
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
            if self.delta_dispatch:
                # Both sides hold one row per peer and token: the sender for
                # what each destination can rebuild, the receiver for what it
                # has rebuilt from each source. A rank never exceeds
                # max_edge_rows local tokens or dispatch falls back.
                self.packet_builder.enable_delta(
                    self.max_edge_rows, margin=self.delta_margin
                )
                self.dispatch_reference = torch.zeros(
                    (self.world_size, self.max_edge_rows, hidden_size),
                    dtype=torch.float16,
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
                if self.combine_ladder_g:
                    # Uniform across destinations, so the kernels still take a
                    # per-destination vector but every entry is the same and the
                    # row pitch is a constant the transport can be sized against.
                    self.combine_ladder = torch.full(
                        (self.world_size,),
                        self.combine_ladder_g,
                        dtype=torch.int32,
                        device=torch.cuda.current_device(),
                    )
                    if self.combine_fill:
                        from ce_a2a_moe.ladder import (
                            ladder_group_bytes as _group_bytes,
                            ladder_row_bytes as _row_bytes,
                        )

                        # Packing is indexed by destination and reduction by
                        # source, and under the fill those are different vectors
                        # of the same count matrix, so they cannot share storage.
                        self.combine_ladder_recv = torch.zeros_like(
                            self.combine_ladder
                        )
                        # The wire widths the two ladders imply, kept as their
                        # own buffers because the control path stages them to
                        # pinned host memory and must not race the solver.
                        self.combine_send_row_bytes = torch.zeros_like(
                            self.combine_ladder
                        )
                        self.combine_recv_row_bytes = torch.zeros_like(
                            self.combine_ladder
                        )
                        self.combine_floor_row = _row_bytes(
                            hidden_size, self.combine_bits, 0, self.codec_group_size
                        )
                        self.combine_ladder_step = _group_bytes(
                            self.codec_group_size, self.combine_bits + 1
                        ) - _group_bytes(self.codec_group_size, self.combine_bits)
                        self.combine_groups = hidden_size // self.codec_group_size
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
        send_row_bytes: torch.Tensor | None = None,
        recv_row_bytes: torch.Tensor | None = None,
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
        if send_row_bytes is None:
            self.control.submit(
                "dispatch" if phase == 0 else "combine",
                send,
                send_counts,
                recv,
                recv_counts,
            )
            return
        # Only the native proxy carries per-edge widths; the other controls
        # would silently send the arena pitch and corrupt every narrow edge.
        if self.control_kind != "native_proxy":
            raise RuntimeError(
                f"combine fill needs VLLM_CE_A2A_CONTROL=native_proxy, "
                f"got {self.control_kind}"
            )
        self.control.submit(
            "dispatch" if phase == 0 else "combine",
            send,
            send_counts,
            recv,
            recv_counts,
            send_row_bytes,
            recv_row_bytes,
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

    def _solve_fill(self, counts: torch.Tensor, peak: torch.Tensor, out: torch.Tensor):
        """Widest ``g`` each edge can carry inside the busiest edge's budget.

        Entirely on device: ``counts`` and ``peak`` are the count tensors the
        control path already keeps in GPU memory, so nothing here forces the D2H
        synchronisation the fixed-control path exists to avoid.
        """

        budget = peak.to(torch.int64) * self.combine_floor_row
        safe = counts.to(torch.int64).clamp(min=1)
        g = (budget // safe - self.combine_floor_row) // self.combine_ladder_step
        g = g.clamp(0, self.combine_groups)
        out.copy_(torch.where(counts > 0, g, torch.zeros_like(g)).to(torch.int32))

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

        # Counted before the admission gate, so the index still tracks the layer
        # when a wave falls back to NCCL.
        self.layer_index = self.layer_calls
        self.layer_calls += 1

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

        if self.delta_probe and self.delta_dispatch:
            # Delta is worth a bit only if the residual's per-group range is
            # smaller than the activation's, since that range is what sets the
            # quantizer's step. Measured against destination zero's reference,
            # which is the real state of that edge rather than an idealization.
            assert self.packet_builder is not None
            rows = int(hidden_states.shape[0])
            groups = hidden_size // self.codec_group_size
            held = self.packet_builder.reference[0, :rows]
            span = hidden_states.view(rows, groups, -1).abs().amax(-1)
            residual_span = (
                (hidden_states - held).view(rows, groups, -1).abs().amax(-1)
            )
            ratio = float((residual_span / span.clamp_min(1e-6)).mean())
            self.delta_span_ratio.append(ratio)
            self._probe_state = (rows, groups, topk_ids)

        if self.delta_dispatch and self.delta_period:
            if self.ce_dispatch_calls % self.delta_period == 0:
                assert self.packet_builder is not None
                assert self.dispatch_reference is not None
                self.packet_builder.reset_delta()
                self.dispatch_reference.zero_()

        with record_function("moe.ce_a2a.coalesce_pack"):
            if self.packet_builder_kind in ("fixed", "fused"):
                assert self.packet_builder is not None
                coalesced = self.packet_builder.build(
                    hidden_states,
                    topk_ids,
                    topk_weights,
                )
                if self.delta_probe and self.delta_dispatch:
                    self._report_delta_error(hidden_states)
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
                    if self.combine_fill:
                        # The budget has to be identical on every rank or the
                        # two ends of an edge would solve different widths. A
                        # max-reduce over the count vectors gives the busiest
                        # edge in the whole matrix, in 16 bytes.
                        peak = coalesced.send_counts.clone()
                        dist.all_reduce(
                            peak, op=dist.ReduceOp.MAX, group=self.device_group
                        )
                        self.combine_peak = peak.max()
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
                        reference=self.dispatch_reference,
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
            if isinstance(active.recv_counts, torch.Tensor):
                self.last_combine_send_counts = active.recv_counts
                self.last_combine_recv_counts = active.send_counts
            if self.combine_ladder_g:
                assert self.combine_send_blocks is not None
                from ce_a2a_moe.ladder import quantize_pack_ladder_blocks

                if self.combine_layer_period:
                    if self.combine_layer_span is not None:
                        start, end = self.combine_layer_span
                        depth = self.layer_index % self.moe_layers
                        narrow = start <= depth < end
                    else:
                        # Narrow rail for the first `low` layers of each period.
                        narrow = (
                            self.layer_index % self.combine_layer_period
                        ) < self.combine_layer_low
                    self.combine_ladder.fill_(0 if narrow else self.combine_ladder_g)
                if self.combine_fill:
                    assert self.combine_peak is not None
                    # Packing is indexed by destination, so it solves from the
                    # rows this rank returns to each owner; the reduction below
                    # is indexed by source and solves from the mirror vector.
                    self._solve_fill(
                        active.recv_counts, self.combine_peak, self.combine_ladder
                    )
                    self._solve_fill(
                        active.send_counts,
                        self.combine_peak,
                        self.combine_ladder_recv,
                    )
                    assert self.combine_send_row_bytes is not None
                    assert self.combine_recv_row_bytes is not None
                    torch.add(
                        self.combine_ladder * self.combine_ladder_step,
                        self.combine_floor_row,
                        out=self.combine_send_row_bytes,
                    )
                    torch.add(
                        self.combine_ladder_recv * self.combine_ladder_step,
                        self.combine_floor_row,
                        out=self.combine_recv_row_bytes,
                    )
                with record_function("moe.ce_a2a.combine_pack"):
                    if self.combine_fill:
                        # Each destination's rows go out at that destination's
                        # own dense pitch with the scales inside the row, so its
                        # block is one contiguous run the transport can send
                        # without also sending the padding up to the widest.
                        from ce_a2a_moe.ladder import quantize_pack_ladder_inline

                        quantize_pack_ladder_inline(
                            blocks,
                            self.combine_send_blocks,
                            self.combine_ladder,
                            base_bits=self.combine_bits,
                            group_size=self.codec_group_size,
                        )
                    else:
                        values, scales = _ladder_views(
                            self.combine_send_blocks, self.combine_value_bytes
                        )
                        quantize_pack_ladder_blocks(
                            blocks,
                            values,
                            scales,
                            self.combine_ladder,
                            base_bits=self.combine_bits,
                            group_size=self.codec_group_size,
                            row_pitch=self.combine_row_bytes,
                        )
                    blocks = self.combine_send_blocks
            elif self.combine_bits:
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
                        self.combine_send_row_bytes if self.combine_fill else None,
                        self.combine_recv_row_bytes if self.combine_fill else None,
                    )
            with record_function("moe.ce_a2a.owner_reduce"):
                if self.combine_fill:
                    from ce_a2a_moe.ladder import reduce_ladder_inline

                    output = reduce_ladder_inline(
                        self.combine_recv_blocks,
                        self.combine_ladder_recv,
                        active.token_positions,
                        block_rows=block_rows,
                        local_rows=active.local_rows,
                        hidden_size=hidden_size,
                        base_bits=self.combine_bits,
                        group_size=self.codec_group_size,
                    )
                elif self.combine_ladder_g:
                    from ce_a2a_moe.ladder import reduce_ladder_owner_partials

                    values, scales = _ladder_views(
                        self.combine_recv_blocks, self.combine_value_bytes
                    )
                    output = reduce_ladder_owner_partials(
                        values,
                        scales,
                        self.combine_ladder,
                        active.token_positions,
                        local_rows=active.local_rows,
                        hidden_size=hidden_size,
                        base_bits=self.combine_bits,
                        group_size=self.codec_group_size,
                        row_pitch=self.combine_row_bytes,
                    )
                elif self.combine_bits:
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

    def _report_delta_error(self, hidden_states: torch.Tensor) -> None:
        """Compare what delta actually rebuilt against plain quantization.

        The reference after a build is the reconstruction the receiver holds,
        so the error is measurable on the sender without any extra exchange.
        Only rows that routed to destination zero are scored, since the others
        left that reference untouched.
        """

        rows, groups, topk_ids = self._probe_state
        held = self.packet_builder.reference[0, :rows]
        routed = (topk_ids // self.experts_per_rank == 0).any(dim=1)
        if not bool(routed.any()):
            return
        truth = hidden_states[routed].float()
        rebuilt = held[routed].float()
        bits = self.dispatch_bits
        limit = 2 ** (bits - 1) - 1
        blocked = truth.view(-1, groups, self.codec_group_size)
        step = (blocked.abs().amax(-1, True) / limit).clamp_min(1e-8)
        plain = ((blocked / step).round().clamp(-limit, limit) * step).view_as(truth)
        scale = truth.square().sum().sqrt().clamp_min(1e-8)
        delta_error = float((rebuilt - truth).square().sum().sqrt() / scale)
        plain_error = float((plain - truth).square().sum().sqrt() / scale)
        index = len(self.delta_span_ratio) - 1
        if self.rank == 0 and index < 96:
            print(
                f"[delta_probe] dispatch {index} "
                f"span_ratio {self.delta_span_ratio[-1]:.4f} "
                f"delta_err {delta_error:.5f} direct_err {plain_error:.5f}",
                flush=True,
            )

    def diagnostics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "backend": "ce_a2a",
            "min_global_rows": self.min_global_rows,
            "max_edge_rows": self.max_edge_rows,
            "packet_builder": self.packet_builder_kind,
            "control": self.control_kind,
            "dispatch_bits": self.dispatch_bits or 16,
            "dispatch_delta": self.delta_dispatch,
            "dispatch_delta_margin": self.delta_margin,
            "delta_span_ratio": self.delta_span_ratio,
            "dispatch_delta_period": self.delta_period,
            "combine_bits": self.combine_bits or 16,
            "codec_group_size": self.codec_group_size,
            "ce_dispatch_calls": self.ce_dispatch_calls,
            "ce_combine_calls": self.ce_combine_calls,
            "nccl_dispatch_calls": self.nccl_dispatch_calls,
            "nccl_combine_calls": self.nccl_combine_calls,
            "capacity_fallbacks": self.capacity_fallbacks,
            "unsupported_fallbacks": self.unsupported_fallbacks,
            "transport_initialized": self.transport is not None,
            # The per-layer schedule keys off this counter, so its alignment is
            # load-bearing: a total that is not a whole number of stacks means
            # some layer skipped a dispatch and every index after it is wrong.
            "combine_fill": self.combine_fill,
            # One row of the combine count matrix per rank; collecting all four
            # reconstructs it, which is what prices any per-edge scheme.
            "combine_send_counts": (
                None if self.last_combine_send_counts is None
                else [int(v) for v in self.last_combine_send_counts.tolist()]
            ),
            "combine_recv_counts": (
                None if self.last_combine_recv_counts is None
                else [int(v) for v in self.last_combine_recv_counts.tolist()]
            ),
            "combine_ladder_send": (
                None if self.combine_ladder is None
                else [int(v) for v in self.combine_ladder.tolist()]
            ),
            "combine_ladder_recv": (
                None if self.combine_ladder_recv is None
                else [int(v) for v in self.combine_ladder_recv.tolist()]
            ),
            "layer_calls": self.layer_calls,
            "layer_calls_mod_stack": (
                self.layer_calls % self.moe_layers if self.moe_layers else None
            ),
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
