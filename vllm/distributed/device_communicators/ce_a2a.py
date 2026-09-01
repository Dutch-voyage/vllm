# SPDX-License-Identifier: Apache-2.0
"""Experimental exact-size copy-engine all-to-all for single-host DP/EP MoE.

The data plane is intentionally narrow: TP1, linear EP placement, FP16/BF16
activations, and FP32 router weights. Calls below the admission threshold (as
well as unsupported calls) use vLLM's existing NCCL AG/RS manager.  Long
prefill chunks may use either eager native-proxy CE control or a captured,
device-only symmetric-memory P2P path.
"""

from __future__ import annotations

import atexit
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import record_function

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
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


def _codec_group_count(hidden_size: int, group_size: int) -> int:
    if hidden_size <= 0 or group_size <= 0 or hidden_size % group_size:
        raise ValueError(
            "CE A2A codec group size must be positive and divide hidden size"
        )
    return hidden_size // group_size


def _value_row_bytes(hidden_size: int, bits: int, group_size: int) -> int:
    if bits == 16:
        return hidden_size * 2
    groups = _codec_group_count(hidden_size, group_size)
    return groups * ((group_size * bits + 7) // 8)


def _pace_policy_name(
    dispatch_bits: int,
    combine_bits: int,
    delta_dispatch: bool,
    delta_max_edge: bool,
    combine_fill: bool,
) -> str:
    """Return the locked experiment arm selected by the codec policy."""

    policy = (
        int(dispatch_bits),
        int(combine_bits),
        bool(delta_dispatch),
        bool(delta_max_edge),
        bool(combine_fill),
    )
    arms = {
        (0, 0, False, False, False): "fp16",
        (6, 6, False, False, False): "uniform_int6",
        (5, 6, True, True, False): "maxedge_delta_5to6",
        (6, 6, False, False, True): "lightedge_fill_6to7",
        (5, 6, True, True, True): "maxedge_delta_plus_fill",
    }
    return arms.get(policy, "custom")


def _solve_max_edge_dispatch_plan(
    counts: list[int], spec: Any, peak_count: int
) -> tuple[list[int], list[int]]:
    """Solve exact promoted groups and aligned packet widths for one rank."""

    from ce_a2a_moe.packet import (
        dispatch_ladder_packet_bytes,
        solve_max_edge_dispatch_ladder,
    )

    promoted = solve_max_edge_dispatch_ladder(
        [int(value) for value in counts],
        spec,
        int(peak_count),
    )
    row_bytes = [
        dispatch_ladder_packet_bytes(spec, groups) for groups in promoted
    ]
    return promoted, row_bytes


def _dispatch_ladder_value_row_bytes(
    hidden_size: int,
    base_bits: int,
    group_size: int,
    promoted_groups: torch.Tensor,
) -> torch.Tensor:
    """Return code-only row widths for value-effective-bit accounting."""

    base = _value_row_bytes(hidden_size, base_bits, group_size)
    return promoted_groups * (group_size // 8) + base


def _to_ce_wire(
    tensor: torch.Tensor, *, compressed: bool = False
) -> torch.Tensor:
    """Return the tensor carried by one CE wire leg.

    Raw transport remains FP16 for compatibility. Compressed codecs preserve
    BF16 so their two-byte group scales retain BF16's exponent range.
    """
    if tensor.dtype == torch.float16:
        return tensor
    if tensor.dtype == torch.bfloat16:
        return tensor if compressed else tensor.to(torch.float16)
    raise ValueError("CE A2A supports FP16 or BF16 compute activations")


def _from_ce_wire(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CE A2A supports FP16 or BF16 compute activations")
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _attention_phase(attn_metadata: Any) -> str:
    """Classify a model forward without synchronizing device metadata."""

    pending = [attn_metadata]
    saw_prefill = False
    saw_decode = False
    while pending:
        metadata = pending.pop()
        if metadata is None:
            continue
        if isinstance(metadata, dict):
            pending.extend(metadata.values())
            continue
        if isinstance(metadata, (list, tuple)):
            pending.extend(metadata)
            continue

        prefill_tokens = getattr(metadata, "num_prefill_tokens", None)
        decode_tokens = getattr(metadata, "num_decode_tokens", None)
        if isinstance(prefill_tokens, int):
            saw_prefill |= prefill_tokens > 0
        if isinstance(decode_tokens, int):
            saw_decode |= decode_tokens > 0

    if saw_prefill and saw_decode:
        return "mixed"
    if saw_prefill:
        return "prefill"
    if saw_decode:
        return "decode"
    return "unknown"


def _forward_phase() -> str:
    if not is_forward_context_available():
        # DP control/RPC progress can invoke an empty model step without a
        # forward context. It carries no user tokens and is therefore idle,
        # while "unknown" remains reserved for malformed active contexts.
        return "idle"
    context = get_forward_context()
    phase = context.additional_kwargs.get("moe_attention_phase")
    if phase in ("prefill", "decode", "mixed", "idle"):
        return phase
    if context.attn_metadata is None:
        # Engine DP-control padding constructs an empty context without the
        # runner's phase signal. It has no attention metadata or user tokens.
        return "idle"
    return _attention_phase(context.attn_metadata)


def _prefill_policy_fallback_reason(phase: str) -> str | None:
    if phase == "prefill":
        return None
    if phase == "mixed":
        return "policy_mixed_batch"
    if phase in ("decode", "idle", "unknown"):
        return f"policy_{phase}"
    raise ValueError(f"unknown CE A2A forward phase: {phase}")


def _proxy_cpu_for_rank(
    base: int,
    rank: int,
    world_size: int,
    cpu_map: str,
) -> int | None:
    """Resolve an optional topology-aware, one-CPU-per-rank proxy map."""

    if not cpu_map.strip():
        return None if base < 0 else base + rank
    try:
        cpus = [int(value.strip()) for value in cpu_map.split(",")]
    except ValueError as exc:
        raise ValueError("VLLM_CE_A2A_PROXY_CPU_MAP must contain integers") from exc
    if len(cpus) != world_size:
        raise ValueError(
            "VLLM_CE_A2A_PROXY_CPU_MAP must name exactly one CPU per rank"
        )
    if any(cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
        raise ValueError(
            "VLLM_CE_A2A_PROXY_CPU_MAP CPUs must be unique and nonnegative"
        )
    return cpus[rank]


def _ladder_views(
    arena: torch.Tensor,
    value_bytes: int,
    scale_dtype: torch.dtype = torch.float16,
):
    """Split one fixed-stride arena into the pair of tensors the ladder wants.

    The codec addresses codes by row pitch and scales as ``[world, rows,
    groups]``, but the transport moves exactly one buffer per phase, so both have
    to live inside it: each row is its codes followed by its scales. These are
    views, not copies -- the arena itself is what goes on the wire.
    """

    world, rows, row_bytes = (int(dim) for dim in arena.shape)
    return (
        arena.view(world, rows * row_bytes),
        arena[:, :, value_bytes:].view(scale_dtype),
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
    output_dtype: torch.dtype = torch.float16


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
        self.prefill_only = envs.VLLM_CE_A2A_PREFILL_ONLY
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
        self.delta_max_edge = envs.VLLM_CE_A2A_DELTA_MAX_EDGE
        if self.delta_max_edge and not self.delta_dispatch:
            raise ValueError("DELTA_MAX_EDGE builds on delta dispatch")
        if self.delta_max_edge and self.dispatch_bits != 5:
            raise ValueError("DELTA_MAX_EDGE implements the locked INT5-to-INT6 arm")
        if self.delta_max_edge and self.control_kind != "native_proxy":
            raise ValueError(
                "DELTA_MAX_EDGE uses per-edge row widths and needs native_proxy"
            )
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
        self.pace_policy = _pace_policy_name(
            self.dispatch_bits,
            self.combine_bits,
            self.delta_dispatch,
            self.delta_max_edge,
            self.combine_fill,
        )
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
        self.fill_count_send: torch.Tensor | None = None
        self.fill_count_recv: torch.Tensor | None = None
        self.dispatch_recv_blocks: torch.Tensor | None = None
        self.dispatch_row_bytes = 0
        self.dispatch_floor_row = 0
        self.dispatch_ladder_step = 0
        self.dispatch_groups = 0
        self.dispatch_peak: torch.Tensor | None = None
        self.dispatch_ladder: torch.Tensor | None = None
        self.dispatch_ladder_recv: torch.Tensor | None = None
        self.dispatch_send_row_bytes: torch.Tensor | None = None
        self.dispatch_recv_row_bytes: torch.Tensor | None = None
        self.last_dispatch_send_counts: torch.Tensor | None = None
        self.last_dispatch_recv_counts: torch.Tensor | None = None
        self.fixed_recv_hidden: torch.Tensor | None = None
        self.fixed_recv_ids: torch.Tensor | None = None
        self.fixed_recv_weights: torch.Tensor | None = None
        self.dispatch_reference: torch.Tensor | None = None
        self.dispatch_wire_dtype: torch.dtype | None = None
        self.combine_wire_dtype: torch.dtype | None = None
        self.combine_recv_blocks: torch.Tensor | None = None
        self.combine_send_blocks: torch.Tensor | None = None
        self.combine_value_bytes = 0
        self.combine_ladder: torch.Tensor | None = None
        self.combine_ladder_recv: torch.Tensor | None = None
        self.combine_floor_row = 0
        self.combine_ladder_step = 0
        self.codec_groups = 0
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
        self.ce_fp16_dispatch_calls = 0
        self.ce_bf16_dispatch_calls = 0
        self.nccl_dispatch_calls = 0
        self.nccl_combine_calls = 0
        self.capacity_fallbacks = 0
        self.unsupported_fallbacks = 0
        self.min_rows_fallbacks = 0
        self.policy_fallbacks = {
            "decode": 0,
            "idle": 0,
            "mixed_batch": 0,
            "unknown": 0,
        }
        self.phase_dispatch_calls = {
            "prefill": 0,
            "decode": 0,
            "idle": 0,
            "mixed": 0,
            "unknown": 0,
        }
        self.last_dispatch_path: str | None = None
        self.last_fallback_reason: str | None = None
        self.ce_dispatch_wire_bytes = 0
        self.ce_combine_wire_bytes = 0
        self.ce_dispatch_payload_bytes = 0
        self.ce_combine_payload_bytes = 0
        self.ce_dispatch_packets = 0
        self.ce_combine_packets = 0
        self.ce_dispatch_messages = 0
        self.ce_combine_messages = 0
        self._ce_wire_bytes_device: torch.Tensor | None = None
        self._ce_payload_bytes_device: torch.Tensor | None = None
        self._ce_packet_counts_device: torch.Tensor | None = None
        self._ce_message_counts_device: torch.Tensor | None = None
        self._dispatch_effective_inputs_device: torch.Tensor | None = None
        self.phase_timing = envs.VLLM_CE_A2A_PHASE_TIMING
        self.phase_exchanges = 0
        self.phase_cpu_ms: dict[str, float] = {}
        self.phase_cuda_events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {}
        self.adaptive_dispatch_fallbacks: dict[str, int] = {}
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

    @contextmanager
    def _timed_phase(self, key: str) -> Iterator[None]:
        if not self.phase_timing:
            yield
            return

        started = time.perf_counter()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        try:
            yield
        finally:
            cuda_end.record()
            self.phase_cpu_ms[key] = self.phase_cpu_ms.get(key, 0.0) + (
                time.perf_counter() - started
            ) * 1000.0
            self.phase_cuda_events.setdefault(key, []).append(
                (cuda_start, cuda_end)
            )

    def _phase_cuda_ms(self) -> dict[str, float]:
        if not self.phase_timing or not self.phase_cuda_events:
            return {}
        torch.cuda.synchronize()
        return {
            key: sum(float(start.elapsed_time(end)) for start, end in events)
            for key, events in self.phase_cuda_events.items()
        }

    def _record_wire_bytes(
        self,
        phase: int,
        counts: tuple[int, ...] | torch.Tensor,
        row_bytes: int | torch.Tensor,
        *,
        payload: bool = False,
    ) -> None:
        if isinstance(counts, torch.Tensor):
            accumulator = (
                self._ce_payload_bytes_device
                if payload
                else self._ce_wire_bytes_device
            )
            assert accumulator is not None
            widths = row_bytes
            if isinstance(widths, int):
                remote = (counts.sum() - counts[self.rank]).to(torch.int64) * widths
            else:
                remote = (counts.to(torch.int64) * widths.to(torch.int64)).sum()
                remote -= counts[self.rank].to(torch.int64) * widths[
                    self.rank
                ].to(torch.int64)
            accumulator[phase].add_(remote)
            if not payload:
                assert self._ce_packet_counts_device is not None
                assert self._ce_message_counts_device is not None
                remote_counts = counts.to(torch.int64).clone()
                remote_counts[self.rank] = 0
                self._ce_packet_counts_device[phase].add_(remote_counts.sum())
                self._ce_message_counts_device[phase].add_(
                    (remote_counts > 0).sum()
                )
            return

        widths = (
            (int(row_bytes),) * self.world_size
            if isinstance(row_bytes, int)
            else tuple(int(value) for value in row_bytes.tolist())
        )
        remote = sum(
            int(count) * widths[peer]
            for peer, count in enumerate(counts)
            if peer != self.rank
        )
        if payload:
            if phase == 0:
                self.ce_dispatch_payload_bytes += remote
            else:
                self.ce_combine_payload_bytes += remote
        elif phase == 0:
            self.ce_dispatch_wire_bytes += remote
            self.ce_dispatch_packets += sum(
                int(count)
                for peer, count in enumerate(counts)
                if peer != self.rank
            )
            self.ce_dispatch_messages += sum(
                int(count) > 0
                for peer, count in enumerate(counts)
                if peer != self.rank
            )
        else:
            self.ce_combine_wire_bytes += remote
            self.ce_combine_packets += sum(
                int(count)
                for peer, count in enumerate(counts)
                if peer != self.rank
            )
            self.ce_combine_messages += sum(
                int(count) > 0
                for peer, count in enumerate(counts)
                if peer != self.rank
            )

    def _fallback_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool,
        extra_tensors: list[torch.Tensor] | None,
        *,
        reason: str,
    ):
        self._active = "nccl"
        self.nccl_dispatch_calls += 1
        self.last_dispatch_path = "nccl"
        self.last_fallback_reason = reason
        if self.delta_max_edge:
            self.adaptive_dispatch_fallbacks[reason] = (
                self.adaptive_dispatch_fallbacks.get(reason, 0) + 1
            )
        if reason == "unsupported":
            self.unsupported_fallbacks += 1
        elif reason == "capacity":
            self.capacity_fallbacks += 1
        elif reason == "min_rows":
            self.min_rows_fallbacks += 1
        elif reason.startswith("policy_"):
            self.policy_fallbacks[reason.removeprefix("policy_")] += 1
        return self.fallback.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors,
        )

    def _ensure_transport(
        self, hidden_size: int, top_k: int, compute_dtype: torch.dtype
    ) -> None:
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

        if compute_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("CE A2A transport requires FP16 or BF16 compute")
        dispatch_wire_dtype = (
            compute_dtype if self.dispatch_bits else torch.float16
        )
        combine_wire_dtype = compute_dtype if self.combine_bits else torch.float16
        if self.transport is not None:
            assert self.packet_spec is not None
            if (
                self.packet_spec.hidden_size != hidden_size
                or self.packet_spec.top_k != top_k
            ):
                raise ValueError("CE A2A v1 supports one MoE packet shape per model")
            if (
                self.dispatch_wire_dtype != dispatch_wire_dtype
                or self.combine_wire_dtype != combine_wire_dtype
            ):
                raise ValueError("CE A2A compute dtype changed after initialization")
            return
        self.dispatch_wire_dtype = dispatch_wire_dtype
        self.combine_wire_dtype = combine_wire_dtype
        self.codec_groups = _codec_group_count(hidden_size, self.codec_group_size)
        self.packet_spec = DispatchPacketSpec(
            hidden_size=hidden_size,
            top_k=top_k,
            activation_bits=self.dispatch_bits or 16,
            group_size=self.codec_group_size,
            carry_token_id=self.delta_dispatch,
        )
        self.dispatch_row_bytes = self.packet_spec.packet_bytes
        if self.delta_max_edge:
            from ce_a2a_moe.packet import dispatch_ladder_packet_bytes

            self.dispatch_floor_row = dispatch_ladder_packet_bytes(
                self.packet_spec, 0
            )
            self.dispatch_groups = self.packet_spec.groups
            self.dispatch_row_bytes = dispatch_ladder_packet_bytes(
                self.packet_spec, self.dispatch_groups
            )
            self.dispatch_ladder_step = (
                self.dispatch_row_bytes - self.dispatch_floor_row
            ) // self.dispatch_groups
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
            builder_options = {}
            if self.delta_max_edge:
                builder_options["max_activation_bits"] = self.dispatch_bits + 1
            if builder_type is FusedBlockDispatchBuilder:
                builder_options["activation_dtype"] = dispatch_wire_dtype
            elif dispatch_wire_dtype != torch.float16:
                raise ValueError(
                    "the fixed packet builder does not support BF16; use fused"
                )
            self.packet_builder = builder_type(
                spec=self.packet_spec,
                experts_per_rank=self.experts_per_rank,
                world_size=self.world_size,
                max_edge_rows=self.max_edge_rows,
                device=torch.cuda.current_device(),
                **builder_options,
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
                    dtype=dispatch_wire_dtype,
                    device=torch.cuda.current_device(),
                )
        direct_inbox_receive = self.control_kind == "native_proxy"
        self.transport = CoalescedCeTransport(
            # The packet ABI reserves one sentinel row. With a compact
            # symmetric-inbox stride, that inbox is also the contiguous fixed
            # receive tensor and the proxy need not submit eight copy-outs.
            max_edge_rows=(
                self.max_edge_rows + 1
                if direct_inbox_receive
                else self.max_edge_rows
            ),
            dispatch_row_bytes=self.dispatch_row_bytes,
            hidden_size=hidden_size,
            combine_row_bytes=self.combine_row_bytes,
            scheduler=self.scheduler,
            process_group=self.device_group,
            compact_block_stride=direct_inbox_receive,
        )
        self._ce_wire_bytes_device = torch.zeros(
            2,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        self._ce_payload_bytes_device = torch.zeros_like(
            self._ce_wire_bytes_device
        )
        self._ce_packet_counts_device = torch.zeros_like(
            self._ce_wire_bytes_device
        )
        self._ce_message_counts_device = torch.zeros_like(
            self._ce_wire_bytes_device
        )
        self._dispatch_effective_inputs_device = torch.zeros(
            2,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        if self.control_kind != "host_sync":
            block_rows = self.max_edge_rows + 1
            total_rows = self.world_size * block_rows
            self.recv_counts_device = torch.empty(
                self.world_size,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            if self.combine_fill or self.delta_max_edge:
                # Carry the source-local peak beside every count. One exact
                # all-to-all then reconstructs both destination counts and the
                # global peak without a separately ordered scalar all-reduce.
                self.fill_count_send = torch.empty(
                    (self.world_size, 2),
                    dtype=torch.int32,
                    device=torch.cuda.current_device(),
                )
                self.fill_count_recv = torch.empty_like(self.fill_count_send)
            self.dispatch_recv_blocks = (
                self.transport._inbox_view(
                    self.transport.dispatch_inbox,
                    dtype=torch.uint8,
                    row_elements=self.dispatch_row_bytes,
                ).local
                if direct_inbox_receive
                else torch.empty(
                    (self.world_size, block_rows, self.dispatch_row_bytes),
                    dtype=torch.uint8,
                    device=torch.cuda.current_device(),
                )
            )
            if direct_inbox_receive and not self.dispatch_recv_blocks.is_contiguous():
                raise RuntimeError("direct dispatch inbox must be contiguous")
            if self.delta_max_edge:
                self.dispatch_ladder = torch.zeros_like(self.recv_counts_device)
                self.dispatch_ladder_recv = torch.zeros_like(
                    self.recv_counts_device
                )
                self.dispatch_send_row_bytes = torch.zeros_like(
                    self.recv_counts_device
                )
                self.dispatch_recv_row_bytes = torch.zeros_like(
                    self.recv_counts_device
                )
            self.fixed_recv_hidden = torch.empty(
                (total_rows, hidden_size),
                dtype=dispatch_wire_dtype,
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
                self.combine_recv_blocks = (
                    self.transport._inbox_view(
                        self.transport.combine_inbox,
                        dtype=torch.uint8,
                        row_elements=self.combine_row_bytes,
                    ).local
                    if direct_inbox_receive
                    else torch.empty_like(self.combine_send_blocks)
                )
                if (
                    direct_inbox_receive
                    and not self.combine_recv_blocks.is_contiguous()
                ):
                    raise RuntimeError("direct combine inbox must be contiguous")
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
                self.combine_recv_blocks = (
                    self.transport._inbox_view(
                        self.transport.combine_inbox,
                        dtype=combine_wire_dtype,
                        row_elements=hidden_size,
                    ).local
                    if direct_inbox_receive
                    else torch.empty(
                        (self.world_size, block_rows, hidden_size),
                        dtype=combine_wire_dtype,
                        device=torch.cuda.current_device(),
                    )
                )
                if (
                    direct_inbox_receive
                    and not self.combine_recv_blocks.is_contiguous()
                ):
                    raise RuntimeError("direct combine inbox must be contiguous")
            proxy_cpu_base = envs.VLLM_CE_A2A_PROXY_CPU
            proxy_cpu = _proxy_cpu_for_rank(
                proxy_cpu_base,
                self.rank,
                self.world_size,
                envs.VLLM_CE_A2A_PROXY_CPU_MAP,
            )
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
                    proxy_cpu=proxy_cpu,
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
                f"variable-width PACE needs VLLM_CE_A2A_CONTROL=native_proxy, "
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
        self.last_dispatch_path = "nccl"
        self.last_fallback_reason = "unsupported_monolithic"
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

    def _solve_max_edge_delta(
        self, counts: torch.Tensor, peak: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Promote light-edge groups inside the low-bit max-edge budget."""

        budget = peak.to(torch.int64) * self.dispatch_floor_row
        safe = counts.to(torch.int64).clamp(min=1)
        g = (
            budget // safe - self.dispatch_floor_row
        ) // self.dispatch_ladder_step
        g = g.clamp(0, self.dispatch_groups)
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

        phase = _forward_phase()
        self.phase_dispatch_calls[phase] += 1
        policy_reason = _prefill_policy_fallback_reason(phase)
        if self.prefill_only and policy_reason is not None:
            return self._fallback_dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
                reason=policy_reason,
            )

        comm_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(int(hidden_states.shape[0]), comm_group)
        global_rows = sum(int(value) for value in sizes)
        unsupported = bool(
            is_sequence_parallel
            or extra_tensors is not None
            or hidden_states.dtype not in (torch.float16, torch.bfloat16)
            or topk_weights.dtype != torch.float32
            or hidden_states.ndim != 2
            or topk_ids.ndim != 2
        )
        if unsupported:
            return self._fallback_dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
                reason="unsupported",
            )
        if global_rows < self.min_global_rows:
            return self._fallback_dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
                reason="min_rows",
            )
        if max(sizes, default=0) > self.max_edge_rows:
            return self._fallback_dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
                reason="capacity",
            )

        from ce_a2a_moe import (
            build_dispatch_packets,
            compact_fixed_block_rows,
            dispatch_packet_views,
            unpack_fixed_dispatch_blocks,
        )

        output_dtype = hidden_states.dtype
        hidden_size = int(hidden_states.shape[1])
        top_k = int(topk_ids.shape[1])
        self._ensure_transport(hidden_size, top_k, output_dtype)
        assert self.transport is not None and self.packet_spec is not None
        hidden_states = _to_ce_wire(
            hidden_states, compressed=bool(self.dispatch_bits)
        )
        if output_dtype == torch.bfloat16:
            self.ce_bf16_dispatch_calls += 1
        else:
            self.ce_fp16_dispatch_calls += 1
        self.last_dispatch_path = "ce_a2a"
        self.last_fallback_reason = None

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

        dispatch_build_phase = (
            "dispatch_layout" if self.delta_max_edge else "dispatch_pack"
        )
        with (
            self._timed_phase(dispatch_build_phase),
            record_function("moe.ce_a2a.coalesce_pack"),
        ):
            if self.packet_builder_kind in ("fixed", "fused"):
                assert self.packet_builder is not None
                if self.delta_max_edge:
                    coalesced = self.packet_builder.prepare(
                        hidden_states, topk_ids, topk_weights
                    )
                else:
                    coalesced = self.packet_builder.build(
                        hidden_states,
                        topk_ids,
                        topk_weights,
                    )
                if self.delta_probe and self.delta_dispatch and not self.delta_max_edge:
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
                with (
                    self._timed_phase("count_collective"),
                    record_function("moe.ce_a2a.count_exchange_gpu"),
                ):
                    if self.combine_fill or self.delta_max_edge:
                        assert self.fill_count_send is not None
                        assert self.fill_count_recv is not None
                        local_peak = coalesced.send_counts.max()
                        self.fill_count_send[:, 0].copy_(
                            coalesced.send_counts
                        )
                        self.fill_count_send[:, 1].copy_(
                            local_peak.expand(self.world_size)
                        )
                        dist.all_to_all_single(
                            self.fill_count_recv,
                            self.fill_count_send,
                            group=self.device_group,
                        )
                        recv_counts_device.copy_(self.fill_count_recv[:, 0])
                        peak = self.fill_count_recv[:, 1].max()
                        if self.combine_fill:
                            self.combine_peak = peak
                        if self.delta_max_edge:
                            self.dispatch_peak = peak
                    else:
                        dist.all_to_all_single(
                            recv_counts_device,
                            coalesced.send_counts,
                            group=self.device_group,
                        )
            dispatch_send_row_bytes = None
            dispatch_recv_row_bytes = None
            if self.delta_max_edge:
                assert self.dispatch_peak is not None
                assert self.dispatch_ladder is not None
                assert self.dispatch_ladder_recv is not None
                assert self.dispatch_send_row_bytes is not None
                assert self.dispatch_recv_row_bytes is not None
                assert self.packet_builder is not None
                with self._timed_phase("dispatch_plan"):
                    self._solve_max_edge_delta(
                        coalesced.send_counts,
                        self.dispatch_peak,
                        self.dispatch_ladder,
                    )
                    self._solve_max_edge_delta(
                        recv_counts_device,
                        self.dispatch_peak,
                        self.dispatch_ladder_recv,
                    )
                    torch.mul(
                        self.dispatch_ladder,
                        self.dispatch_ladder_step,
                        out=self.dispatch_send_row_bytes,
                    )
                    self.dispatch_send_row_bytes.add_(self.dispatch_floor_row)
                    torch.mul(
                        self.dispatch_ladder_recv,
                        self.dispatch_ladder_step,
                        out=self.dispatch_recv_row_bytes,
                    )
                    self.dispatch_recv_row_bytes.add_(self.dispatch_floor_row)
                with self._timed_phase("dispatch_pack"):
                    coalesced = self.packet_builder.pack_prepared_ladder(
                        hidden_states,
                        topk_ids,
                        topk_weights,
                        self.dispatch_ladder,
                    )
                if self.delta_probe:
                    self._report_delta_error(hidden_states)
                self.last_dispatch_send_counts = coalesced.send_counts
                self.last_dispatch_recv_counts = recv_counts_device
                dispatch_send_row_bytes = self.dispatch_send_row_bytes
                dispatch_recv_row_bytes = self.dispatch_recv_row_bytes
                account_counts = coalesced.send_counts.to(torch.int64).clone()
                account_counts[self.rank] = 0
                assert self._dispatch_effective_inputs_device is not None
                self._dispatch_effective_inputs_device[0].add_(
                    account_counts.sum()
                )
                self._dispatch_effective_inputs_device[1].add_(
                    (account_counts * self.dispatch_ladder.to(torch.int64)).sum()
                )
            wire_row_bytes = (
                dispatch_send_row_bytes
                if dispatch_send_row_bytes is not None
                else self.packet_spec.packet_bytes
            )
            payload_row_bytes = (
                _dispatch_ladder_value_row_bytes(
                    hidden_size,
                    self.dispatch_bits,
                    self.codec_group_size,
                    self.dispatch_ladder,
                )
                if self.delta_max_edge
                else _value_row_bytes(
                    hidden_size,
                    self.dispatch_bits or 16,
                    self.codec_group_size,
                )
            )
            self._record_wire_bytes(0, coalesced.send_counts, wire_row_bytes)
            self._record_wire_bytes(
                0, coalesced.send_counts, payload_row_bytes, payload=True
            )
            if self.control_kind != "host_sync":
                assert self.control is not None
                assert self.dispatch_recv_blocks is not None
                with (
                    self._timed_phase("dispatch_control"),
                    record_function("moe.ce_a2a.dispatch_control"),
                ):
                    self._submit_fixed_exchange(
                        0,
                        coalesced.packets,
                        coalesced.send_counts,
                        self.dispatch_recv_blocks,
                        recv_counts_device,
                        dispatch_send_row_bytes,
                        dispatch_recv_row_bytes,
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
            with (
                self._timed_phase("dispatch_unpack"),
                record_function("moe.ce_a2a.unpack"),
            ):
                if self.delta_max_edge:
                    from ce_a2a_moe import unpack_ladder_dispatch_blocks

                    assert self.dispatch_ladder_recv is not None
                    assert self.dispatch_reference is not None
                    recv_hidden, recv_topk_ids, recv_topk_weights = (
                        unpack_ladder_dispatch_blocks(
                            self.dispatch_recv_blocks,
                            recv_counts_device,
                            self.dispatch_ladder_recv,
                            spec=self.packet_spec,
                            expert_rank=self.rank,
                            experts_per_rank=self.experts_per_rank,
                            output_hidden=self.fixed_recv_hidden,
                            output_ids=self.fixed_recv_ids,
                            output_weights=self.fixed_recv_weights,
                            reference=self.dispatch_reference,
                        )
                    )
                else:
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
                output_dtype=output_dtype,
            )
            self.ce_dispatch_calls += 1
            return (
                _from_ce_wire(recv_hidden, output_dtype),
                recv_topk_weights,
                recv_topk_ids,
            )

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
            output_dtype=output_dtype,
        )
        self.ce_dispatch_calls += 1
        return (
            _from_ce_wire(recv_hidden, output_dtype),
            recv_topk_weights,
            recv_topk_ids,
        )

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
        if (
            hidden_states.dtype not in (torch.float16, torch.bfloat16)
            or hidden_states.ndim != 2
        ):
            raise ValueError("CE A2A combine requires a rank-2 FP16 or BF16 tensor")
        wire_hidden_states = _to_ce_wire(
            hidden_states, compressed=bool(self.combine_bits)
        )
        if wire_hidden_states.dtype != self.combine_wire_dtype:
            raise ValueError("CE A2A combine dtype changed after initialization")

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
            hidden_size = int(wire_hidden_states.shape[1])
            blocks = wire_hidden_states.reshape(
                self.world_size, block_rows, hidden_size
            )
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
                    with self._timed_phase("combine_plan"):
                        self._solve_fill(
                            active.recv_counts,
                            self.combine_peak,
                            self.combine_ladder,
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
                with (
                    self._timed_phase("combine_pack"),
                    record_function("moe.ce_a2a.combine_pack"),
                ):
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
                            self.combine_send_blocks,
                            self.combine_value_bytes,
                            wire_hidden_states.dtype,
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
                with (
                    self._timed_phase("combine_pack"),
                    record_function("moe.ce_a2a.combine_pack"),
                ):
                    blocks = quantize_pack_combine_blocks(
                        blocks,
                        self.combine_send_blocks,
                        bits=self.combine_bits,
                        group_size=self.codec_group_size,
                    )
            with record_function("moe.ce_a2a.combine_transport"):
                with (
                    self._timed_phase("combine_control"),
                    record_function("moe.ce_a2a.combine_control"),
                ):
                    if self.combine_ladder is not None:
                        assert self.combine_ladder is not None
                        narrow_group_bytes = (
                            self.codec_group_size * self.combine_bits + 7
                        ) // 8
                        wide_group_bytes = (
                            self.codec_group_size * (self.combine_bits + 1) + 7
                        ) // 8
                        payload_row_bytes: int | torch.Tensor = (
                            self.codec_groups * narrow_group_bytes
                            + self.combine_ladder
                            * (wide_group_bytes - narrow_group_bytes)
                        )
                    else:
                        payload_row_bytes = _value_row_bytes(
                            hidden_size,
                            self.combine_bits or 16,
                            self.codec_group_size,
                        )
                    self._record_wire_bytes(
                        1,
                        active.recv_counts,
                        payload_row_bytes,
                        payload=True,
                    )
                    self._record_wire_bytes(
                        1,
                        active.recv_counts,
                        (
                            self.combine_send_row_bytes
                            if self.combine_fill
                            else self.combine_row_bytes
                        ),
                    )
                    self._submit_fixed_exchange(
                        1,
                        blocks,
                        active.recv_counts,
                        self.combine_recv_blocks,
                        active.send_counts,
                        self.combine_send_row_bytes if self.combine_fill else None,
                        self.combine_recv_row_bytes if self.combine_fill else None,
                    )
            with (
                self._timed_phase("owner_reduce"),
                record_function("moe.ce_a2a.owner_reduce"),
            ):
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
                        output_dtype=wire_hidden_states.dtype,
                    )
                elif self.combine_ladder_g:
                    from ce_a2a_moe.ladder import reduce_ladder_owner_partials

                    values, scales = _ladder_views(
                        self.combine_recv_blocks,
                        self.combine_value_bytes,
                        wire_hidden_states.dtype,
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
                        output_dtype=wire_hidden_states.dtype,
                    )
                elif self.combine_bits:
                    output = reduce_packed_owner_partials(
                        self.combine_recv_blocks,
                        active.token_positions,
                        local_rows=active.local_rows,
                        hidden_size=hidden_size,
                        bits=self.combine_bits,
                        group_size=self.codec_group_size,
                        output_dtype=wire_hidden_states.dtype,
                    )
                else:
                    output = reduce_fixed_owner_partials(
                        self.combine_recv_blocks,
                        active.token_positions,
                        local_rows=active.local_rows,
                    )
            self._active = None
            self.ce_combine_calls += 1
            self.phase_exchanges += 1
            return _from_ce_wire(output, active.output_dtype)

        assert isinstance(active.send_counts, tuple)
        assert isinstance(active.recv_counts, tuple)
        assert active.owner_token_ids is not None
        owner_partials = torch.empty(
            (sum(active.send_counts), int(wire_hidden_states.shape[1])),
            dtype=torch.float16,
            device=wire_hidden_states.device,
        )
        self._record_wire_bytes(
            1,
            active.recv_counts,
            int(wire_hidden_states.shape[1]) * 2,
        )
        self._record_wire_bytes(
            1,
            active.recv_counts,
            int(wire_hidden_states.shape[1]) * 2,
            payload=True,
        )
        with (
            self._timed_phase("combine_control"),
            record_function("moe.ce_a2a.combine_transport"),
        ):
            self.transport.combine(
                wire_hidden_states,
                active.recv_counts,
                owner_partials,
                active.send_counts,
            )
        with (
            self._timed_phase("owner_reduce"),
            record_function("moe.ce_a2a.owner_reduce"),
        ):
            output = reduce_owner_partials(
                owner_partials,
                active.owner_token_ids,
                local_rows=active.local_rows,
            )
        self._active = None
        self.ce_combine_calls += 1
        self.phase_exchanges += 1
        return _from_ce_wire(output, active.output_dtype)

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
        device_wire_bytes = (
            (0, 0)
            if self._ce_wire_bytes_device is None
            else tuple(int(value) for value in self._ce_wire_bytes_device.tolist())
        )
        actual_dispatch_wire_bytes = (
            self.ce_dispatch_wire_bytes + device_wire_bytes[0]
        )
        actual_combine_wire_bytes = self.ce_combine_wire_bytes + device_wire_bytes[1]
        device_payload_bytes = (
            (0, 0)
            if self._ce_payload_bytes_device is None
            else tuple(
                int(value) for value in self._ce_payload_bytes_device.tolist()
            )
        )
        actual_dispatch_payload_bytes = (
            self.ce_dispatch_payload_bytes + device_payload_bytes[0]
        )
        actual_combine_payload_bytes = (
            self.ce_combine_payload_bytes + device_payload_bytes[1]
        )
        device_packet_counts = (
            (0, 0)
            if self._ce_packet_counts_device is None
            else tuple(
                int(value) for value in self._ce_packet_counts_device.tolist()
            )
        )
        device_message_counts = (
            (0, 0)
            if self._ce_message_counts_device is None
            else tuple(
                int(value) for value in self._ce_message_counts_device.tolist()
            )
        )
        actual_dispatch_packets = (
            self.ce_dispatch_packets + device_packet_counts[0]
        )
        actual_combine_packets = (
            self.ce_combine_packets + device_packet_counts[1]
        )
        actual_dispatch_messages = (
            self.ce_dispatch_messages + device_message_counts[0]
        )
        actual_combine_messages = (
            self.ce_combine_messages + device_message_counts[1]
        )
        phase_cuda_ms = self._phase_cuda_ms()
        effective_inputs = (
            (0, 0)
            if self._dispatch_effective_inputs_device is None
            else tuple(
                int(value)
                for value in self._dispatch_effective_inputs_device.tolist()
            )
        )
        dispatch_value_effective_bits = None
        dispatch_packet_effective_bits = None
        if (
            self.delta_max_edge
            and effective_inputs[0]
            and self.packet_spec is not None
        ):
            dispatch_value_effective_bits = self.dispatch_bits + (
                effective_inputs[1]
                / (effective_inputs[0] * self.dispatch_groups)
            )
            dispatch_packet_effective_bits = (
                8 * actual_dispatch_wire_bytes
            ) / (effective_inputs[0] * self.packet_spec.hidden_size)
        result: dict[str, Any] = {
            "backend": "ce_a2a",
            "min_global_rows": self.min_global_rows,
            "max_edge_rows": self.max_edge_rows,
            "packet_builder": self.packet_builder_kind,
            "control": self.control_kind,
            "dispatch_bits": self.dispatch_bits or 16,
            "dispatch_delta": self.delta_dispatch,
            "dispatch_delta_max_edge": self.delta_max_edge,
            "pace_policy": self.pace_policy,
            "adaptive_dispatch_fallbacks": dict(
                self.adaptive_dispatch_fallbacks
            ),
            "dispatch_delta_margin": self.delta_margin,
            "delta_span_ratio": self.delta_span_ratio,
            "dispatch_delta_period": self.delta_period,
            "combine_bits": self.combine_bits or 16,
            "codec_group_size": self.codec_group_size,
            "codec_groups": self.codec_groups or None,
            "prefill_only": self.prefill_only,
            "ce_dispatch_calls": self.ce_dispatch_calls,
            "ce_combine_calls": self.ce_combine_calls,
            "ce_fp16_dispatch_calls": self.ce_fp16_dispatch_calls,
            "ce_bf16_dispatch_calls": self.ce_bf16_dispatch_calls,
            "nccl_dispatch_calls": self.nccl_dispatch_calls,
            "nccl_combine_calls": self.nccl_combine_calls,
            "capacity_fallbacks": self.capacity_fallbacks,
            "min_rows_fallbacks": self.min_rows_fallbacks,
            "unsupported_fallbacks": self.unsupported_fallbacks,
            "policy_fallbacks": dict(self.policy_fallbacks),
            "phase_dispatch_calls": dict(self.phase_dispatch_calls),
            "last_dispatch_path": self.last_dispatch_path,
            "last_fallback_reason": self.last_fallback_reason,
            "actual_dispatch_wire_bytes": actual_dispatch_wire_bytes,
            "actual_combine_wire_bytes": actual_combine_wire_bytes,
            "actual_total_wire_bytes": (
                actual_dispatch_wire_bytes + actual_combine_wire_bytes
            ),
            "actual_dispatch_payload_bytes": actual_dispatch_payload_bytes,
            "actual_combine_payload_bytes": actual_combine_payload_bytes,
            "actual_total_payload_bytes": (
                actual_dispatch_payload_bytes + actual_combine_payload_bytes
            ),
            "actual_dispatch_packets": actual_dispatch_packets,
            "actual_combine_packets": actual_combine_packets,
            "actual_total_packets": (
                actual_dispatch_packets + actual_combine_packets
            ),
            "actual_dispatch_messages": actual_dispatch_messages,
            "actual_combine_messages": actual_combine_messages,
            "actual_total_messages": (
                actual_dispatch_messages + actual_combine_messages
            ),
            "phase_timing": self.phase_timing,
            "phase_exchanges": self.phase_exchanges,
            "phase_cpu_ms": {
                key: round(value, 3)
                for key, value in sorted(self.phase_cpu_ms.items())
            },
            "phase_cuda_ms": {
                key: round(value, 3)
                for key, value in sorted(phase_cuda_ms.items())
            },
            "pace_cuda_ms": round(sum(phase_cuda_ms.values()), 3),
            "dispatch_effective_inputs": {
                "remote_rows": effective_inputs[0],
                "promoted_group_rows": effective_inputs[1],
                "hidden_size": (
                    None
                    if self.packet_spec is None
                    else self.packet_spec.hidden_size
                ),
                "groups_per_row": self.dispatch_groups or None,
                "base_bits": self.dispatch_bits if self.delta_max_edge else None,
            },
            "dispatch_value_effective_bits": dispatch_value_effective_bits,
            "dispatch_packet_effective_bits": dispatch_packet_effective_bits,
            "dispatch_peak_rows": (
                None
                if self.dispatch_peak is None
                else int(self.dispatch_peak.item())
            ),
            "dispatch_send_counts": (
                None
                if self.last_dispatch_send_counts is None
                else [int(v) for v in self.last_dispatch_send_counts.tolist()]
            ),
            "dispatch_recv_counts": (
                None
                if self.last_dispatch_recv_counts is None
                else [int(v) for v in self.last_dispatch_recv_counts.tolist()]
            ),
            "dispatch_promoted_groups_send": (
                None
                if self.dispatch_ladder is None
                else [int(v) for v in self.dispatch_ladder.tolist()]
            ),
            "dispatch_promoted_groups_recv": (
                None
                if self.dispatch_ladder_recv is None
                else [int(v) for v in self.dispatch_ladder_recv.tolist()]
            ),
            "dispatch_row_bytes_send": (
                None
                if self.dispatch_send_row_bytes is None
                else [int(v) for v in self.dispatch_send_row_bytes.tolist()]
            ),
            "dispatch_row_bytes_recv": (
                None
                if self.dispatch_recv_row_bytes is None
                else [int(v) for v in self.dispatch_recv_row_bytes.tolist()]
            ),
            "dispatch_edge_wire_bytes_send": (
                None
                if self.last_dispatch_send_counts is None
                else [
                    0 if peer == self.rank else int(count) * int(width)
                    for peer, (count, width) in enumerate(
                        zip(
                            self.last_dispatch_send_counts.tolist(),
                            self.dispatch_send_row_bytes.tolist(),
                        )
                    )
                ]
            ),
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
        self.fill_count_send = None
        self.fill_count_recv = None
        self.dispatch_recv_blocks = None
        self.fixed_recv_hidden = None
        self.fixed_recv_ids = None
        self.fixed_recv_weights = None
        self.dispatch_reference = None
        self.combine_recv_blocks = None
        self.combine_send_blocks = None
        self._ce_wire_bytes_device = None
        self._ce_payload_bytes_device = None
        self.dispatch_ladder = None
        self.dispatch_ladder_recv = None
        self.dispatch_send_row_bytes = None
        self.dispatch_recv_row_bytes = None
        self.last_dispatch_send_counts = None
        self.last_dispatch_recv_counts = None
        self._dispatch_effective_inputs_device = None
        self._active = None
