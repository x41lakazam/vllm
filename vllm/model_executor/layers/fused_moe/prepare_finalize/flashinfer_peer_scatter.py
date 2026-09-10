# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer-scatter combine for the FlashInfer CuTe-DSL NVFP4 MoE.

The combine normally runs as its own collective after the experts finish:
`MoEPrepareAndFinalizeNaiveDPEPModular.finalize()` reduces locally and then
calls `get_ep_group().combine(...)`, which is a `reduce_scatterv`. That is a
kernel boundary, so the combine cannot overlap with GEMM2's compute.

Here GEMM2 does the combine itself. Its finalize epilogue writes each
`(token, k_slot)` result straight into the combine buffer of the rank that owns
the token, at tile granularity, overlapped with its own mainloop. Every route
is computed by exactly one rank -- the one owning that expert -- so every
destination slot has exactly one writer in the world, the store needs no
accumulation, and the buffer needs no zero-initialisation.

What is left for `finalize()` is only "wait for the writes, then reduce over
top_k locally", which still fits the `FusedMoEPrepareAndFinalize` contract:
that contract is "take compute's output, produce the final result", not "run a
combine collective".

`prepare()` is inherited unchanged. It is a plain NCCL `all_gatherv`, and
because an all-gather is rank-contiguous by construction, the map from a
gathered row back to `(owning rank, index within that rank)` is fully
determined by the `sizes` vector that dispatch already computes.
"""

import os
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

# Completion signal between the peer writes and the local reduction.
#
# "sync" is a device synchronize plus a host barrier. It is the strongest
# ordering available and it is what the kernel-level multi-GPU test validated,
# so it is the default. It also costs a full device sync per MoE layer, which
# is exactly the cost this project is trying to remove, so it is only the right
# default while correctness is still being established.
#
# "stream" enqueues a tiny NCCL all-reduce on the compute stream instead. It is
# far cheaper and is what a production version would want, but the memory
# ordering it gives for peer writes has NOT been validated here. Opt in with
# VLLM_MOE_PEER_SCATTER_BARRIER=stream and verify before trusting it.
_BARRIER_MODE_ENV = "VLLM_MOE_PEER_SCATTER_BARRIER"


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, (max(value, 1) - 1).bit_length())


class PeerScatterCombineState:
    """State shared by the peer-scatter experts and prepare/finalize.

    GEMM2 performs the combine, so the two halves of the modular kernel have to
    agree on one symmetric buffer and one destination map. `prepare()` fills
    this in for the forward, the experts `apply()` reads the map and the peer
    table, and `finalize()` reads the buffer back.
    """

    def __init__(
        self,
        top_k: int,
        hidden_dim: int,
        dtype: torch.dtype,
        release_fence: bool = False,
    ) -> None:
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        # When set, GEMM2 drains its outstanding bulk copies and issues a
        # system-scope release fence before exiting, so the kernel cannot
        # finish until its peer writes have landed. Without it the ordering
        # rests on kernel completion alone, which is why the cheap
        # stream-ordered wait below is only empirically correct.
        self.release_fence = release_fence

        self._buffer = None  # flashinfer PeerCombineBuffer
        self._capacity = 0  # tokens per rank the buffer can hold
        self._group: GroupCoordinator | None = None

        # The destination map is a pure function of `sizes`, and `sizes` is
        # the same for every MoE layer in a forward and for every forward with
        # the same batch shape. Rebuilding it per layer put two host-built
        # tensors and their H2D copies on the critical path, which measured as
        # a fixed ~0.4 ms per layer -- larger than the combine itself.
        self._map_key: tuple[int, ...] | None = None
        self._map_device: torch.device | None = None
        self._token_dst_rank: torch.Tensor | None = None
        self._token_dst_local_idx: torch.Tensor | None = None

        # Per-forward, set by prepare() and cleared by finalize().
        self.armed = False
        self.sizes: list[int] | None = None
        self.local_offset = 0
        self.num_local_tokens = 0
        self.token_dst_rank: torch.Tensor | None = None
        self.token_dst_local_idx: torch.Tensor | None = None

    @property
    def peer_addresses(self) -> torch.Tensor:
        assert self._buffer is not None
        return self._buffer.peer_addresses

    @property
    def combine_buffer(self) -> torch.Tensor:
        assert self._buffer is not None
        return self._buffer.tensor

    def _ensure_buffer(
        self, group: "GroupCoordinator", capacity: int, device: torch.device
    ) -> None:
        """(Re)allocate the symmetric buffer. Collective on `group`.

        `capacity` is derived from `sizes`, which every rank computes from the
        same DP metadata, so every rank takes the same branch here. If that
        ever stopped holding, the ranks that skipped would hang the ones that
        did not -- hence the explicit check below rather than a silent
        assumption.
        """
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            allocate_peer_combine_buffer,
        )

        if (
            self._buffer is not None
            and self._group is group
            and capacity <= self._capacity
        ):
            return

        agreed = torch.tensor([capacity], dtype=torch.int64, device=device)
        dist.all_reduce(agreed, op=dist.ReduceOp.MAX, group=group.device_group)
        if int(agreed.item()) != capacity:
            raise RuntimeError(
                "peer-scatter combine capacity disagrees across ranks "
                f"(this rank wants {capacity}, the world wants "
                f"{int(agreed.item())}); the destination map would be "
                "inconsistent, so refusing to allocate"
            )

        # Drop the old mapping before creating the new one so the symmetric
        # allocator does not hold both.
        self._buffer = None
        self._buffer = allocate_peer_combine_buffer(
            (capacity * self.top_k, self.hidden_dim),
            self.dtype,
            device,
            group.device_group,
        )
        self._capacity = capacity
        self._group = group
        logger.debug_once(
            "peer-scatter combine buffer: capacity=%d tokens/rank, rows=%d, "
            "peer table via %s",
            capacity,
            capacity * self.top_k,
            self._buffer.source,
        )

    def begin_forward(
        self,
        group: "GroupCoordinator",
        sizes: list[int],
        device: torch.device,
    ) -> None:
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            build_peer_scatter_destination_map,
        )

        rank = group.rank_in_group
        self.sizes = list(sizes)
        self.num_local_tokens = self.sizes[rank]
        self.local_offset = sum(self.sizes[:rank])
        # Round up so a batch-size wobble does not reallocate symmetric memory
        # every forward. max(sizes) is identical on every rank.
        self._ensure_buffer(group, _next_power_of_two(max(self.sizes)), device)

        key = tuple(self.sizes)
        if key != self._map_key or self._map_device != device:
            self._token_dst_rank, self._token_dst_local_idx = (
                build_peer_scatter_destination_map(self.sizes, device)
            )
            self._map_key = key
            self._map_device = device
        self.token_dst_rank = self._token_dst_rank
        self.token_dst_local_idx = self._token_dst_local_idx
        self.armed = True

    def end_forward(self) -> None:
        # Only the per-forward arming is cleared; the cached destination map
        # survives, keyed on `sizes`.
        self.armed = False
        self.sizes = None
        self.token_dst_rank = None
        self.token_dst_local_idx = None

    def moe_kwargs(self) -> dict[str, Any]:
        """The peer arguments to hand the FlashInfer MoE call."""
        return {
            "peer_addresses": self.peer_addresses,
            "token_dst_rank": self.token_dst_rank,
            "token_dst_local_idx": self.token_dst_local_idx,
            "combine_buffer": self.combine_buffer,
            "peer_release_fence": self.release_fence,
            # Peer scatter gives every route its own destination slot, so GEMM2
            # must use the plain non-accumulating store and defer the routing
            # weights to finalize().
            "use_fused_finalize": False,
        }


class MoEPrepareAndFinalizePeerScatter(MoEPrepareAndFinalizeNaiveDPEPModular):
    """Dispatch by all-gather; combine inside GEMM2 via peer writes.

    `prepare()` is the inherited `all_gatherv` verbatim; the only addition is
    recording the destination map the epilogue needs. `finalize()` drops the
    `reduce_scatterv` and only waits and reduces locally.
    """

    def __init__(
        self,
        combine_state: PeerScatterCombineState,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
        self.combine_state = combine_state

    def _comm_group_and_sizes(
        self, num_local_tokens: int
    ) -> tuple["GroupCoordinator", list[int]]:
        """Resolve the same group and `sizes` that dispatch will use.

        Deliberately reuses `AgRsAll2AllManager`'s own helpers rather than
        recomputing the split. If the two ever disagreed, tokens would be
        attributed to the wrong owner and the combine would silently return
        another rank's activations, which is far worse than depending on a
        private method that lives two lines away from the `all_gatherv` it
        feeds.
        """
        manager = get_ep_group().device_communicator.all2all_manager
        group = manager._get_comm_group(self.is_sequence_parallel)
        sizes = manager._get_sizes(num_local_tokens, group)
        if sizes[group.rank_in_group] != num_local_tokens:
            raise RuntimeError(
                f"sizes[{group.rank_in_group}]={sizes[group.rank_in_group]} does "
                f"not match this rank's {num_local_tokens} tokens"
            )
        return group, sizes

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
        group, sizes = self._comm_group_and_sizes(a1.shape[0])
        self.combine_state.begin_forward(group, sizes, a1.device)
        return super().prepare(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )

    def _await_peer_writes(
        self, group: "GroupCoordinator", device: torch.device
    ) -> None:
        mode = os.environ.get(_BARRIER_MODE_ENV, "sync")
        if mode == "stream":
            flag = torch.zeros(1, dtype=torch.int32, device=device)
            dist.all_reduce(flag, group=group.device_group)
            return
        if mode != "sync":
            raise ValueError(
                f"{_BARRIER_MODE_ENV} must be 'sync' or 'stream', got {mode!r}"
            )
        torch.cuda.synchronize(device)
        dist.barrier(group=group.device_group)

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        state = self.combine_state
        if not state.armed:
            # The experts object did not do peer writes this forward, so the
            # result is still a normal local output and needs the ordinary
            # reduce-scatter combine.
            super().finalize(
                output,
                fused_expert_output,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                weight_and_reduce_impl,
            )
            return

        from flashinfer.fused_moe.cute_dsl.moe_utils import peer_scatter_local_reduce

        group, _ = self._comm_group_and_sizes(state.num_local_tokens)
        try:
            self._await_peer_writes(group, output.device)

            # `topk_weights` here is the gathered tensor prepare() returned, so
            # take the slice belonging to the tokens this rank owns.
            my_scales = topk_weights[
                state.local_offset : state.local_offset + state.num_local_tokens
            ]
            peer_scatter_local_reduce(
                state.combine_buffer,
                my_scales,
                state.num_local_tokens,
                state.top_k,
                output=output,
            )
        finally:
            state.end_forward()
