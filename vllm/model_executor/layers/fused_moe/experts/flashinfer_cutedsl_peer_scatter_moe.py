# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer CuTe-DSL NVFP4 experts that combine inside GEMM2.

Only the destination of GEMM2's finalize store changes. Instead of writing each
`(token, k_slot)` row into a local buffer, the epilogue writes it into the
combine buffer of the rank that owns the token, selected from a peer-pointer
table. The GEMM math, the store instruction and the SMEM staging are untouched.

Pair this with `MoEPrepareAndFinalizePeerScatter`, which supplies the shared
buffer and destination map and does the local reduction afterwards.
"""

from typing import Any

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_peer_scatter import (  # noqa: E501
    PeerScatterCombineState,
)


class FlashInferCuteDSLPeerScatterExperts(FlashInferCuteDSLExperts):
    """CuteDSL NvFP4 experts whose GEMM2 scatters straight to peer GPUs."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        combine_state: PeerScatterCombineState,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.combine_state = combine_state

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        # Peer scatter only means anything when there is more than one rank to
        # scatter to, and every route must be owned by exactly one rank, which
        # is what expert parallelism gives.
        return moe_parallel_config.ep_size > 1

    def _extra_moe_kwargs(self) -> dict[str, Any]:
        state = self.combine_state
        if not state.armed:
            # prepare() did not arm the combine this forward (for example a
            # profiling run that bypasses it), so fall back to the ordinary
            # local finalize and let prepare/finalize do a reduce-scatter.
            return {}
        return state.moe_kwargs()
