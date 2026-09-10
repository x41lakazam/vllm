# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the peer-scatter MoE combine wiring.

These cover the parts that do not need four GPUs: the destination bookkeeping,
the interface contract against the base prepare/finalize, and the fact that the
keyword arguments handed to FlashInfer are the ones its entry point accepts.

The numerical behaviour of the peer writes themselves is covered on the
FlashInfer side, in ``tests/moe/test_cute_dsl_moe_peer_scatter_multigpu.py``,
which needs four Blackwell GPUs.
"""

import inspect

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_peer_scatter import (  # noqa: E501
    MoEPrepareAndFinalizePeerScatter,
    PeerScatterCombineState,
    _next_power_of_two,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)


class _FakeGroup:
    """Enough of a GroupCoordinator for the bookkeeping under test."""

    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size
        self.device_group = None


@pytest.mark.parametrize(
    "value,expected",
    [(0, 1), (1, 1), (2, 2), (3, 4), (64, 64), (65, 128), (1000, 1024)],
)
def test_next_power_of_two(value, expected):
    assert _next_power_of_two(value) == expected


@pytest.mark.parametrize(
    "sizes,rank,expect_offset,expect_local",
    [
        ([8, 8, 8, 8], 0, 0, 8),
        ([8, 8, 8, 8], 2, 16, 8),
        ([5, 0, 7, 3], 2, 5, 7),
        ([5, 0, 7, 3], 3, 12, 3),
    ],
)
def test_begin_forward_bookkeeping(
    monkeypatch, sizes, rank, expect_offset, expect_local
):
    """finalize() slices the gathered tensors with these, so they must be exact.

    An off-by-one here would silently reduce another rank's tokens into this
    rank's output, which no shape check would catch.
    """
    state = PeerScatterCombineState(top_k=4, hidden_dim=128, dtype=torch.bfloat16)
    # The buffer allocation is collective and needs real symmetric memory.
    monkeypatch.setattr(PeerScatterCombineState, "_ensure_buffer", lambda *a, **k: None)
    state.begin_forward(_FakeGroup(rank, len(sizes)), sizes, torch.device("cpu"))

    assert state.armed
    assert state.local_offset == expect_offset
    assert state.num_local_tokens == expect_local
    assert state.token_dst_rank.tolist() == [
        r for r, n in enumerate(sizes) for _ in range(n)
    ]
    assert state.token_dst_local_idx.tolist() == [i for n in sizes for i in range(n)]

    state.end_forward()
    assert not state.armed
    assert state.token_dst_rank is None


def test_capacity_is_identical_on_every_rank():
    """The symmetric allocation is collective, so capacity must not vary.

    It is derived from max(sizes), which every rank computes from the same DP
    metadata, so every rank must land on the same number.
    """
    sizes = [5, 9, 2, 9]
    capacities = {_next_power_of_two(max(sizes)) for _ in range(len(sizes))}
    assert capacities == {16}


def test_finalize_signature_matches_base():
    """The modular kernel calls finalize() positionally; drift breaks silently."""
    base = inspect.signature(MoEPrepareAndFinalizeNaiveDPEPModular.finalize)
    override = inspect.signature(MoEPrepareAndFinalizePeerScatter.finalize)
    assert list(base.parameters) == list(override.parameters)


def test_prepare_signature_matches_base():
    base = inspect.signature(MoEPrepareAndFinalizeNaiveDPEPModular.prepare)
    override = inspect.signature(MoEPrepareAndFinalizePeerScatter.prepare)
    assert list(base.parameters) == list(override.parameters)


def test_peer_kwargs_are_accepted_by_flashinfer():
    """The kwargs vLLM sends must be exactly what the FlashInfer API takes.

    This is the seam that rots first: a rename on either side would otherwise
    only show up as a TypeError in a four-GPU run.
    """
    flashinfer = pytest.importorskip("flashinfer")

    state = PeerScatterCombineState(top_k=2, hidden_dim=64, dtype=torch.bfloat16)
    state.token_dst_rank = torch.zeros(1, dtype=torch.int32)
    state.token_dst_local_idx = torch.zeros(1, dtype=torch.int32)

    class _Buf:
        tensor = torch.zeros(1)
        peer_addresses = torch.zeros(1, dtype=torch.int64)

    state._buffer = _Buf()

    accepted = set(inspect.signature(flashinfer.cute_dsl_fused_moe).parameters)
    unknown = [k for k in state.moe_kwargs() if k not in accepted]
    assert not unknown, f"flashinfer.cute_dsl_fused_moe does not accept {unknown}"


def test_peer_scatter_requires_deterministic_finalize():
    """GEMM2 must not also reduce: every slot has exactly one writer."""
    state = PeerScatterCombineState(top_k=2, hidden_dim=64, dtype=torch.bfloat16)
    state.token_dst_rank = torch.zeros(1, dtype=torch.int32)
    state.token_dst_local_idx = torch.zeros(1, dtype=torch.int32)

    class _Buf:
        tensor = torch.zeros(1)
        peer_addresses = torch.zeros(1, dtype=torch.int64)

    state._buffer = _Buf()
    assert state.moe_kwargs()["use_fused_finalize"] is False
