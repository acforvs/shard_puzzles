"""Collectives move data between simulated ranks and record modeled send volume.

For bf16 payloads and ring/pairwise algorithms, with group size g:
    all_gather       2*n_out*(g-1)/g
    reduce_scatter   2*n_in*(g-1)/g
    all_reduce       4*n*(g-1)/g
    all_to_all       2*n_in*(g-1)/g
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import sympy as sp
import torch

from . import rules
from .errors import PuzzleError
from .layout import Dim, DimRef, Layout
from .simulation import BYTES_PER_ELEMENT, GroupRef, Rank, current, group_name, size_symbol
from .tensor import Pending, ShardedTensor, Shards, require_tensor


def finish(
    op: str,
    group: str,
    layout: Layout,
    shards: Shards,
    bytes_formula: sp.Expr,
    free: bool = False,
    is_weight: bool = False,
    senders: set[Rank] | None = None,
) -> Pending:
    """Log the collective, and hand its result back for the learner to .expect().

    free: integer metadata (token positions) costs nothing in the byte model."""
    comms = () if free else (current().record_comm(op, group, bytes_formula, senders),)
    return Pending(ShardedTensor(layout, shards, is_weight=is_weight), f"{op}({group})", comms)


def ring_fraction(group: str) -> sp.Expr:
    """The share of a ring collective's data that leaves each rank: (g-1)/g."""
    g = size_symbol(group)
    return (g - 1) / g


def sent(layout: Layout) -> sp.Expr:
    """Bytes in one rank's copy of a tensor with this layout."""
    return BYTES_PER_ELEMENT * layout.local_numel_formula()


def all_gather(t: ShardedTensor, dim: DimRef, group: GroupRef) -> Pending:
    """Make a dim that is split over `group` whole again: [s:tp] -> [s]."""
    t, group = require_tensor(t, "all_gather"), group_name(group)
    rules.require_no_partial_max(t.layout, "all_gather")
    i = t.dim_index(dim)
    if t.dims[i].split_over != group:
        raise PuzzleError(f"all_gather over {group}: {t.dims[i]} isn't split over {group} in [{t.spec}]")
    if group in t.partial:
        raise PuzzleError(f"all_gather: [{t.spec}] is a partial sum over {group}; reduce it instead")
    shards = {}
    for members in current().communicators(group):
        whole = torch.cat([t.shards[r] for r in members], dim=i)
        shards |= {r: whole.clone() for r in members}
    layout = t.layout.with_dim(i, Dim(t.dims[i].size))
    return finish("all_gather", group, layout, shards, sent(layout) * ring_fraction(group), is_weight=t.is_weight)


def reduce_scatter(t: ShardedTensor, dim: DimRef, group: GroupRef) -> Pending:
    """Sum a partial tensor over `group` and split the result along `dim`: [s] partial(tp) -> [s:tp]."""
    t, group = require_tensor(t, "reduce_scatter"), group_name(group)
    if group not in t.partial:
        raise PuzzleError(
            f"reduce_scatter over {group}: [{t.spec}] isn't a partial sum over {group}, so there is nothing to "
            "reduce. To split a replicated tensor use shard(), which is free."
        )
    i = t.dim_index(dim)
    if t.dims[i].group:
        raise PuzzleError(f"reduce_scatter along {t.dims[i]}: that dim is already split")
    split = Dim(t.dims[i].size, group)
    n = split.local_size()
    shards = {}
    for members in current().communicators(group):
        total = torch.stack([t.shards[r] for r in members]).sum(0)
        shards |= {r: total.narrow(i, k * n, n).contiguous() for k, r in enumerate(members)}
    dims = t.dims[:i] + (split,) + t.dims[i + 1 :]
    layout = dataclasses.replace(t.layout, dims=dims, partial=t.partial - {group})
    return finish("reduce_scatter", group, layout, shards, sent(t.layout) * ring_fraction(group), is_weight=t.is_weight)


def all_reduce(t: ShardedTensor, group: GroupRef, op: Literal["sum", "max"] = "sum") -> Pending:
    """Reduce a partial sum (op="sum") or partial max (op="max") over `group`; every rank gets the result."""
    t, group = require_tensor(t, "all_reduce"), group_name(group)
    if op not in ("sum", "max"):
        raise PuzzleError("all_reduce: op must be 'sum' or 'max'")
    pending, other = (t.partial, t.partial_max) if op == "sum" else (t.partial_max, t.partial)
    if group not in pending:
        if group in other:
            flipped = "max" if op == "sum" else "sum"
            raise PuzzleError(f"all_reduce over {group}: [{t.spec}] is a partial {flipped}; use op={flipped!r}")
        raise PuzzleError(f"all_reduce over {group}: [{t.spec}] isn't a partial {op} over {group}; nothing to reduce")
    shards = {}
    for members in current().communicators(group):
        values = torch.stack([t.shards[r] for r in members])
        total = values.sum(0) if op == "sum" else values.amax(0)
        shards |= {r: total.clone() for r in members}
    if op == "sum":
        layout = dataclasses.replace(t.layout, partial=t.partial - {group})
    else:
        layout = dataclasses.replace(t.layout, partial_max=t.partial_max - {group})
    return finish("all_reduce", group, layout, shards, 2 * sent(t.layout) * ring_fraction(group), is_weight=t.is_weight)


def all_to_all(t: ShardedTensor, gather_dim: DimRef, scatter_dim: DimRef, group: GroupRef) -> Pending:
    """Re-shard: `gather_dim` (split over group) becomes whole, `scatter_dim` (whole) becomes split."""
    t, group = require_tensor(t, "all_to_all"), group_name(group)
    rules.require_no_partial_max(t.layout, "all_to_all")
    gi, si = t.dim_index(gather_dim), t.dim_index(scatter_dim)
    if gi == si:
        raise PuzzleError("all_to_all: gather_dim and scatter_dim must differ")
    if t.dims[gi].split_over != group:
        raise PuzzleError(f"all_to_all: gather_dim {t.dims[gi]} isn't split over {group}")
    if t.dims[si].group:
        raise PuzzleError(f"all_to_all: scatter_dim {t.dims[si]} is already split")
    if group in t.partial:
        raise PuzzleError(f"all_to_all: [{t.spec}] is a partial sum over {group}")
    shards = {}
    for members in current().communicators(group):
        # the k-th chunk (along scatter_dim) of every sender goes to the k-th receiver
        chunks = {r: torch.chunk(t.shards[r], len(members), dim=si) for r in members}
        for k, receiver in enumerate(members):
            shards[receiver] = torch.cat([chunks[sender][k] for sender in members], dim=gi).contiguous()
    layout = t.layout.with_dim(gi, Dim(t.dims[gi].size)).with_dim(si, Dim(t.dims[si].size, group))
    return finish("all_to_all", group, layout, shards, sent(t.layout) * ring_fraction(group), is_weight=t.is_weight)
