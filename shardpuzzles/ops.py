"""Local tensor operations with checked symbolic layouts."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Literal

import sympy as sp
import torch
import torch.nn.functional as F

from . import rules
from .errors import PuzzleError
from .layout import Dim, DimRef, Layout, Size, parse_dims, parse_size, same_size
from .simulation import current
from .tensor import Pending, ShardedTensor, contract, require_tensor


def shard(t: ShardedTensor, spec: str) -> ShardedTensor:
    """Keep only this rank's slice of a replicated tensor (free, no communication)."""
    t = require_tensor(t, "shard")
    dims = parse_dims(spec)
    if len(dims) != t.ndim or not all(same_size(new.size, old.size) for new, old in zip(dims, t.dims)):
        raise PuzzleError(f"shard: [{spec}] doesn't match the tensor's sizes [{t.spec}]")
    for new, old in zip(dims, t.dims):
        if old.group and new.group != old.group:
            raise PuzzleError(f"shard: {old} in [{t.spec}] is already split; shard only slices whole dims")
        if new.group and new.group in t.partial:
            raise PuzzleError(f"shard: [{t.spec}] is a partial sum over {new.group}; reduce_scatter it instead")
    simulation = current()
    shards = {}
    for rank, x in t.shards.items():
        for i, (new, old) in enumerate(zip(dims, t.dims)):
            if new.group and not old.group:
                n = new.local_size()
                x = x.narrow(i, simulation.coordinate(rank, new.group) * n, n)
        shards[rank] = x.contiguous()
    return t.with_data(shards, dims)


def cat(tensors: Sequence[ShardedTensor], dim: DimRef) -> ShardedTensor:
    """Concatenate along one dim. Along a split dim, every rank concatenates its own slices."""
    tensors = [require_tensor(t, "cat") for t in tensors]
    if not tensors:
        raise PuzzleError("cat: needs at least one tensor")
    first = tensors[0]
    i = first.dim_index(dim)
    for t in tensors[1:]:
        if t.ndim != first.ndim or not first.layout.with_dim(i, None).matches(t.layout.with_dim(i, None)):
            raise PuzzleError(f"cat: [{first.spec}] and [{t.spec}] differ outside dim {i}")
    pieces = [t.dims[i] for t in tensors]
    group = pieces[0].group
    if any(d.group != group for d in pieces):
        raise PuzzleError(f"cat: the pieces are split differently along dim {i}: {', '.join(map(str, pieces))}")
    joined = Dim(sp.simplify(sp.Add(*(d.size for d in pieces))), group)
    shards = {r: torch.cat([t.shards[r] for t in tensors], dim=i) for r in first.shards}
    return ShardedTensor(first.layout.with_dim(i, joined), shards, is_weight=any(t.is_weight for t in tensors))


def repeat_interleave(t: ShardedTensor, n: Size, dim: DimRef) -> ShardedTensor:
    """Repeat each slice along `dim` n times (e.g. kv heads -> query heads)."""
    t = require_tensor(t, "repeat_interleave")
    i = t.dim_index(dim)
    factor = parse_size(n)
    times = current().evaluate(factor)
    if times <= 0:
        raise PuzzleError("repeat_interleave: repeat count must be positive")
    dims = t.layout.with_dim(i, t.dims[i].resized(t.dims[i].size * factor)).dims
    return t.with_data(t.map_shards(lambda x: x.repeat_interleave(times, dim=i)), dims)


def pointwise(fn: Callable[[torch.Tensor], torch.Tensor], name: str) -> Callable[[ShardedTensor], ShardedTensor]:
    def op(t: ShardedTensor) -> ShardedTensor:
        t = require_tensor(t, name)
        rules.require_reduced(t.layout, name)
        return t.with_data(t.map_shards(fn))

    op.__name__ = name
    op.__doc__ = f"{name}, elementwise. Needs a fully reduced input."
    return op


gelu = pointwise(F.gelu, "gelu")
silu = pointwise(F.silu, "silu")
relu = pointwise(torch.relu, "relu")
sigmoid = pointwise(torch.sigmoid, "sigmoid")
exp = pointwise(torch.exp, "exp")
log = pointwise(torch.log, "log")


def softmax(t: ShardedTensor, dim: DimRef) -> ShardedTensor:
    t = require_tensor(t, "softmax")
    rules.require_reduced(t.layout, "softmax")
    i = t.dim_index(dim)
    if t.dims[i].split_over:
        raise PuzzleError(f"softmax over {t.dims[i]}: each rank only sees part of the row it normalizes")
    return t.with_data(t.map_shards(lambda x: torch.softmax(x, dim=i)))


def reduce_dim(t: ShardedTensor, dim: DimRef, keepdim: bool, kind: Literal["sum", "max"]) -> Pending:
    """Sum or max over one dim. Over a split dim each rank only reduced its own slice: a partial result."""
    i = t.dim_index(dim)
    d = t.dims[i]
    layout = t.layout.with_dim(i, Dim(sp.Integer(1)) if keepdim else None)
    if kind == "sum":
        shards = t.map_shards(lambda x: x.sum(i, keepdim=keepdim))
    else:
        shards = t.map_shards(lambda x: x.amax(i, keepdim=keepdim))
    if d.split_over and kind == "sum":
        if d.split_over in t.partial:
            raise PuzzleError(f"sum over {d}: already a partial sum over {d.split_over}")
        layout = dataclasses.replace(layout, partial=t.partial | {d.split_over})
    elif d.split_over:
        layout = dataclasses.replace(layout, partial_max=t.partial_max | {d.split_over})
    return Pending(ShardedTensor(layout, shards), kind)


def sum(t: ShardedTensor, dim: DimRef, keepdim: bool = False) -> Pending:  # noqa: A001
    """Sum over `dim`. Over a dim split across a group, each rank holds a partial sum."""
    t = require_tensor(t, "sum")
    rules.require_no_partial_max(t.layout, "sum")
    return reduce_dim(t, dim, keepdim, "sum")


def amax(t: ShardedTensor, dim: DimRef, keepdim: bool = False) -> Pending:
    """Max over `dim`. Over a split dim each rank holds a partial max (all_reduce it with op='max')."""
    t = require_tensor(t, "amax")
    rules.require_reduced(t.layout, "amax")
    return reduce_dim(t, dim, keepdim, "max")


def causal_mask(spec: str) -> ShardedTensor:
    """Additive [queries keys] mask: 0 where key position <= query position, -inf otherwise.

    The query dim may be split; each rank gets its contiguous chunk of global positions.
    """
    dims = parse_dims(spec)
    if len(dims) != 2 or dims[1].group:
        raise PuzzleError("causal_mask takes [queries keys] with whole keys: 's s' or 's:tp s'")
    simulation = current()
    mask = torch.full([simulation.evaluate(d.size) for d in dims], float("-inf"), dtype=torch.float64).triu(1)
    rows = dims[0].local_size()
    shards = {}
    for rank in simulation.ranks:
        offset = simulation.coordinate(rank, dims[0].group) * rows if dims[0].group else 0
        shards[rank] = mask[offset : offset + rows].clone()
    return ShardedTensor(Layout(dims), shards)


def full(spec: str, value: float) -> ShardedTensor:
    """A tensor laid out as `spec` with every entry equal to `value`."""
    layout = Layout(parse_dims(spec))
    shape = layout.local_shape()
    return ShardedTensor(layout, {r: torch.full(shape, value, dtype=torch.float64) for r in current().ranks})


def size(expr: Size) -> int:
    """The current value of a size expression, e.g. size("d") or size("tp")."""
    return current().evaluate(parse_size(expr, allow_group_sizes=True))


def einsum(spec: str, a: ShardedTensor, b: ShardedTensor) -> Pending:
    """Contract two tensors with named dims: einsum("b s d, b s f -> d f", x, dy).

    Each operand's dims get a label, in order. Labels missing from the output are summed over.
    Summing a dim that is split over a group leaves a partial sum over that group, because each
    rank only summed its own slice."""
    a, b = require_tensor(a, "einsum"), require_tensor(b, "einsum")
    try:
        inputs, output = spec.split("->")
        side_a, side_b = inputs.split(",")
    except ValueError:
        raise PuzzleError(f"einsum spec must look like 'b s d, b s f -> d f', got {spec!r}") from None
    labels = [side_a.split(), side_b.split(), output.split()]
    return Pending(contract(f"einsum({spec!r})", a, b, *labels), "einsum")
