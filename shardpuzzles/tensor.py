"""ShardedTensor: a Layout plus one small torch tensor per simulated rank.

Results whose layout is worth thinking about (matmuls, einsums, collectives, anything that
can create a partial sum) come back as a `Pending` result. It can only be used after you state
its layout with `.expect("b s d:tp", partial=..., bytes=...)`.
"""

from __future__ import annotations

import string
from collections.abc import Callable, Sequence
from typing import Any, NoReturn, TypeVar

import sympy as sp
import torch

from . import rules
from .errors import PuzzleError, Unfinished, caller_location
from .layout import Dim, DimRef, GroupsRef, Layout, Size, format_size, parse_dims, parse_size, same_size
from .simulation import CommRecord, Phase, Rank, current

Shards = dict[Rank, torch.Tensor]
T = TypeVar("T")
Scalar = int | float


class ShardedTensor:
    """A global tensor, as the ranks hold it. Built in on_init (or passed to it), it counts as a weight."""

    def __init__(self, layout: Layout, shards: Shards, is_weight: bool | None = None):
        self.layout = layout
        self.shards = shards  # rank -> this rank's local data
        self.is_weight = current().phase == Phase.INIT if is_weight is None else is_weight
        if set(shards) != set(current().ranks):
            raise PuzzleError("a puzzle tensor must have exactly one local tensor for every simulated rank")
        expected = layout.local_shape()
        for rank, shard in shards.items():
            if tuple(shard.shape) != expected:
                raise AssertionError(f"engine bug: rank {rank} holds {tuple(shard.shape)}, expected {expected}")
        current().record_tensor(layout, self.local_numel)

    # -- layout

    @property
    def dims(self) -> tuple[Dim, ...]:
        return self.layout.dims

    @property
    def partial(self) -> frozenset[str]:
        return self.layout.partial

    @property
    def partial_max(self) -> frozenset[str]:
        return self.layout.partial_max

    @property
    def spec(self) -> str:
        return str(self.layout)

    @property
    def ndim(self) -> int:
        return self.layout.ndim

    @property
    def local_shape(self) -> tuple[int, ...]:
        return self.layout.local_shape()

    @property
    def local_numel(self) -> int:
        return torch.Size(self.local_shape).numel()

    @property
    def is_integer(self) -> bool:
        return next(iter(self.shards.values())).dtype == torch.int64

    def dim_index(self, dim: DimRef) -> int:
        return self.layout.index(dim)

    def __repr__(self) -> str:
        return f"ShardedTensor[{self.spec}]"

    # -- building results

    def map_shards(self, fn: Callable[[torch.Tensor], T]) -> dict[Rank, T]:
        return {rank: fn(shard) for rank, shard in self.shards.items()}

    def with_data(self, shards: Shards, dims: Sequence[Dim] | None = None) -> ShardedTensor:
        """A result with new data (and optionally new dims), partial over the same groups."""
        return ShardedTensor(self.layout if dims is None else self.layout.with_dims(dims), shards, self.is_weight)

    # -- checks

    def expect(
        self, spec: str, partial: GroupsRef = None, partial_max: GroupsRef = None, bytes: Size | None = None
    ) -> ShardedTensor:
        """State this tensor's layout; fails right here if it's different. Returns the tensor.

        A layout says how the ranks of a mesh group hold one global tensor. Sizes are always global.
            "b s d"                  replicated: every rank holds the whole tensor
            "b s:tp d"               split: tp rank r holds the r-th slice of s; the slices concatenate to it
            "b s d", partial=tp      partial sum: every tp rank holds a full-size [b s d] and the real tensor
                                     is their SUM (each rank summed over only its slice of a split dim)
            "b s 1", partial_max=tp  the same, but the real tensor is their max
        The checker works out the actual layout itself, and compares."""
        check_layout(self.layout, spec, partial, partial_max)
        if bytes is not None:
            raise PuzzleError("bytes= only applies to the result of a collective")
        return self

    # -- arithmetic

    def __matmul__(self, other: ShardedTensor) -> Pending:
        other = require_tensor(other, "@")
        labels = rules.matmul_labels(self.ndim, other.ndim)
        simulation = current()
        if other.is_weight and not self.is_weight and simulation.phase == Phase.FORWARD:
            simulation.weight_matmuls += 1
        return Pending(contract(f"[{self.spec}] @ [{other.spec}]", self, other, *labels), "@")

    def elementwise(self, other: ShardedTensor | Scalar, fn: Callable, op: str) -> ShardedTensor:
        if isinstance(other, (int, float)):
            rules.require_no_partial_max(self.layout, op)
            if op in ("+", "-") and self.partial and other != 0:
                raise PuzzleError(
                    f"[{self.spec}] {op} number: the number would be added once per rank of "
                    f"{','.join(sorted(self.partial))} after the reduction"
                )
            return self.with_data(self.map_shards(lambda x: fn(x, other)))
        other = require_tensor(other, op)
        layout = rules.elementwise(self.layout, other.layout, op)
        return ShardedTensor(layout, {r: fn(self.shards[r], other.shards[r]) for r in self.shards})

    def __add__(self, other: ShardedTensor | Scalar) -> ShardedTensor:
        return self.elementwise(other, torch.add, "+")

    def __radd__(self, other: Scalar) -> ShardedTensor:
        return self.elementwise(other, torch.add, "+")

    def __sub__(self, other: ShardedTensor | Scalar) -> ShardedTensor:
        return self.elementwise(other, torch.sub, "-")

    def __rsub__(self, other: Scalar) -> ShardedTensor:
        return (-self).elementwise(other, torch.add, "+")

    def __mul__(self, other: ShardedTensor | Scalar) -> ShardedTensor:
        return self.elementwise(other, torch.mul, "*")

    def __rmul__(self, other: Scalar) -> ShardedTensor:
        return self.elementwise(other, torch.mul, "*")

    def __truediv__(self, other: ShardedTensor | Scalar) -> ShardedTensor:
        return self.elementwise(other, torch.div, "/")

    def __neg__(self) -> ShardedTensor:
        return self * -1.0

    # -- layout ops

    def transpose(self, i: DimRef, j: DimRef) -> ShardedTensor:
        i, j = self.dim_index(i), self.dim_index(j)
        dims = list(self.dims)
        dims[i], dims[j] = dims[j], dims[i]
        return self.with_data(self.map_shards(lambda x: x.transpose(i, j).contiguous()), dims)

    def view(self, spec: str) -> ShardedTensor:
        """Reshape to `spec` (a local reshape: it can't move sharding)."""
        dims = parse_dims(spec)
        rules.check_view(self.dims, dims)
        shape = tuple(d.local_size() for d in dims)
        return self.with_data(self.map_shards(lambda x: x.reshape(shape)), dims)

    def split(self, sizes: Sequence[Size], dim: DimRef) -> tuple[ShardedTensor, ...]:
        """Split one dim into pieces of the given global sizes; on a sharded dim, every rank splits its slice."""
        i = self.dim_index(dim)
        whole = self.dims[i]
        pieces = [whole.resized(parse_size(s)) for s in sizes]
        if not same_size(sum((p.size for p in pieces), sp.Integer(0)), whole.size):
            raise PuzzleError(f"split sizes {' + '.join(format_size(p.size) for p in pieces)} don't add up to {whole}")
        chunks = self.map_shards(lambda x: torch.split(x, [p.local_size() for p in pieces], dim=i))
        return tuple(
            ShardedTensor(self.layout.with_dim(i, piece), {r: c[k].contiguous() for r, c in chunks.items()})
            for k, piece in enumerate(pieces)
        )


def contract(
    where: str, a: ShardedTensor, b: ShardedTensor, labels_a: list[str], labels_b: list[str], labels_out: list[str]
) -> ShardedTensor:
    """einsum on every rank, with the layout from rules.contraction."""
    layout = rules.contraction(where, a.layout, labels_a, b.layout, labels_b, labels_out)
    letter = dict(zip(dict.fromkeys(labels_a + labels_b), string.ascii_letters))
    equation = "".join(letter[x] for x in labels_a) + "," + "".join(letter[x] for x in labels_b)
    equation += "->" + "".join(letter[x] for x in labels_out)
    shards = {r: torch.einsum(equation, a.shards[r], b.shards[r]) for r in a.shards}
    return ShardedTensor(layout, shards, is_weight=False)


# ---------------------------------------------------------------------------
# Pending results


class Pending:
    """A result whose layout must be stated with .expect(...) before it can be used.

    The value is already computed; .expect checks your spec against it and returns the tensor.
    Pending results can be added or subtracted (e.g. the sum of several matmuls) and checked once."""

    __slots__ = ("tensor", "made_by", "comms", "location")

    def __init__(self, tensor: ShardedTensor, made_by: str, comms: tuple[CommRecord, ...] = ()):
        self.tensor = tensor
        self.made_by = made_by
        self.comms = comms
        self.location = caller_location()

    def expect(
        self, spec: str, partial: GroupsRef = None, partial_max: GroupsRef = None, bytes: Size | None = None
    ) -> ShardedTensor:
        """State the layout of this result (and, for a collective, the bytes each rank sent): this is
        what makes it usable. Fails right here if the statement is wrong. Returns the checked tensor.

        A layout says how the ranks of a mesh group hold one global tensor. Sizes are always global.
            "b s d"                  replicated: every rank holds the whole tensor
            "b s:tp d"               split: tp rank r holds the r-th slice of s; the slices concatenate to it
            "b s d", partial=tp      partial sum: every tp rank holds a full-size [b s d] and the real tensor
                                     is their SUM (each rank summed over only its slice of a split dim)
            "b s 1", partial_max=tp  the same, but the real tensor is their max
        The checker works out the actual layout itself, and compares.

        bytes: the maximum sent by any rank for this operation, as a formula in the size names
        ("4*b*s*d*(tp-1)/tp"). Uniform collectives send the same amount on every rank."""
        check_layout(self.tensor.layout, spec, partial, partial_max)
        if self.comms:
            if bytes is None and current().require_bytes:
                raise PuzzleError(
                    f".expect(...) on the result of {self.made_by}: this puzzle also asks for the bytes each rank "
                    "sent; add bytes=... to .expect(...)."
                )
            if bytes is not None:
                check_bytes(self.comms, bytes)
        elif bytes is not None:
            raise PuzzleError(f"bytes=: the result of {self.made_by} involved no communication")
        return self.tensor

    def __add__(self, other: Pending) -> Pending:
        return self.combine(other, "+")

    def __sub__(self, other: Pending) -> Pending:
        return self.combine(other, "-")

    def combine(self, other: Pending, op: str) -> Pending:
        if not isinstance(other, Pending):
            self.refuse()
        total = self.tensor + other.tensor if op == "+" else self.tensor - other.tensor
        return Pending(total, f"{self.made_by} {op} {other.made_by}", self.comms + other.comms)

    def refuse(self, *_args: object) -> NoReturn:
        line = f" (line {self.location[1]})" if self.location else ""
        raise PuzzleError(
            f"the result of {self.made_by}{line} hasn't been checked yet. "
            'State its layout first: (...).expect("dims", partial=...)'
        )

    # Any other use of a Pending result is a mistake: attribute access and every operator refuse.
    def __getattr__(self, _name: str) -> NoReturn:
        self.refuse()

    __matmul__ = __rmatmul__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = refuse
    __radd__ = __rsub__ = __neg__ = __getitem__ = __iter__ = __bool__ = refuse

    def __repr__(self) -> str:
        return f"Pending({self.made_by}: call .expect)"


class Todo:
    """Marks a blank in a puzzle stub. Any use of it points at the line to fill in."""

    def refuse(self, *_args: object) -> NoReturn:
        raise Unfinished("this is still a TODO: replace it with code")

    __getattr__ = __matmul__ = __rmatmul__ = __add__ = __radd__ = __sub__ = __rsub__ = refuse
    __mul__ = __rmul__ = __truediv__ = __neg__ = __getitem__ = __iter__ = __bool__ = __call__ = refuse

    def __repr__(self) -> str:
        return "TODO"


TODO: Any = Todo()  # typed Any so that a stub with blanks type-checks


def require_tensor(value: object, op: str) -> ShardedTensor:
    """Every op calls this on its tensor inputs: it explains Pending and TODO misuse."""
    if isinstance(value, ShardedTensor):
        return value
    if isinstance(value, Pending):
        value.refuse()
    if value is TODO or value is Ellipsis:
        raise Unfinished(f"{op}: an input is still a TODO; fill it in")
    raise PuzzleError(f"{op}: expected a tensor, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Checks


def check_layout(actual: Layout, spec: str, partial: GroupsRef, partial_max: GroupsRef) -> None:
    expected = Layout.parse(spec, partial, partial_max)
    if expected.matches(actual):
        return
    hint = ""
    if expected.same_dims(actual) and actual.partial != expected.partial:
        missing, extra = sorted(actual.partial - expected.partial), sorted(expected.partial - actual.partial)
        if missing:
            groups = ",".join(missing)
            hint = (
                f"\n  The dims are right, but this is a partial sum over {groups}: some sum ran over a dim split "
                f"across {groups}, so each rank added up only its own slice. The real tensor is the sum over "
                f"the ranks of {groups}. Say partial={groups} (or reduce it first)."
            )
        else:
            groups = ",".join(extra)
            hint = f"\n  The dims are right, but it isn't a partial sum over {groups}: each rank's values are complete."
    elif expected.same_dims(actual) and actual.partial_max != expected.partial_max:
        hint = "\n  The dims are right; the difference is partial_max= (each rank holds a max of only its slice)."
    raise PuzzleError(f"expect failed\n  expected: {expected}\n  actual:   {actual}{hint}")


def check_bytes(comms: tuple[CommRecord, ...], expected: Size) -> None:
    """`expected` must equal the bytes each rank sent to produce this result (group sizes allowed)."""
    want = parse_size(expected, allow_group_sizes=True)
    actual = sp.simplify(sp.Add(*(c.formula for c in comms)))
    if not same_size(want, actual):
        ops = " + ".join(f"{c.op}({c.group})" for c in comms)
        raise PuzzleError(
            f"incorrect byte count for {ops}\n  submitted: {format_size(want)}\n"
            "  Use --hint for guidance or --solution for a worked answer."
        )


# ---------------------------------------------------------------------------
# Global <-> per-rank (used by the checker)


def from_global(x: torch.Tensor, spec: str, is_weight: bool = False) -> ShardedTensor:
    """Give every rank its slice of a global tensor laid out as `spec`."""
    simulation = current()
    layout = Layout(parse_dims(spec))
    sizes = layout.local_shape()
    shards = {}
    for rank in simulation.ranks:
        shard = x
        for i, (dim, n) in enumerate(zip(layout.dims, sizes)):
            if dim.split_over:
                shard = shard.narrow(i, simulation.coordinate(rank, dim.split_over) * n, n)
        shards[rank] = shard.clone()
    return ShardedTensor(layout, shards, is_weight)


def to_global(t: ShardedTensor) -> torch.Tensor:
    """Reassemble the global tensor, verifying that replicated copies agree across ranks."""
    if t.partial or t.partial_max:
        raise PuzzleError(f"result [{t.spec}] still holds per-rank partial values; reduce it first")
    simulation = current()
    shape = [simulation.evaluate(d.size) for d in t.dims]
    out = torch.empty(shape, dtype=torch.float64)
    written = torch.zeros(shape, dtype=torch.bool)
    sizes = t.local_shape
    for rank in simulation.ranks:
        index = []
        for dim, n in zip(t.dims, sizes):
            start = simulation.coordinate(rank, dim.group) * n if dim.group else 0
            index.append(slice(start, start + n))
        already, mine, covered = out[tuple(index)], t.shards[rank], written[tuple(index)]
        if covered.any() and not torch.allclose(already[covered], mine[covered], rtol=1e-7, atol=1e-9):
            raise PuzzleError(f"ranks disagree on the replicated part of the result [{t.spec}]")
        out[tuple(index)] = mine
        written[tuple(index)] = True
    if not written.all():
        raise PuzzleError(f"ranks do not cover every entry of the global result [{t.spec}]")
    return out
