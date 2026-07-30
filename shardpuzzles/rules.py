"""The sharding rules: which layouts an op accepts, and what layout it produces.

Every function here looks only at Layouts (no data) and raises a PuzzleError that explains the
mistake in terms a learner can act on. There are three rules worth knowing:

* contraction (einsum, @): matching labels must be laid out identically; summing a label that is
  split over a group leaves a partial sum over that group.
* elementwise (+ - * /): dims broadcast from the right; partial sums only add to partial sums.
* view: a local reshape, so it can merge and split dims but never move sharding.
"""

from __future__ import annotations

from collections.abc import Sequence

from .errors import PuzzleError
from .layout import Dim, Layout, format_dims, same_size
from .simulation import current


def require_no_partial_max(t: Layout, op: str) -> None:
    if t.partial_max:
        raise PuzzleError(
            f"{op}([{t}]): each rank holds only a partial max over {','.join(sorted(t.partial_max))}; "
            "all_reduce(..., op='max') first"
        )


def require_reduced(t: Layout, op: str) -> None:
    """Nonlinear ops need the real value, not a per-rank piece of a sum."""
    require_no_partial_max(t, op)
    if t.partial:
        raise PuzzleError(
            f"{op}([{t}]): the input is an unreduced partial sum over {','.join(sorted(t.partial))}; "
            f"{op}(a + b) != {op}(a) + {op}(b). Reduce first."
        )


# ---------------------------------------------------------------------------
# Contraction


def contraction(
    where: str, a: Layout, labels_a: Sequence[str], b: Layout, labels_b: Sequence[str], labels_out: Sequence[str]
) -> Layout:
    """The layout of einsum(labels_a, labels_b -> labels_out). `where` names the op in errors."""
    require_no_partial_max(a, where)
    require_no_partial_max(b, where)
    if len(labels_a) != a.ndim or len(labels_b) != b.ndim:
        raise PuzzleError(f"{where}: label counts don't match [{a}] and [{b}]")
    if len(set(labels_out)) != len(labels_out) or not set(labels_out) <= set(labels_a) | set(labels_b):
        raise PuzzleError(f"{where}: output labels must be distinct input labels")

    dim_of: dict[str, Dim] = dict(zip(labels_a, a.dims))
    for label, dim in zip(labels_b, b.dims):
        if label in dim_of and not dim_of[label].same_as(dim):
            raise PuzzleError(mismatch(where, dim_of[label], dim, contracted=label not in labels_out))
        dim_of[label] = dim

    if a.partial & b.partial:
        both = ",".join(sorted(a.partial & b.partial))
        raise PuzzleError(f"{where}: both operands are partial sums over {both}; a product of sums isn't a sum")
    partial = a.partial | b.partial
    for label, dim in dim_of.items():
        if label not in labels_out and dim.split_over:  # each rank summed only its own slice
            if dim.split_over in partial:
                raise PuzzleError(f"{where}: summing over {dim} while already partial over {dim.split_over}")
            partial |= {dim.split_over}
    return Layout(tuple(dim_of[label] for label in labels_out), partial)


def mismatch(where: str, left: Dim, right: Dim, contracted: bool) -> str:
    if not same_size(left.size, right.size):
        return f"{where}: sizes don't match ({left} vs {right})"
    if contracted:
        return (
            f"{where}: the contracted dim is {left} on the left and {right} on the right. Each rank would "
            "multiply its slice by the wrong rows: gather the split side, or split the other side the same way."
        )
    return f"{where}: {left} on the left and {right} on the right must be laid out the same way"


def matmul_labels(a_ndim: int, b_ndim: int) -> tuple[list[str], list[str], list[str]]:
    """a [... m k] @ b [k n] or b [... k n] as einsum labels; b's batch dims must equal a's."""
    if a_ndim < 2 or b_ndim < 2:
        raise PuzzleError("@ needs operands with at least 2 dims")
    if b_ndim > 2 and b_ndim != a_ndim:
        raise PuzzleError("@ with a batched right operand needs the same number of dims on both sides")
    batch = [f"batch{i}" for i in range(a_ndim - 2)]
    labels_b = (batch if b_ndim > 2 else []) + ["k", "n"]
    return batch + ["m", "k"], labels_b, batch + ["m", "n"]


# ---------------------------------------------------------------------------
# Elementwise


def elementwise(a: Layout, b: Layout, op: str) -> Layout:
    """+ - * / of two tensors. Dims align from the right; size-1 replicated dims broadcast.

    + and - need both sides partial over the same groups. * and / may have one partial side
    (scaling a sum scales every piece), but / can't divide by a partial sum."""
    require_no_partial_max(a, op)
    require_no_partial_max(b, op)
    n = max(a.ndim, b.ndim)
    left = (None,) * (n - a.ndim) + a.dims
    right = (None,) * (n - b.ndim) + b.dims
    dims = []
    for x, y in zip(left, right):
        if x is None or (y is not None and x.is_one):
            dims.append(y)
        elif y is None or y.is_one or x.same_as(y):
            dims.append(x)
        else:
            raise PuzzleError(f"[{a}] {op} [{b}]: dims {x} and {y} don't line up")

    if op in ("+", "-"):
        if a.partial != b.partial:
            raise PuzzleError(
                f"[{a}] {op} [{b}]: partial-ness differs. Adding a full tensor to a partial sum "
                "counts it once per rank after the reduction."
            )
        partial = a.partial
    else:
        if a.partial & b.partial:
            raise PuzzleError(f"[{a}] {op} [{b}]: a product of two partial sums isn't the sum of the products")
        if op == "/" and b.partial:
            raise PuzzleError(f"[{a}] / [{b}]: dividing by a partial sum isn't linear")
        partial = a.partial | b.partial
    return Layout(tuple(dims), partial)


# ---------------------------------------------------------------------------
# View


def check_view(source: Sequence[Dim], target: Sequence[Dim]) -> None:
    """A view is a local reshape: it can merge or split dims, but never move sharding.

    The dims on both sides are paired up in runs of equal total size ("b (h*hd):tp" -> "b h:tp hd"
    pairs [b] with [b] and [h*hd] with [h hd]). In a run, each rank holds a contiguous chunk of the
    OUTERMOST dim only, so a sharded dim must come first on both sides. Size-1 dims are ignored."""
    where = f"view [{format_dims(source)}] -> [{format_dims(target)}]"

    # A named axis can also be singleton in this config (e.g. hd=1). It has no effect on
    # element order. Keep grouped axes so a view still cannot erase a layout annotation.
    def significant(dim: Dim) -> bool:
        return dim.group is not None or current().evaluate(dim.size) != 1

    for left, right in equal_size_runs(
        [d for d in source if significant(d)], [d for d in target if significant(d)], where
    ):
        check_run(left, right, where)


def equal_size_runs(source: list[Dim], target: list[Dim], where: str) -> list[tuple[list[Dim], list[Dim]]]:
    """Pair up consecutive dims of the two sides into runs whose sizes multiply to the same total."""
    simulation = current()

    def total(dims: list[Dim]) -> int:
        product = 1
        for d in dims:
            product *= simulation.evaluate(d.size)
        return product

    runs = []
    i = j = 0
    while i < len(source) and j < len(target):
        left, right = [source[i]], [target[j]]
        i, j = i + 1, j + 1
        while total(left) != total(right):
            if total(left) < total(right) and i < len(source):
                left.append(source[i])
                i += 1
            elif total(right) < total(left) and j < len(target):
                right.append(target[j])
                j += 1
            else:
                raise PuzzleError(f"{where}: total sizes differ")
        runs.append((left, right))
    if i < len(source) or j < len(target):
        raise PuzzleError(f"{where}: total sizes differ")
    return runs


def check_run(left: list[Dim], right: list[Dim], where: str) -> None:
    groups_left, groups_right = [d.group for d in left if d.group], [d.group for d in right if d.group]
    if groups_left != groups_right:
        raise PuzzleError(
            f"{where}: a view can't add, drop or move sharding; it's a local reshape. "
            "Use a collective (or shard) to change how a dim is split."
        )
    if len(groups_left) > 1:
        raise PuzzleError(f"{where}: more than one sharded dim in one merged group")
    if groups_left and not (left[0].group and right[0].group):
        raise PuzzleError(
            f"{where}: each rank holds a contiguous chunk of the *outermost* dim of a merged group, "
            "so the sharded dim must come first on both sides"
        )
