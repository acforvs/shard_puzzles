"""Layouts: symbolic dims, how they are split across ranks, and spec strings.

A spec string lists a tensor's dims: "b s:tp (h*hd)". Sizes are always global; `:tp` says the
dim is split across the tp group, so each rank holds 1/tp of it.

A Layout is the dims plus the groups over which each rank holds an unreduced partial sum
(`partial`) or partial max (`partial_max`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from .errors import PuzzleError, Unfinished
from .simulation import GROUP_NAMES, Group, GroupRef, current, group_name, size_symbol

DimRef = int | str  # a dim by position, or by its size ("s", "h*hd")
Size = int | str  # a size expression over the puzzle's dim names, e.g. "h/hkv"
GroupsRef = GroupRef | Iterable[GroupRef] | None  # partial=tp, partial=[tp, dp]


# ---------------------------------------------------------------------------
# Sizes


def parse_size(text: Size, allow_group_sizes: bool = False) -> sp.Expr:
    """'h*hd' -> a sympy expression over the puzzle's dim names (and tp/dp if allowed)."""
    if isinstance(text, int):
        return sp.Integer(text)
    if "?" in text:
        raise Unfinished(f"fill in the ??? in {text!r}")
    return parse_size_cached(text, current().dim_names, allow_group_sizes)


@cache
def parse_size_cached(text: str, dim_names: tuple[str, ...], allow_group_sizes: bool) -> sp.Expr:
    names: dict[str, Any] = {name: size_symbol(name) for name in dim_names}
    if allow_group_sizes:
        names |= {name: size_symbol(name) for name in GROUP_NAMES}
    try:
        expr = parse_expr(text, local_dict=names)
    except (sp.SympifyError, SyntaxError, TypeError) as exc:
        raise PuzzleError(f"can't parse size {text!r}: {exc}") from None
    unknown = sorted(str(s) for s in expr.free_symbols if s not in names.values())
    if set(unknown) & set(GROUP_NAMES):
        raise PuzzleError(f"{text!r}: write sharding as `s:tp`, not `s/tp`; sizes are always global")
    if unknown:
        raise PuzzleError(f"{text!r}: unknown dim name(s) {unknown}; known: {' '.join(dim_names)}")
    return sp.simplify(expr)


@cache
def same_size(a: sp.Expr, b: sp.Expr) -> bool:
    return sp.simplify(a - b) == 0


def format_size(expr: sp.Expr) -> str:
    return str(expr).replace(" ", "")


# ---------------------------------------------------------------------------
# Dims


@dataclass(frozen=True)
class Dim:
    size: sp.Expr  # global size
    group: str | None = None  # the group it is split over
    text: str | None = field(default=None, compare=False)  # how it was written, for display

    @property
    def split_over(self) -> str | None:
        """The group this dimension is split across, or None if it is replicated."""
        return self.group

    @property
    def is_one(self) -> bool:
        return not self.group and same_size(self.size, sp.Integer(1))

    def same_as(self, other: Dim) -> bool:
        return same_size(self.size, other.size) and self.group == other.group

    def resized(self, size: sp.Expr) -> Dim:
        """The same sharding, a different size."""
        return Dim(sp.simplify(size), self.group)

    def local_size(self) -> int:
        """How many entries of this dim one rank holds."""
        simulation = current()
        total = simulation.evaluate(self.size)
        if total < 0:
            raise PuzzleError(f"{self}: a dimension cannot have negative size ({total})")
        if not self.split_over:
            return total
        parts = simulation.group_size(self.split_over)
        if total % parts:
            raise PuzzleError(f"{self}: {format_size(self.size)}={total} isn't divisible by {self.split_over}={parts}")
        return total // parts

    def local_size_formula(self) -> sp.Expr:
        return self.size / size_symbol(self.split_over) if self.split_over else self.size

    def __str__(self) -> str:
        text = self.text or format_size(self.size)
        if not self.group:
            return text
        if not (self.size.is_Symbol or self.size.is_Integer):
            text = f"({text})"
        return text + ":" + self.group


def parse_dim(token: str) -> Dim:
    """'s' or '(h*hd):tp' -> Dim."""
    size_text, found, group_text = token.rpartition(":")
    if found and group_text.isidentifier() and size_text.count("(") == size_text.count(")"):
        group = group_name(group_text)
        current().group_size(group)
        return Dim(parse_size(size_text), group, unwrap(size_text))
    return Dim(parse_size(token), text=unwrap(token))


def unwrap(text: str) -> str:
    """'(h*hd)' -> 'h*hd': the size as the learner wrote it, for display."""
    text = text.replace(" ", "")
    return text[1:-1] if text.startswith("(") and text.endswith(")") and text.count("(") == 1 else text


def spec_tokens(spec: str) -> list[str]:
    """Split on spaces that aren't inside parentheses: 'b (h * hd):tp' -> ['b', '(h * hd):tp']."""
    tokens, depth, token = [], 0, ""
    for char in spec.strip():
        depth += (char == "(") - (char == ")")
        if char.isspace() and depth == 0:
            if token:
                tokens.append(token)
            token = ""
        else:
            token += char
    if token:
        tokens.append(token)
    if depth != 0:
        raise PuzzleError(f"unbalanced parentheses in {spec!r}")
    return tokens


def parse_dims(spec: str) -> tuple[Dim, ...]:
    if not isinstance(spec, str):
        raise PuzzleError(f"expected a spec string like 'b s:tp d', got {spec!r}")
    if "?" in spec:
        raise Unfinished(f"fill in the ??? in {spec!r}")
    return tuple(parse_dim(token) for token in spec_tokens(spec))


def parse_groups(groups: GroupsRef) -> frozenset[str]:
    """`tp`, `[tp, dp]` or None -> a set of group names."""
    if groups is None:
        return frozenset()
    if groups == "???":
        raise Unfinished("fill in the ??? first")
    names = (
        frozenset({group_name(groups)})
        if isinstance(groups, (str, Group))
        else frozenset(group_name(g) for g in groups)
    )
    for name in names:
        current().group_size(name)
    return names


def format_dims(dims: Iterable[Dim]) -> str:
    return " ".join(str(d) for d in dims) or "(scalar)"


# ---------------------------------------------------------------------------
# Layouts


@dataclass(frozen=True)
class Layout:
    dims: tuple[Dim, ...]
    partial: frozenset[str] = frozenset()  # groups over which each rank holds an unreduced sum
    partial_max: frozenset[str] = frozenset()  # ... or an unreduced max

    def __post_init__(self) -> None:
        split = [d.split_over for d in self.dims if d.split_over]
        if len(split) != len(set(split)):
            raise PuzzleError(
                f"[{format_dims(self.dims)}]: one mesh group cannot split two dimensions of the same tensor; "
                "that stores only diagonal blocks. Use different mesh groups for a 2-D split."
            )
        used = {d.group for d in self.dims if d.group}
        for group in used | self.partial | self.partial_max:
            current().group_size(group)
        conflict = used & (self.partial | self.partial_max)
        if conflict:
            raise PuzzleError(
                f"[{format_dims(self.dims)}]: cannot also be partial over {','.join(sorted(conflict))}; "
                "those ranks hold different entries, not contributions to the same entries."
            )
        if self.partial and self.partial_max:
            raise PuzzleError("mixed partial sums and partial maxima are unsupported: finish one reduction first")

    @staticmethod
    def parse(spec: str, partial: GroupsRef = None, partial_max: GroupsRef = None) -> Layout:
        return Layout(parse_dims(spec), parse_groups(partial), parse_groups(partial_max))

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def same_dims(self, other: Layout) -> bool:
        return self.ndim == other.ndim and all(a.same_as(b) for a, b in zip(self.dims, other.dims))

    def matches(self, other: Layout) -> bool:
        return self.same_dims(other) and (self.partial, self.partial_max) == (other.partial, other.partial_max)

    def index(self, dim: DimRef) -> int:
        """A dim by position, or by size name/expression ("s", "h*hd")."""
        if isinstance(dim, int):
            if not -self.ndim <= dim < self.ndim:
                raise PuzzleError(f"dim {dim} out of range for [{self}]")
            return dim % self.ndim
        size = parse_size(dim, allow_group_sizes=True)
        matches = [i for i, d in enumerate(self.dims) if same_size(d.size, size)]
        if len(matches) != 1:
            problem = "no" if not matches else "more than one"
            raise PuzzleError(f"{problem} dim {dim!r} in [{self}]; pass an index instead")
        return matches[0]

    def with_dims(self, dims: Iterable[Dim]) -> Layout:
        return dataclasses.replace(self, dims=tuple(dims))

    def with_dim(self, i: int, dim: Dim | None) -> Layout:
        """Replace dim i (or drop it, if dim is None)."""
        return self.with_dims(self.dims[:i] + ((dim,) if dim else ()) + self.dims[i + 1 :])

    def local_shape(self) -> tuple[int, ...]:
        return tuple(d.local_size() for d in self.dims)

    def local_numel_formula(self) -> sp.Expr:
        """Elements per rank, symbolically (used for byte formulas)."""
        return sp.simplify(sp.Mul(*(d.local_size_formula() for d in self.dims)))

    def expect_arguments(self) -> str:
        """The arguments of an .expect(...) that matches this layout: '"b s d", partial=tp'."""
        args = [f'"{format_dims(self.dims)}"']
        for name, groups in (("partial", self.partial), ("partial_max", self.partial_max)):
            if groups:
                names = sorted(groups)
                args.append(f"{name}={names[0] if len(names) == 1 else '[' + ', '.join(names) + ']'}")
        return ", ".join(args)

    def __str__(self) -> str:
        text = format_dims(self.dims)
        for name, groups in (("partial", self.partial), ("partial_max", self.partial_max)):
            if groups:
                text += f"  {name}({','.join(sorted(groups))})"
        return text
