"""The simulated cluster: mesh groups, ranks, and what happened during a run.

A run has one Simulation. It knows the concrete sizes of the current config (b=2, tp=4, ...),
which ranks exist, which ranks take part in a collective over a group, and it records a trace
of every result plus every collective with its byte count.
"""

from __future__ import annotations

import enum
import itertools
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sympy as sp

from .errors import Location, PuzzleError, caller_location

if TYPE_CHECKING:
    from .layout import Layout

GROUP_NAMES = ("tp", "dp", "ep", "cp")
BYTES_PER_ELEMENT = 2  # communication volumes assume bf16

Rank = tuple[int, ...]  # one coordinate per mesh group of the puzzle, in GROUP_NAMES order


class Group:
    """A mesh axis that collectives run over: tp, dp, ep or cp."""

    def __init__(self, name: str):
        self.name = name

    @property
    def size(self) -> int:
        return current().group_size(self.name)

    def __repr__(self) -> str:
        return self.name


tp, dp, ep, cp = (Group(name) for name in GROUP_NAMES)

GroupRef = Group | str


def group_name(group: GroupRef) -> str:
    if isinstance(group, Group):
        return group.name
    if isinstance(group, str) and group in GROUP_NAMES:
        return group
    raise PuzzleError(f"expected a mesh group (tp, dp, ep or cp), got {group!r}")


def size_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name, positive=True, integer=True)


class Phase(enum.Enum):
    SETUP = "setup"  # the checker building inputs: not traced
    INIT = "init"
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class CommRecord:
    op: str
    group: str
    formula: sp.Expr  # maximum bytes sent by one rank in this operation, symbolically
    nbytes: float  # the same, for the current config
    phase: Phase
    per_rank: dict[Rank, float]  # actual modeled send volume, including zero for boundary ranks


@dataclass
class TraceLine:
    """What one source line produced: its last result, and every collective it ran."""

    location: Location
    layout: Layout | None = None
    numel: int = 0  # elements per rank of that result
    comms: list[CommRecord] = field(default_factory=list)


class Simulation:
    def __init__(self, sizes: dict[str, int], dim_names: list[str], require_bytes: bool = False):
        self.sizes = dict(sizes)
        self.groups = [name for name in GROUP_NAMES if name in sizes]
        for group in self.groups:
            if not isinstance(sizes[group], int) or isinstance(sizes[group], bool) or sizes[group] < 1:
                raise PuzzleError(f"mesh group {group} must have a positive integer size, got {sizes[group]!r}")
        self.ranks: list[Rank] = list(itertools.product(*(range(sizes[g]) for g in self.groups)))
        self.dim_names = tuple(dim_names)
        self.require_bytes = require_bytes
        self.phase = Phase.SETUP
        self.trace: dict[Phase, dict[Location, TraceLine]] = {phase: {} for phase in Phase if phase != Phase.SETUP}
        self.comms: list[CommRecord] = []
        self.weight_matmuls = 0  # forward matmuls by a weight
        self.show_values = False  # whether show() prints (the checker turns it on for the first config)
        self.values: dict[sp.Expr, sp.Expr] = {}  # cache for evaluate

    # -- sizes

    def group_size(self, name: str) -> int:
        if name not in self.sizes:
            raise PuzzleError(f"group {name} isn't part of this puzzle's mesh ({', '.join(self.groups) or 'none'})")
        return self.sizes[name]

    def substitute(self, expr: sp.Expr) -> sp.Expr:
        if expr not in self.values:
            self.values[expr] = sp.nsimplify(expr.subs({size_symbol(k): v for k, v in self.sizes.items()}))
        return self.values[expr]

    def evaluate(self, expr: sp.Expr) -> int:
        """Value of a size expression in the current config; it must be a whole number."""
        value = self.substitute(expr)
        if not value.is_Integer:
            raise PuzzleError(f"size {str(expr).replace(' ', '')} = {value} isn't a whole number for this config")
        return int(value)

    def evaluate_float(self, expr: sp.Expr) -> float:
        return float(self.substitute(expr))

    # -- ranks

    def coordinate(self, rank: Rank, group: str) -> int:
        """This rank's index within `group` (0 .. group size - 1)."""
        return rank[self.groups.index(group)]

    def communicators(self, group: str) -> list[list[Rank]]:
        """The sets of ranks that take part in one collective over `group`, each ordered by coordinate.

        On a tp x dp mesh, a collective over tp runs separately inside every dp slice."""
        axis = self.groups.index(group)
        by_other_coordinates: dict[Rank, list[Rank]] = {}
        for rank in self.ranks:
            by_other_coordinates.setdefault(rank[:axis] + rank[axis + 1 :], []).append(rank)
        return [sorted(members, key=lambda r: r[axis]) for members in by_other_coordinates.values()]

    def bytes_sent(self, phase: Phase | None = None) -> float:
        """Maximum total bytes sent by any rank (or in one phase), for the current config.

        Sum each rank's traffic before taking the maximum across ranks.
        """
        return max(self.bytes_by_rank(phase).values(), default=0.0)

    def bytes_by_rank(self, phase: Phase | None = None) -> dict[Rank, float]:
        """Total modeled send volume for every rank, including boundary ranks that send nothing."""
        return {r: sum(c.per_rank[r] for c in self.comms if phase in (None, c.phase)) for r in self.ranks}

    # -- trace

    def trace_line(self) -> TraceLine | None:
        if self.phase == Phase.SETUP:
            return None
        location = caller_location() or ("?", 0)
        return self.trace[self.phase].setdefault(location, TraceLine(location))

    def record_tensor(self, layout: Layout, numel: int) -> None:
        line = self.trace_line()
        if line:
            line.layout, line.numel = layout, numel

    def record_comm(self, op: str, group: str, formula: sp.Expr, senders: set[Rank] | None = None) -> CommRecord:
        if self.phase == Phase.INIT:
            raise PuzzleError(f"{op} in on_init: setup should only shard weights, not communicate")
        formula = sp.simplify(formula) if senders is None or senders else sp.Integer(0)
        nbytes = self.evaluate_float(formula)
        per_rank = {r: nbytes if senders is None or r in senders else 0.0 for r in self.ranks}
        comm = CommRecord(op, group, formula, nbytes, self.phase, per_rank)
        self.comms.append(comm)
        line = self.trace_line()
        if line:
            line.comms.append(comm)
        return comm


_current: Simulation | None = None


def current() -> Simulation:
    if _current is None:
        raise PuzzleError("no puzzle is running; use check.py")
    return _current


@contextmanager
def running(simulation: Simulation) -> Iterator[Simulation]:
    """Make `simulation` the one that ops and collectives act on."""
    global _current
    previous, _current = _current, simulation
    try:
        yield simulation
    finally:
        _current = previous
