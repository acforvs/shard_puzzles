"""show(t): print what every simulated rank holds, to see a layout as numbers."""

from __future__ import annotations

import linecache

import torch

from .errors import caller_location
from .layout import format_size
from .simulation import Rank, Simulation, current
from .tensor import ShardedTensor, require_tensor

VALUES_SHOWN = 6


def show(t: ShardedTensor, name: str = "") -> ShardedTensor:
    """Print each rank's piece of `t`: its local shape, which global slice it holds, and its first values.

    A partial sum also prints the sum over its group: the real value. Prints for the first config only
    and returns `t`, so `y = show(y)` and `show(x @ W ...)` both work. Pending results must be .expect()ed first."""
    t = require_tensor(t, "show")
    sim = current()
    if not sim.show_values:
        return t
    location = caller_location()
    where = f"L{location[1]:<4} " if location else ""
    name = name or (linecache.getline(*location).strip() if location else "")
    config = " ".join(f"{k}={v}" for k, v in sim.sizes.items())
    print(f"    {where}{name}\n      [{t.spec}]   ({config})")
    for rank in sim.ranks:
        print(f"      {rank_name(sim, rank):<12}{held(t, sim, rank):<28}{values(t.shards[rank])}")
    first = sim.ranks[0]
    for groups, what in ((t.partial, "sum"), (t.partial_max, "max")):
        if groups:
            # the ranks that hold terms of rank 0's value: the same coordinate in every other group
            same = [i for i, g in enumerate(sim.groups) if g not in groups]
            stacked = torch.stack([t.shards[r] for r in sim.ranks if all(r[i] == first[i] for i in same)])
            total = stacked.sum(0) if what == "sum" else stacked.amax(0)
            label = f"{what} over {','.join(sorted(groups))}"
            print(f"      {label:<40}{values(total)}   <- the real value at {rank_name(sim, first)}")
    return t


def rank_name(sim: Simulation, rank: Rank) -> str:
    return " ".join(f"{g}={c}" for g, c in zip(sim.groups, rank)) or "rank 0"


def held(t: ShardedTensor, sim: Simulation, rank: Rank) -> str:
    """The local shape, plus the global range of every split dim, e.g. "[2, 2, 8] s[2:4]"."""
    parts = [str(list(t.local_shape))]
    for dim, n in zip(t.dims, t.local_shape):
        if dim.split_over:
            start = sim.coordinate(rank, dim.split_over) * n
            parts.append(f"{dim.text or format_size(dim.size)}[{start}:{start + n}]")
    return " ".join(parts)


def values(x: torch.Tensor) -> str:
    flat = x.flatten().tolist()
    shown = " ".join(f"{v:+.3f}" if isinstance(v, float) else str(v) for v in flat[:VALUES_SHOWN])
    return shown + (" ..." if len(flat) > VALUES_SHOWN else "")
