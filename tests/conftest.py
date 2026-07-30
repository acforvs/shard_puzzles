import os
import sys
from contextlib import ExitStack

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shardpuzzles import simulation  # noqa: E402
from shardpuzzles.layout import parse_dims  # noqa: E402
from shardpuzzles.simulation import GROUP_NAMES, Phase, Simulation  # noqa: E402
from shardpuzzles.tensor import ShardedTensor, from_global  # noqa: E402


class Mesh:
    """mesh(b=2, s=4, d=6, tp=2) starts a simulation; mesh.tensor(spec) makes a random tensor laid out as spec."""

    def __init__(self, stack: ExitStack):
        self.stack = stack

    def __call__(self, **sizes: int) -> "Mesh":
        self.sim = Simulation(sizes, [k for k in sizes if k not in GROUP_NAMES])
        self.sim.phase = Phase.FORWARD
        self.stack.enter_context(simulation.running(self.sim))
        return self

    def tensor(self, spec: str, is_weight: bool = False) -> ShardedTensor:
        shape = [self.sim.evaluate(d.size) for d in parse_dims(spec)]
        return from_global(torch.randn(shape, dtype=torch.float64), spec, is_weight=is_weight)


@pytest.fixture
def mesh():
    with ExitStack() as stack:
        yield Mesh(stack)
