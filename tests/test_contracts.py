"""Correct values must not bypass an exercise's storage or communication contract."""

from types import SimpleNamespace

import pytest

from shardpuzzles import *
from shardpuzzles.catalog import discover
from shardpuzzles.errors import WrongResult
from shardpuzzles.runner import run
from solutions.p00b_partial_sums import RowParallelLinear
from solutions.p00c_data_parallel import Linear

PUZZLES = discover()


def test_replicated_weights_fail_the_storage_budget():
    class Replicated(RowParallelLinear):
        def on_init(self, p):
            self.W = p.W

    puzzle = PUZZLES["00b"]
    for config in puzzle.spec.configs:
        with pytest.raises(WrongResult, match="stored weights.*storage budget"):
            run(puzzle.spec, Replicated, config)


@pytest.mark.parametrize(
    "wrap", [lambda p: p, lambda p: [p], lambda p: {"originals": p}, lambda p: SimpleNamespace(p=p)]
)
def test_hidden_full_weights_are_counted(wrap):
    class Hidden(RowParallelLinear):
        def on_init(self, p):
            super().on_init(p)
            self.hidden = wrap(p)
            self.cycle = self

    puzzle = PUZZLES["00b"]
    with pytest.raises(WrongResult, match="stored weights.*storage budget"):
        run(puzzle.spec, Hidden, puzzle.spec.configs[0])


def test_aliases_do_not_count_twice():
    class Aliased(RowParallelLinear):
        def on_init(self, p):
            super().on_init(p)
            self.alias = self.W

    puzzle = PUZZLES["00b"]
    run(puzzle.spec, Aliased, puzzle.spec.configs[0])


def test_raw_shards_cannot_hide_weight_storage():
    class Raw(RowParallelLinear):
        def on_init(self, p):
            super().on_init(p)
            self.raw = p.W.shards

    puzzle = PUZZLES["00b"]
    with pytest.raises(WrongResult, match="raw torch tensor"):
        run(puzzle.spec, Raw, puzzle.spec.configs[0])


def test_cached_full_weights_are_checked_after_forward():
    class Cached(RowParallelLinear):
        def forward(self, x):
            y = super().forward(x)
            self.cached = all_gather(self.W, "d", tp).expect("d f", bytes="2*d*f*(tp-1)/tp")
            return y

    puzzle = PUZZLES["00b"]
    with pytest.raises(WrongResult, match="after forward, stored weights"):
        run(puzzle.spec, Cached, puzzle.spec.configs[0])


def test_weight_copies_are_checked_after_backward():
    class Cached(Linear):
        def backward(self, dy):
            grads = super().backward(dy)
            self.cached = self.W * 1
            return grads

    puzzle = PUZZLES["00c"]
    with pytest.raises(WrongResult, match="after backward, stored weights"):
        run(puzzle.spec, Cached, puzzle.spec.configs[0])


def test_redundant_collectives_fail_even_with_correct_values():
    class ExtraCollective(RowParallelLinear):
        def forward(self, x):
            y = super().forward(x)
            all_gather(self.W, "d", tp).expect("d f", bytes="2*d*f*(tp-1)/tp")
            return y

    puzzle = PUZZLES["00b"]
    with pytest.raises(WrongResult, match="forward uses 2 payload collectives; at most 1"):
        run(puzzle.spec, ExtraCollective, puzzle.spec.configs[0])


def test_an_alternative_resharding_algorithm_is_accepted():
    class Rescale(Module):
        def on_init(self, p):
            self.g = p.g

        def forward(self, x):
            return all_to_all(x * self.g, "s", "d", tp).expect("b s d:tp", bytes="2*b*s/tp*d*(tp-1)/tp")

    puzzle = PUZZLES["00a"]
    for config in puzzle.spec.configs:
        run(puzzle.spec, Rescale, config)
