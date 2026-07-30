"""The learner-facing checker must not reveal byte answers on failed or incomplete runs."""

import pytest

from shardpuzzles import *
from shardpuzzles.catalog import Puzzle, discover
from shardpuzzles.cli import check, main
from shardpuzzles.runner import Outcome

CORRECT_FORMULA = "4*b*f*s*(tp-1)/tp"


@pytest.fixture
def attempt(monkeypatch):
    puzzle = discover()["00b"]
    original_module_class = Puzzle.module_class

    def install(fault):
        claimed_bytes = "2*b*s*f*(tp-1)/tp" if fault == "wrong" else "4*b*s*f*(tp-1)/tp"

        class Attempt(Module):
            def on_init(self, p):
                self.W = shard(p.W, "d:tp f")

            def forward(self, x):
                y = (x @ self.W).expect("b s f", partial=tp)
                result = all_reduce(y, tp)
                if fault == "unchecked":
                    return result
                if fault == "stopped":
                    raise NotImplementedError
                if fault == "missing":
                    return result.expect("b s f")
                if fault == "malformed":
                    return result.expect("b s f", bytes="unknown")
                if fault == "layout":
                    return result.expect("b s f:tp", bytes=claimed_bytes)
                return result.expect("b s f", bytes=claimed_bytes)

        def module_class(self, solution=False):
            if self.id == puzzle.id and not solution:
                return Attempt
            return original_module_class(self, solution)

        monkeypatch.setattr(Puzzle, "module_class", module_class)
        return puzzle

    return install


@pytest.mark.parametrize("fault", ["wrong", "missing", "malformed", "layout", "unchecked", "stopped"])
@pytest.mark.parametrize("verbose", [False, True])
def test_unsuccessful_checks_hide_computed_bytes(attempt, capsys, fault, verbose):
    puzzle = attempt(fault)
    expected = Outcome.UNFINISHED if fault == "stopped" else Outcome.FAILED
    assert check(puzzle, solution=False, verbose=verbose) == expected
    output = capsys.readouterr().out
    assert CORRECT_FORMULA not in output
    if verbose:
        assert "[all_reduce(tp)]" in output
        assert "[all_reduce(tp) " not in output
    if fault == "wrong":
        assert "incorrect byte count for all_reduce(tp)" in output
        assert "submitted: 2*b*f*s*(tp-1)/tp" in output
        assert "actual:" not in output


def test_successful_checks_show_computed_bytes(attempt, capsys):
    assert check(attempt(None), solution=False, verbose=True) == Outcome.PASSED
    assert f"[all_reduce(tp) {CORRECT_FORMULA} B]" in capsys.readouterr().out


def test_explicit_solution_shows_computed_bytes(attempt, capsys):
    attempt("wrong")
    assert main(["00b", "--solution"]) == 0
    assert f"[all_reduce(tp) {CORRECT_FORMULA} B]" in capsys.readouterr().out


def test_backward_byte_mismatch_hides_the_answer_in_the_cli(monkeypatch, capsys):
    original_module_class = Puzzle.module_class

    class Attempt(Module):
        def on_init(self, p):
            self.W = p.W

        def forward(self, x):
            self.x = x
            return einsum("b:dp s d, d f -> b:dp s f", x, self.W).expect("b:dp s f")

        def backward(self, dy):
            dx = einsum("b:dp s f, d f -> b:dp s d", dy, self.W).expect("b:dp s d")
            dW = einsum("b:dp s d, b:dp s f -> d f", self.x, dy).expect("d f", partial=dp)
            dW = all_reduce(dW, dp).expect("d f", bytes="2*d*f * (dp-1)/dp")
            return {"x": dx, "W": dW}

    def module_class(self, solution=False):
        if self.id == "00c" and not solution:
            return Attempt
        return original_module_class(self, solution)

    monkeypatch.setattr(Puzzle, "module_class", module_class)
    assert main(["0c"]) == 1
    output = capsys.readouterr().out
    assert "incorrect byte count for all_reduce(dp)" in output
    assert "submitted: 2*d*f*(dp-1)/dp" in output
    assert "backward up to the failure:" in output
    assert "[all_reduce(dp)]" in output
    assert "4*d*f*(dp-1)/dp" not in output
    assert "actual:" not in output
