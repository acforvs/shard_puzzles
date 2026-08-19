"""uv run check.py [puzzle] [--solution] [--hint K] [--scaffold] [-q]"""

from __future__ import annotations

import argparse

from . import report
from .catalog import Puzzle, discover, select
from .errors import PuzzleError, Unfinished
from .runner import Outcome, execute, new_simulation, run
from .simulation import Simulation


def check(puzzle: Puzzle, solution: bool, verbose: bool) -> Outcome:
    """Run every config of one puzzle, printing as it goes."""
    report.title(puzzle)
    spec, cls = puzzle.spec, puzzle.module_class(solution)
    first: Simulation | None = None
    for seed, config in enumerate(spec.configs):
        sim = new_simulation(spec, config)
        sim.show_values = seed == 0
        try:
            execute(spec, cls, sim, seed)
        except Unfinished:
            report.unfinished(puzzle.path)
            return Outcome.UNFINISHED
        except NotImplementedError:
            report.stopped(sim, verbose)
            return Outcome.UNFINISHED
        except PuzzleError as error:
            report.failure(error, config, sim, verbose)
            return Outcome.FAILED
        except Exception:
            report.python_error(config)
            return Outcome.FAILED
        report.config_passed(config)
        first = first or sim

    assert first is not None
    report.summary(first, None if solution else reference_run(puzzle))
    if verbose:
        print(report.styled("2", "  trace:"))
        report.trace(first)
    return Outcome.PASSED


def reference_run(puzzle: Puzzle) -> Simulation | None:
    """The reference solution on the first config, to compare communication volumes."""
    try:
        return run(puzzle.spec, puzzle.module_class(solution=True), puzzle.spec.configs[0])
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check shard-puzzles.")
    parser.add_argument(
        "puzzle", nargs="?", help="section or puzzle ID, e.g. 0, 0b or 1a; default: all available puzzles"
    )
    parser.add_argument("--solution", action="store_true", help="check the reference solution instead of your code")
    parser.add_argument("--hint", nargs="?", type=int, const=1, metavar="K", help="show hint K (default 1)")
    parser.add_argument("--scaffold", action="store_true", help="print a fill-in-the-blanks skeleton with checkpoints")
    parser.add_argument("-q", "--quiet", action="store_true", help="don't print the shape/comm trace")
    args = parser.parse_args(argv)
    puzzles = select(discover(), args.puzzle)

    if args.hint is not None or args.scaffold:
        from .scaffold import scaffold

        for puzzle in puzzles:
            print(report.styled("1", f"{report.RULE} puzzle {puzzle.id} {report.RULE}"))
            if args.hint is not None:
                report.hint(puzzle, args.hint)
            if args.scaffold:
                print(scaffold(puzzle))
        return 0

    verbose = not args.quiet and len(puzzles) <= 3
    outcomes = {puzzle.id: check(puzzle, args.solution, verbose) for puzzle in puzzles}
    if len(puzzles) > 3:
        report.overview(outcomes)
    return 0 if all(outcome == Outcome.PASSED for outcome in outcomes.values()) else 1
