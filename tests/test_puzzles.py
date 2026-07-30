"""Every reference solution passes, and every puzzle file is in sync with its solution."""

import ast
import os
import subprocess
import sys

import pytest

from shardpuzzles.catalog import ROOT, discover
from shardpuzzles.errors import Unfinished
from shardpuzzles.runner import run
from shardpuzzles.scaffold import scaffold

PUZZLES = discover()


@pytest.mark.parametrize("puzzle_id", list(PUZZLES))
def test_solution_passes_every_config(puzzle_id):
    puzzle = PUZZLES[puzzle_id]
    for seed, config in enumerate(puzzle.spec.configs):
        run(puzzle.spec, puzzle.module_class(solution=True), config, seed)


@pytest.mark.parametrize("puzzle_id", list(PUZZLES))
def test_stub_fails_only_on_its_blanks(puzzle_id):
    """An untouched puzzle file must stop at a TODO / ??? (never a NameError or another mistake)."""
    puzzle = PUZZLES[puzzle_id]
    with pytest.raises(Unfinished):
        run(puzzle.spec, puzzle.module_class(), puzzle.spec.configs[0])


@pytest.mark.parametrize("puzzle_id", list(PUZZLES))
def test_scaffold_and_hints(puzzle_id):
    puzzle = PUZZLES[puzzle_id]
    ast.parse(scaffold(puzzle))
    assert puzzle.hints


def test_puzzle_files_are_up_to_date():
    script = os.path.join(ROOT, "tools", "build_stubs.py")
    result = subprocess.run([sys.executable, script, "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_puzzle_sources_are_ascii():
    """Editors flag non-ASCII characters in code; keep every .py file plain ASCII."""
    for folder in ("puzzles", "solutions", "shardpuzzles", "tools", "tests"):
        for name in os.listdir(os.path.join(ROOT, folder)):
            if name.endswith(".py"):
                text = open(os.path.join(ROOT, folder, name), encoding="utf-8").read()
                assert text.isascii(), f"{folder}/{name}: {sorted({c for c in text if not c.isascii()})}"


def test_puzzle_and_solution_files_match():
    """Do not ship orphan solutions or undiscoverable puzzle files."""
    names = {
        folder: {
            name for name in os.listdir(os.path.join(ROOT, folder)) if name.endswith(".py") and name != "__init__.py"
        }
        for folder in ("puzzles", "solutions")
    }
    assert names["puzzles"] == names["solutions"]
    assert names["puzzles"] == {f"{puzzle.name}.py" for puzzle in PUZZLES.values()}
