"""Find numbered puzzle modules and their reference solutions on disk."""

from __future__ import annotations

import importlib
import inspect
import os
import re
import sys
from dataclasses import dataclass

from .definition import Module, Spec
from .errors import PACKAGE_DIR

ROOT = os.path.dirname(PACKAGE_DIR)


@dataclass(frozen=True)
class Puzzle:
    id: str  # "00a"
    name: str  # "p00a_layouts", the module name in puzzles/ and solutions/

    @property
    def path(self) -> str:
        return os.path.join(ROOT, "puzzles", f"{self.name}.py")

    @property
    def title(self) -> str:
        return (importlib.import_module(f"puzzles.{self.name}").__doc__ or self.name).strip().splitlines()[0]

    @property
    def spec(self) -> Spec:
        return importlib.import_module(f"puzzles.{self.name}").SPEC

    @property
    def hints(self) -> list[str]:
        return getattr(importlib.import_module(f"solutions.{self.name}"), "HINTS", [])

    def module_class(self, solution: bool = False) -> type[Module]:
        """The Module subclass defined in the puzzle file (or in its reference solution)."""
        module = importlib.import_module(f"{'solutions' if solution else 'puzzles'}.{self.name}")
        classes = [
            value
            for value in vars(module).values()
            if inspect.isclass(value) and issubclass(value, Module) and value.__module__ == module.__name__
        ]
        if len(classes) != 1:
            raise RuntimeError(f"{module.__name__} should define exactly one Module subclass, found {len(classes)}")
        return classes[0]


def discover() -> dict[str, Puzzle]:
    """Puzzle id -> Puzzle, ordered by section and letter."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    found = {}
    for filename in os.listdir(os.path.join(ROOT, "puzzles")):
        match = re.fullmatch(r"p(\d\d[a-z])_\w+\.py", filename)
        if match:
            found[match.group(1)] = Puzzle(match.group(1), filename.removesuffix(".py"))
    return dict(sorted(found.items()))


def select(puzzles: dict[str, Puzzle], arg: str | None) -> list[Puzzle]:
    """'0' -> section 00, '0b' -> just 00b, None -> every available puzzle."""
    if arg is None:
        return list(puzzles.values())
    arg = arg.strip().lower()
    match = re.fullmatch(r"(\d+)([a-z]?)", arg)
    if not match:
        raise SystemExit(f"unknown puzzle {arg!r}; use e.g. 0, 0a, 0b or 0c")
    prefix = f"{int(match.group(1)):02d}{match.group(2)}"
    chosen = [puzzle for puzzle_id, puzzle in puzzles.items() if puzzle_id.startswith(prefix)]
    if not chosen:
        raise SystemExit(f"no puzzle {arg!r}")
    return chosen
