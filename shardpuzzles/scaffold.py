"""Generate puzzle stubs and optional scaffolds from reference solution code.

A stub replaces each TODO region with named blanks. A scaffold also shows the layout
for each blank, while keeping byte answers hidden.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable

from .catalog import Puzzle
from .runner import run

TODO_START, TODO_END = "# TODO:", "# END TODO"


def solution_source(puzzle: Puzzle) -> tuple[str, ast.ClassDef]:
    source = textwrap.dedent(inspect.getsource(puzzle.module_class(solution=True)))
    cls = ast.parse(source).body[0]
    assert isinstance(cls, ast.ClassDef)
    return source, cls


def segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ""


def assigned_names(statement: ast.stmt, source: str) -> str | None:
    if isinstance(statement, ast.Assign):
        return " = ".join(segment(source, t) for t in statement.targets)
    if isinstance(statement, (ast.AugAssign, ast.AnnAssign)):
        return segment(source, statement.target)
    return None


def layouts_by_line(puzzle: Puzzle) -> dict[int, str]:
    """Run the solution once; for each line of its class, the .expect(...) arguments that describe
    the last result produced there (byte answers hidden)."""
    spec, cls = puzzle.spec, puzzle.module_class(solution=True)
    path = inspect.getsourcefile(cls)
    first_line = inspect.getsourcelines(cls)[1]
    sim = run(spec, cls, spec.configs[0])
    layouts = {}
    for lines in sim.trace.values():
        for line in lines.values():
            if line.location[0] == path and line.layout is not None:
                arguments = line.layout.expect_arguments()
                if line.comms and spec.require_bytes:
                    arguments += ', bytes="???"'
                layouts[line.location[1] - first_line + 1] = arguments
    return layouts


# ---------------------------------------------------------------------------
# Generators


def scaffold(puzzle: Puzzle) -> str:
    """The stub, but each blank states the layout it must produce (byte answers hidden)."""
    layouts = layouts_by_line(puzzle)

    def blank(statement: ast.stmt, source: str) -> str | None:
        target = assigned_names(statement, source)
        if target is None:
            return "return TODO" if isinstance(statement, ast.Return) else None
        lines = range(statement.lineno, (statement.end_lineno or statement.lineno) + 1)
        layout = next((layouts[n] for n in reversed(lines) if n in layouts), None)
        single = isinstance(statement, ast.Assign) and not isinstance(statement.targets[0], ast.Tuple)
        return f"{target} = TODO.expect({layout})" if layout and single else f"{target} = TODO"

    header = "# The puzzle's stub, with the layout each blank must produce. Fill in the TODOs.\n"
    return header + fill_todo_regions(puzzle, blank)


def stub(puzzle: Puzzle) -> str:
    """The solution with each `# TODO: prompt` ... `# END TODO` region replaced by `name = TODO` lines."""

    def blank(statement: ast.stmt, source: str) -> str | None:
        target = assigned_names(statement, source)
        return f"{target} = TODO" if target else "return TODO" if isinstance(statement, ast.Return) else None

    return fill_todo_regions(puzzle, blank)


def fill_todo_regions(puzzle: Puzzle, blank: Callable[[ast.stmt, str], str | None]) -> str:
    """Copy the solution class, replacing each statement inside a TODO region with blank(statement)."""
    source, cls = solution_source(puzzle)
    starts = {n.lineno: n for n in ast.walk(cls) if isinstance(n, ast.stmt)}
    out, inside, indent, seen = [], False, "", set()
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(TODO_START):
            out.append(line)
            inside, indent, seen = True, line[: len(line) - len(line.lstrip())], set()
        elif stripped == TODO_END:
            inside = False
        elif not inside:
            out.append(line)
        elif number in starts and (text := blank(starts[number], source)) and text not in seen:
            seen.add(text)  # one blank per name, even if the solution assigns it twice
            out.append(indent + text)
    return "\n".join(out)
