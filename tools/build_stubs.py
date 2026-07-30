"""Regenerate the "your code" section of every puzzle file from its reference solution.

    uv run python tools/build_stubs.py          # rewrite puzzle files
    uv run python tools/build_stubs.py --check  # fail if any puzzle file is out of date (used by the tests)

Each `# TODO: ...` region in the solution becomes `name = TODO` lines.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shardpuzzles.catalog import Puzzle, discover  # noqa: E402
from shardpuzzles.scaffold import stub  # noqa: E402

YOUR_CODE = "# ---- your code "
DEFINITION = "# ---- puzzle definition "


def rebuilt(puzzle: Puzzle) -> str:
    """The puzzle file with its "your code" section regenerated from the solution."""
    text = open(puzzle.path).read()
    start, end = text.index(YOUR_CODE), text.index(DEFINITION)
    header_end = text.index("\n", start) + 1
    code = stub(puzzle)
    return text[:header_end] + code.rstrip() + "\n\n\n" + text[end:]


def main() -> int:
    check_only = "--check" in sys.argv
    puzzles = discover().values()
    stale = []
    for puzzle in puzzles:
        new = rebuilt(puzzle)
        if new != open(puzzle.path).read():
            stale.append(puzzle.name)
            if not check_only:
                with open(puzzle.path, "w") as f:
                    f.write(new)
    if check_only and stale:
        print("out of date (run tools/build_stubs.py):", ", ".join(stale))
        return 1
    print(
        f"{'checked' if check_only else 'rebuilt'} {len(puzzles)} puzzles"
        + (f"; updated {len(stale)}" if stale else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
