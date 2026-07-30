"""Printing results: pass/fail lines, the shape/comm trace, failures and hints."""

from __future__ import annotations

import linecache
import os
import sys
import tokenize
import traceback
from collections import Counter

from .catalog import ROOT, Puzzle
from .errors import PACKAGE_DIR, Location, PuzzleError
from .layout import format_size
from .runner import Outcome
from .simulation import CommRecord, Phase, Simulation

PASS, FAIL, TODO_MARK, RULE = "\u2713", "\u2717", "\u00b7", "\u2501\u2501"  # check, cross, middle dot, heavy line
YOUR_CODE = "# ---- your code"


def styled(code: str, text: str) -> str:
    """ANSI colour when printing to a terminal: 1 bold, 2 dim, 31 red, 32 green, 33 yellow, 36 cyan."""
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def source_line(location: Location | None) -> str:
    return linecache.getline(*location).strip() if location else ""


def short_path(location: Location | None) -> str:
    return f"{os.path.relpath(location[0], ROOT)}:{location[1]}" if location else "?"


def format_config(config: dict[str, int]) -> str:
    return " ".join(f"{k}={v}" for k, v in config.items())


def format_comms(comms: list[CommRecord], *, show_bytes: bool = True) -> str:
    if not show_bytes:
        return "; ".join(f"{c.op}({c.group})" for c in comms)
    return "; ".join(
        f"{c.op}({c.group}) {'max ' if len(set(c.per_rank.values())) > 1 else ''}{format_size(c.formula)} B"
        for c in comms
    )


# ---------------------------------------------------------------------------
# Lines


def title(puzzle: Puzzle) -> None:
    print(styled("1", f"{RULE} {puzzle.title} {RULE}"))


def config_passed(config: dict[str, int]) -> None:
    print(f"  {styled('32', PASS)} {format_config(config)}")


def summary(sim: Simulation, reference: Simulation | None) -> None:
    """The collectives of the first config, and how their volume compares with the reference solution."""
    counts = Counter(f"{c.op}({c.group})" for c in sim.comms)
    ops = [op if n == 1 else f"{op} x{n}" for op, n in sorted(counts.items())]
    volume = ""
    if reference is not None:
        mine, theirs = sim.bytes_sent(), reference.bytes_sent()
        if mine > theirs:
            volume = styled("33", f"  - {mine:g} B/rank vs {theirs:g} for the reference: can you move less?")
        elif mine < theirs:
            volume = styled("36", f"  - {mine:g} B/rank, less than the reference ({theirs:g})!")
        else:
            volume = styled("2", "  (same volume as the reference)")
    print(f"  {styled('32', PASS)} comms ({format_config(sim.sizes)}): {', '.join(ops) or 'none'}{volume}")
    totals = sim.bytes_by_rank()
    if len(set(totals.values())) > 1:
        low, high = min(totals.values()), max(totals.values())
        print(f"    send volume varies by rank: {low:g}..{high:g} B; maximum {high:g} B/rank")


def trace(
    sim: Simulation,
    phases: tuple[Phase, ...] = (Phase.INIT, Phase.FORWARD, Phase.BACKWARD),
    *,
    show_bytes: bool = True,
) -> None:
    """Trace layouts and collectives; hide computed byte answers on failed or incomplete runs."""
    computed = [line for phase in (Phase.FORWARD, Phase.BACKWARD) for line in sim.trace[phase].values()]
    largest = max(computed, key=lambda line: line.numel, default=None)
    for phase in phases:
        if not sim.trace[phase]:
            continue
        if len(phases) > 1:
            print(styled("2", f"  {phase.value}:"))
        for line in sim.trace[phase].values():
            code = source_line(line.location)
            code = code if len(code) <= 56 else code[:53] + "..."
            text = f"    L{line.location[1]:<4} {code:<56}  {line.layout or ''}"
            if line.comms:
                text += styled("36", f"   [{format_comms(line.comms, show_bytes=show_bytes)}]")
            if line is largest and line.layout is not None:
                elements = format_size(line.layout.local_numel_formula())
                text += styled("33", f"   <- largest: {elements} elems/rank")
            print(text)


# ---------------------------------------------------------------------------
# Not passing


def unfinished(path: str) -> None:
    line = first_blank(path)
    where = f": the first TODO / ??? is at\n    {short_path((path, line))}  {source_line((path, line))}" if line else ""
    print(styled("2", f"  {TODO_MARK} not finished{where}"))


def stopped(sim: Simulation, verbose: bool) -> None:
    """The learner raised NotImplementedError: show that everything before it checked out."""
    print(styled("2", f"  {TODO_MARK} stopped at NotImplementedError; everything before it checked out"))
    if verbose and sim.phase != Phase.SETUP and sim.trace[sim.phase]:
        trace(sim, phases=(sim.phase,), show_bytes=False)


def first_blank(path: str) -> int | None:
    """The line of the first TODO or "???" in the file's "your code" section."""
    with open(path) as f:
        start = next((n for n, text in enumerate(f, start=1) if text.startswith(YOUR_CODE)), 0)
    with open(path) as f:
        for token in tokenize.generate_tokens(f.readline):
            blank = (token.type == tokenize.NAME and token.string == "TODO") or (
                token.type == tokenize.STRING and "???" in token.string
            )
            if blank and token.start[0] > start:
                return token.start[0]
    return None


def failure(error: PuzzleError, config: dict[str, int], sim: Simulation, verbose: bool) -> None:
    print(f"  {styled('31', FAIL)} {format_config(config)}")
    if error.location:
        print(f"    {short_path(error.location)}  {source_line(error.location)}")
    print("    " + error.message.replace("\n", "\n    "))
    if verbose and sim.phase != Phase.SETUP and sim.trace[sim.phase]:
        print(styled("2", f"  {sim.phase.value} up to the failure:"))
        trace(sim, phases=(sim.phase,), show_bytes=False)


def python_error(config: dict[str, int]) -> None:
    """A plain Python exception: show the learner's frames only."""
    print(f"  {styled('31', FAIL)} {format_config(config)}: Python error")
    frames = [f for f in traceback.extract_tb(sys.exc_info()[2]) if not f.filename.startswith(PACKAGE_DIR)]
    print("    " + "".join(traceback.format_list(frames[-3:])).rstrip().replace("\n", "\n    "))
    print("    " + traceback.format_exc().strip().splitlines()[-1])


def hint(puzzle: Puzzle, number: int) -> None:
    hints = puzzle.hints
    if not hints:
        print(styled("2", "  no hints for this puzzle"))
        return
    number = max(1, min(number, len(hints)))
    print(f"  hint {number}/{len(hints)}: {hints[number - 1]}")
    if number < len(hints):
        print(styled("2", f"  (next: --hint {number + 1}; skeleton with checkpoints: --scaffold)"))


def overview(outcomes: dict[str, Outcome]) -> None:
    """One progress line with a status mark for each puzzle."""
    marks = {
        Outcome.PASSED: styled("32", PASS),
        Outcome.FAILED: styled("31", FAIL),
        Outcome.UNFINISHED: styled("2", TODO_MARK),
    }
    print("\n" + "  ".join(f"{puzzle_id}{marks[outcome]}" for puzzle_id, outcome in outcomes.items()))
