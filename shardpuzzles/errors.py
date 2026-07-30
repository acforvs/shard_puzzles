"""Errors that point at the learner's line, not at the engine."""

from __future__ import annotations

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

Location = tuple[str, int]  # (file, line)


def caller_location() -> Location | None:
    """File and line of the first stack frame outside this package (the puzzle code)."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if not filename.startswith(PACKAGE_DIR):
            return filename, frame.f_lineno
        frame = frame.f_back
    return None


class PuzzleError(Exception):
    """A mistake in the puzzle code, explained in terms of shapes and sharding."""

    def __init__(self, message: str, location: Location | None = None):
        super().__init__(message)
        self.message = message
        self.location = location or caller_location()


class Unfinished(PuzzleError):
    """A TODO or ??? is still in the code: not a mistake, just not done yet."""


class WrongResult(PuzzleError):
    """The code ran, but what it returned has the wrong layout or the wrong numbers."""

    def __init__(self, message: str):
        super().__init__(message)
        self.location = None  # about the whole result, not one line
