"""Python port of Oink's multi-strategy FPI solver.

Ported from ``src/solvers/fpim.cpp`` in trolando/oink (Apache-2.0),
copyright 2017-2024 Tom van Dijk.
"""

from __future__ import annotations

from ._oink_common import OinkResult
from .game import ParityGame
from .oink_fpi import _solve_fpi


def oink_fpim(game: ParityGame, *, max_iterations: int = 1_000_000) -> OinkResult:
    """Solve ``game`` with Oink's multi-strategy FPI algorithm."""
    return _solve_fpi(game, multi_strategy=True, max_iterations=max_iterations)
