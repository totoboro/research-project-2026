"""Python port of Oink's explicit Distraction Fixpoint Iteration solver.

Ported from ``src/solvers/fpi.cpp`` in trolando/oink (Apache-2.0),
copyright 2017-2018 Tom van Dijk, Johannes Kepler University Linz.
"""

from __future__ import annotations

from ._oink_common import (
    OinkGame,
    OinkResult,
    check_iteration_limit,
    current_winners,
    priority_blocks,
)
from .game import ParityGame


def _solve_fpi(
    game: ParityGame, *, multi_strategy: bool, max_iterations: int
) -> OinkResult:
    check_iteration_limit(max_iterations)
    oink = OinkGame(game)
    blocks = priority_blocks(oink)
    distraction = [False] * oink.n
    frozen = [0] * oink.n
    candidate = [True] * oink.n
    strategies: list[set[int]] = [set() for _ in range(oink.n)]
    iterations = 1
    priority = 0

    while priority <= oink.max_priority:
        if iterations > max_iterations:
            raise RuntimeError(
                f"Oink FPI did not converge within {max_iterations:,} iterations"
            )
        winners = current_winners(oink, distraction)
        changed = 0
        for vertex in blocks[priority]:
            if not candidate[vertex]:
                continue
            owner = oink.owner[vertex]
            winning_targets = {
                target for target in oink.outs[vertex] if winners[target] == owner
            }
            if multi_strategy:
                strategies[vertex] = winning_targets
            else:
                strategies[vertex] = (
                    {next(target for target in oink.outs[vertex] if target in winning_targets)}
                    if winning_targets
                    else set()
                )
            one_step_winner = owner if winning_targets else 1 - owner
            if oink.parity[vertex] != one_step_winner:
                distraction[vertex] = True
                candidate[vertex] = False
                changed += 1

        if changed == 0:
            priority += 1
            continue

        if priority != 0:
            player = priority & 1
            for vertex in range(blocks[priority][0]):
                if frozen[vertex] >= priority:
                    continue
                if frozen[vertex] != 0:
                    if (frozen[vertex] & 1) == player:
                        frozen[vertex] = priority
                    else:
                        frozen[vertex] = 0
                        distraction[vertex] = False
                        candidate[vertex] = True
                elif distraction[vertex]:
                    if oink.parity[vertex] == player:
                        frozen[vertex] = priority
                    else:
                        distraction[vertex] = False
                        candidate[vertex] = True
                elif oink.parity[vertex] != player:
                    frozen[vertex] = priority
                    candidate[vertex] = False
            priority = 0
        iterations += 1

    return oink.result(distraction, strategies, iterations)


def oink_fpi(game: ParityGame, *, max_iterations: int = 1_000_000) -> OinkResult:
    """Solve ``game`` with Oink's sequential FPI algorithm."""
    return _solve_fpi(game, multi_strategy=False, max_iterations=max_iterations)
