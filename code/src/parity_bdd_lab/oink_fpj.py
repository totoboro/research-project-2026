"""Python port of Oink's explicit Fixpoint Iteration with Justifications.

Ported from ``src/solvers/fpj.cpp`` in trolando/oink (Apache-2.0),
copyright 2020 Tom van Dijk.
"""

from __future__ import annotations

from collections import deque

from ._oink_common import (
    OinkGame,
    OinkResult,
    check_iteration_limit,
    current_winners,
    parity_block_ends,
)
from .game import ParityGame


def oink_fpj(game: ParityGame, *, max_iterations: int = 1_000_000) -> OinkResult:
    """Solve ``game`` with Oink's non-greedy sequential FPJ algorithm."""
    check_iteration_limit(max_iterations)
    oink = OinkGame(game)
    block_end = parity_block_ends(oink)
    justified = [False] * oink.n
    distraction = [False] * oink.n
    strategies: list[set[int]] = [set() for _ in range(oink.n)]
    queue: deque[int] = deque()
    iterations = 1
    index = 0

    while index < oink.n:
        if iterations > max_iterations:
            raise RuntimeError(
                f"Oink FPJ did not converge within {max_iterations:,} iterations"
            )
        end = block_end[index]
        block_start = index
        block_parity = oink.parity[index]
        block_changed = False
        winners = current_winners(oink, distraction)

        for vertex in range(index, end):
            if justified[vertex]:
                continue
            owner = oink.owner[vertex]
            target = next(
                (target for target in oink.outs[vertex] if winners[target] == owner),
                None,
            )
            strategies[vertex] = set() if target is None else {target}
            one_step_winner = 1 - owner if target is None else owner
            if block_parity != one_step_winner:
                queue.append(vertex)
                distraction[vertex] = True
                justified[vertex] = True
                block_changed = True

        index = end
        if block_changed:
            winners = current_winners(oink, distraction)
            while queue:
                changed_vertex = queue.popleft()
                for predecessor in oink.ins[changed_vertex]:
                    # Oink preserves justifications for the new winner of the batch.
                    if winners[predecessor] == 1 - block_parity:
                        continue
                    if justified[predecessor] and (
                        not strategies[predecessor]
                        or changed_vertex in strategies[predecessor]
                    ):
                        justified[predecessor] = False
                        distraction[predecessor] = False
                        queue.append(predecessor)
                        index = min(index, predecessor)
            iterations += 1
            if index > block_start:
                index = block_start
        else:
            for vertex in range(block_start, end):
                justified[vertex] = True

    return oink.result(distraction, strategies, iterations)
