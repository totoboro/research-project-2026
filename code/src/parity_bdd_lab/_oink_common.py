"""Shared data structures for the explicit Oink solver ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .game import ParityGame

Vertex = Hashable
Edge = tuple[Vertex, Vertex]


@dataclass(frozen=True)
class OinkResult:
    """Winning regions and strategy edges returned by an Oink solver port."""

    even_wins: frozenset[Vertex]
    odd_wins: frozenset[Vertex]
    even_strategy: frozenset[Edge]
    odd_strategy: frozenset[Edge]
    iterations: int


class OinkGame:
    """Dense, priority-ordered view of a :class:`ParityGame`.

    Oink orders vertices by priority before invoking these solvers.  This view
    reproduces that contract without imposing restrictions on Python labels.
    """

    def __init__(self, game: ParityGame) -> None:
        original_order = {vertex: index for index, vertex in enumerate(game.vertices)}
        self.labels = tuple(
            sorted(game.vertices, key=lambda vertex: (game.priority[vertex], original_order[vertex]))
        )
        self.ids = {vertex: index for index, vertex in enumerate(self.labels)}
        self.owner = tuple(game.owner[vertex] for vertex in self.labels)
        self.priority = tuple(game.priority[vertex] for vertex in self.labels)
        self.parity = tuple(priority & 1 for priority in self.priority)
        self.outs = tuple(
            tuple(self.ids[target] for target in game.successors[vertex])
            for vertex in self.labels
        )
        incoming: list[list[int]] = [[] for _ in self.labels]
        for source, targets in enumerate(self.outs):
            for target in targets:
                incoming[target].append(source)
        self.ins = tuple(tuple(sources) for sources in incoming)

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def max_priority(self) -> int:
        return self.priority[-1]

    def result(
        self,
        distraction: list[bool],
        strategies: list[set[int]],
        iterations: int,
    ) -> OinkResult:
        winners = tuple(self.parity[v] ^ distraction[v] for v in range(self.n))
        regions = {
            player: frozenset(self.labels[v] for v in range(self.n) if winners[v] == player)
            for player in (0, 1)
        }
        edges: dict[int, frozenset[Edge]] = {}
        for player in (0, 1):
            edges[player] = frozenset(
                (self.labels[v], self.labels[target])
                for v in range(self.n)
                if winners[v] == player and self.owner[v] == player
                for target in strategies[v]
            )
        return OinkResult(
            even_wins=regions[0],
            odd_wins=regions[1],
            even_strategy=edges[0],
            odd_strategy=edges[1],
            iterations=iterations,
        )


def check_iteration_limit(max_iterations: int) -> None:
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")


def priority_blocks(game: OinkGame) -> tuple[tuple[int, ...], ...]:
    """Return dense vertex indices grouped by exact priority."""
    blocks: list[list[int]] = [[] for _ in range(game.max_priority + 1)]
    for vertex, priority in enumerate(game.priority):
        blocks[priority].append(vertex)
    return tuple(tuple(block) for block in blocks)


def parity_block_ends(game: OinkGame) -> tuple[int, ...]:
    """Return Oink's end index for each consecutive equal-parity block."""
    ends = [game.n] * game.n
    next_vertex = game.n
    for vertex in range(game.n - 1, -1, -1):
        if next_vertex == game.n:
            ends[vertex] = game.n
        elif game.parity[next_vertex] != game.parity[vertex]:
            ends[vertex] = next_vertex
        else:
            ends[vertex] = ends[next_vertex]
        next_vertex = vertex
    return tuple(ends)


def current_winners(game: OinkGame, distraction: list[bool]) -> tuple[int, ...]:
    return tuple(game.parity[v] ^ distraction[v] for v in range(game.n))
