"""Explicit Zielonka parity-game solver used as a benchmark baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .game import ParityGame

Vertex = Hashable
Edge = tuple[Vertex, Vertex]


@dataclass(frozen=True)
class ZielonkaResult:
    """Winning regions and positional strategies from Zielonka's algorithm."""

    even_wins: frozenset[Vertex]
    odd_wins: frozenset[Vertex]
    even_strategy: frozenset[Edge]
    odd_strategy: frozenset[Edge]


def zielonka(game: ParityGame) -> ZielonkaResult:
    """Solve a finite max-parity game with the recursive Zielonka algorithm."""

    def attractor(vertices: frozenset[Vertex], player: int, target: frozenset[Vertex]):
        attracted = set(target)
        choices: dict[Vertex, Vertex] = {}
        while True:
            added = False
            for vertex in vertices:
                if vertex in attracted:
                    continue
                successors = tuple(w for w in game.successors[vertex] if w in vertices)
                if game.owner[vertex] == player:
                    winning_moves = tuple(w for w in successors if w in attracted)
                    if winning_moves:
                        attracted.add(vertex)
                        choices[vertex] = winning_moves[0]
                        added = True
                elif successors and all(w in attracted for w in successors):
                    attracted.add(vertex)
                    added = True
            if not added:
                return frozenset(attracted), choices

    def solve(vertices: frozenset[Vertex]):
        if not vertices:
            return frozenset(), frozenset(), {}, {}

        highest = max(game.priority[vertex] for vertex in vertices)
        alpha = highest % 2
        beta = 1 - alpha
        top = frozenset(v for v in vertices if game.priority[v] == highest)
        attract_alpha, strategy_alpha_attr = attractor(vertices, alpha, top)

        sub_even, sub_odd, sub_even_strategy, sub_odd_strategy = solve(vertices - attract_alpha)
        sub_regions = {0: sub_even, 1: sub_odd}
        sub_strategies = {0: sub_even_strategy, 1: sub_odd_strategy}

        if not sub_regions[beta]:
            wins = {0: set(sub_even), 1: set(sub_odd)}
            wins[alpha].update(attract_alpha)
            strategies = {0: dict(sub_even_strategy), 1: dict(sub_odd_strategy)}
            strategies[alpha].update(strategy_alpha_attr)
            return frozenset(wins[0]), frozenset(wins[1]), strategies[0], strategies[1]

        attract_beta, strategy_beta_attr = attractor(vertices, beta, sub_regions[beta])
        rest_even, rest_odd, rest_even_strategy, rest_odd_strategy = solve(vertices - attract_beta)
        rest_regions = {0: set(rest_even), 1: set(rest_odd)}
        rest_strategies = {0: dict(rest_even_strategy), 1: dict(rest_odd_strategy)}
        rest_regions[beta].update(attract_beta)

        # Keep the first recursive strategy on its beta-winning subregion and
        # the beta attractor choices on the rest of the newly added attractor.
        rest_strategies[beta].update(sub_strategies[beta])
        rest_strategies[beta].update(strategy_beta_attr)
        return (
            frozenset(rest_regions[0]),
            frozenset(rest_regions[1]),
            rest_strategies[0],
            rest_strategies[1],
        )

    even_wins, odd_wins, even_choices, odd_choices = solve(frozenset(game.vertices))

    # Attractor targets are seeded directly and therefore may not have acquired
    # a move in the attractor construction. Complete those target choices with
    # a successor inside the same winning region.
    for player, region, choices in (
        (0, even_wins, even_choices),
        (1, odd_wins, odd_choices),
    ):
        for vertex in region:
            if game.owner[vertex] == player and vertex not in choices:
                winning_successors = tuple(w for w in game.successors[vertex] if w in region)
                if not winning_successors:
                    raise RuntimeError(f"no winning successor found for player {player} at {vertex!r}")
                choices[vertex] = winning_successors[0]

    even_edges = frozenset(even_choices.items())
    odd_edges = frozenset(odd_choices.items())
    return ZielonkaResult(even_wins, odd_wins, even_edges, odd_edges)
