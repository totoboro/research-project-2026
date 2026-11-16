"""Cached maximisation with direct vertex relations instead of chain BDDs."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Hashable, Iterable

from .cq_cached_multistrategy import IncrementalForcedWinCache
from .decision_diagrams import new_bdd
from .game import ParityGame
from .multistrategy import (
    CycleKernel,
    _cycle_kernels,
    _justify_from_cycles,
    _solve_explicit_parity_game,
)

Vertex = Hashable
Edge = tuple[Vertex, Vertex]


@dataclass(frozen=True)
class QCachedMultiStrategyResult:
    """Forced-winning strategies and direct vertex-transition BDDs."""

    even_wins: frozenset[Vertex]
    odd_wins: frozenset[Vertex]
    even_strategy: frozenset[Edge]
    odd_strategy: frozenset[Edge]
    even_cycle_kernels: tuple[CycleKernel, ...]
    odd_cycle_kernels: tuple[CycleKernel, ...]
    justification: frozenset[Edge]
    # possible_bdd: object
    # possible_relation: object
    # possible_vertex_ids: tuple[Vertex, ...]
    # even_bdd: object
    # even_relation: object
    # even_vertex_ids: tuple[Vertex, ...]
    # odd_bdd: object
    # odd_relation: object
    # odd_vertex_ids: tuple[Vertex, ...]
    iterations: int
    # bdd_nodes: int


def _ordered_vertices(vertices: Iterable[Vertex]) -> tuple[Vertex, ...]:
    return tuple(
        sorted(vertices, key=lambda value: (type(value).__name__, repr(value)))
    )


def _interleaved_code(source: int, target: int, width: int) -> int:
    """Encode source/target bits in the BDD manager's variable order."""
    code = 0
    for bit in reversed(range(width)):
        code = (code << 1) | ((source >> bit) & 1)
        code = (code << 1) | ((target >> bit) & 1)
    return code


def _vertex_transition_bdd(
    vertices: tuple[Vertex, ...], edges: Iterable[Edge]
):
    """Build a vertex-transition relation directly from interleaved pair codes.

    The recursive builder consumes one sorted code range.  It creates the
    reduced BDD without first constructing source/target set BDDs and without
    invoking a general BDD intersection or union for every graph vertex.
    """
    count = len(vertices)
    width = max(1, (max(2, count) - 1).bit_length())
    current = tuple(f"v{i}" for i in range(width))
    following = tuple(f"v{i}_next" for i in range(width))
    variables = tuple(
        variable
        for pair in zip(reversed(current), reversed(following))
        for variable in pair
    )
    bdd = new_bdd(variables)
    variable_nodes = tuple(bdd.var(variable) for variable in variables)
    ids = {vertex: identifier for identifier, vertex in enumerate(vertices)}
    codes = tuple(
        sorted(
            {
                _interleaved_code(ids[source], ids[target], width)
                for source, target in edges
            }
        )
    )
    total_bits = 2 * width

    def build(position: int, begin: int, end: int):
        if begin == end:
            return bdd.false
        if position < 0:
            return bdd.true
        # Recursion fixes bits from most to least significant.  Within one
        # prefix, a complete suffix represents the constant true function.
        if end - begin == 1 << (position + 1):
            return bdd.true

        prefix = (codes[begin] >> (position + 1)) << (position + 1)
        split = bisect_left(
            codes, prefix | (1 << position), begin, end
        )
        low = build(position - 1, begin, split)
        high = build(position - 1, split, end)
        if low == high:
            return low
        level = total_bits - position - 1
        return bdd.ite(variable_nodes[level], high, low)

    relation = build(total_bits - 1, 0, len(codes))
    # ``build`` is recursively self-referential and closes over the temporary
    # variable functions.  Break that cycle before returning the manager so
    # dd.autoref can release those references deterministically.
    variable_nodes = ()
    build = None
    return bdd, relation, vertices


def q_cached_multistrategy(game: ParityGame) -> QCachedMultiStrategyResult:
    """Solve directly on vertices, maximise with the cache, and build node BDDs."""
    solved = _solve_explicit_parity_game(game)
    regions = {0: solved.even_wins, 1: solved.odd_wins}
    selected = {0: solved.even_choices, 1: solved.odd_choices}
    strategies: dict[int, frozenset[Edge]] = {}
    kernels: dict[int, tuple[CycleKernel, ...]] = {}
    justification_edges: set[Edge] = set()

    for player in (0, 1):
        region = regions[player]
        initial, player_justification, _ = _justify_from_cycles(
            game, player, region, selected
        )
        allowed = IncrementalForcedWinCache(
            game, player, region, initial
        ).maximise()
        strategies[player] = allowed
        kernels[player] = _cycle_kernels(game, player, region, allowed)
        justification_edges.update(player_justification)
        justification_edges.update(allowed)

    vertices = _ordered_vertices(game.vertices)
    possible_edges = tuple(
        (source, target)
        for source in vertices
        for target in game.successors[source]
    )
    strategy_edges = {}
    for player in (0, 1):
        region = regions[player]
        strategy_edges[player] = tuple(
            (source, target)
            for source in region
            for target in game.successors[source]
            if target in region
            and (
                game.owner[source] != player
                or (source, target) in strategies[player]
            )
        )

    # possible_bdd, possible_relation, possible_ids = _vertex_transition_bdd(
    #     vertices, possible_edges
    # )
    # even_bdd, even_relation, even_ids = _vertex_transition_bdd(
    #     vertices, strategy_edges[0]
    # )
    # odd_bdd, odd_relation, odd_ids = _vertex_transition_bdd(
    #     vertices, strategy_edges[1]
    # )
    return QCachedMultiStrategyResult(
        even_wins=regions[0],
        odd_wins=regions[1],
        even_strategy=strategies[0],
        odd_strategy=strategies[1],
        even_cycle_kernels=kernels[0],
        odd_cycle_kernels=kernels[1],
        justification=frozenset(justification_edges),
        # possible_bdd=possible_bdd,
        # possible_relation=possible_relation,
        # possible_vertex_ids=possible_ids,
        # even_bdd=even_bdd,
        # even_relation=even_relation,
        # even_vertex_ids=even_ids,
        # odd_bdd=odd_bdd,
        # odd_relation=odd_relation,
        # odd_vertex_ids=odd_ids,
        iterations=solved.iterations,
        # bdd_nodes=len(possible_bdd) + len(even_bdd) + len(odd_bdd),
    )
