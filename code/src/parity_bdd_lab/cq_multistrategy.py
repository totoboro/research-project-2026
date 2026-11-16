"""CQ-MS: a maximised Chain-Quotient MultiStrategy solver.

CQ-MS starts from Chain-MS's conservative SCC justification, then revisits
every deferred in-region move.  A move is retained exactly when adding it to
the moves accepted so far keeps the entire multistrategy forced-winning.  This
greedy pass is deterministic and produces an inclusion-maximal (though not
necessarily maximum-cardinality) multistrategy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .game import ParityGame
from .multistrategy import (
    ExplicitSolverTraceEvent,
    ForcedMultiStrategyResult,
    _cycle_kernels,
    _chain_decomposition,
    _chain_transition_bdd,
    _compress_to_chain_game,
    _expand_solution,
    _justify_from_cycles,
    _solve_explicit_parity_game,
)
from .zielonka import zielonka


@dataclass(frozen=True)
class CQStrategySnapshot:
    """Winning regions and allowed edges at one CQ-MS construction stage."""

    even_wins: frozenset
    odd_wins: frozenset
    even_strategy: frozenset
    odd_strategy: frozenset


@dataclass(frozen=True)
class CQMultiStrategyStages:
    """The conservative CQ-MS seed and its greedily maximised result."""

    initial: CQStrategySnapshot
    maximized: ForcedMultiStrategyResult


def _ordered_edges(edges):
    try:
        return tuple(sorted(edges))
    except TypeError:
        return tuple(sorted(edges, key=lambda edge: (repr(edge[0]), repr(edge[1]))))


def _is_forced_winning(
    game: ParityGame,
    player: int,
    region: frozenset,
    allowed: frozenset,
) -> bool:
    """Check universal safety by giving every restricted choice to the opponent."""
    if not region:
        return True
    
    adversary = 1 - player
    restricted = ParityGame(
        owner={vertex: adversary for vertex in region},
        priority={vertex: game.priority[vertex] for vertex in region},
        successors={
            vertex: tuple(
                target
                for target in game.successors[vertex]
                if target in region
                and (game.owner[vertex] != player or (vertex, target) in allowed)
            )
            for vertex in region
        },
        name=f"{game.name} [CQ-MS player {player} safety check]",
    )

    solution = zielonka(restricted)
    winning_region = solution.even_wins if player == 0 else solution.odd_wins
    return winning_region == region


def _maximise_allowed_edges(
    game: ParityGame,
    player: int,
    region: frozenset,
    initial: frozenset,
) -> frozenset:
    """Greedily retain every deferred edge that preserves forced winning.

    Once an edge is unsafe it cannot become safe after enabling more choices,
    so one deterministic pass is sufficient for inclusion-maximality.
    """
    allowed = set(initial)
    candidates = _ordered_edges(
        (vertex, target)
        for vertex in region
        if game.owner[vertex] == player
        for target in game.successors[vertex]
        if target in region and (vertex, target) not in allowed
    )
    for edge in candidates:
        extended = frozenset(allowed | {edge})
        winning = _is_forced_winning(game, player, region, extended)
        if winning:
            allowed.add(edge)

    return frozenset(allowed)


def _solve_cq_quotient(
    game: ParityGame,
    *,
    trace: list[ExplicitSolverTraceEvent] | None = None,
):
    """Solve CQ-MS's quotient; this is the explicit-solver replacement seam."""
    return _solve_explicit_parity_game(game, trace=trace)


def chain_quotient_multistrategy_stages(game: ParityGame) -> CQMultiStrategyStages:
    """Return both CQ-MS's conservative seed and maximised move sets.

    The quotient and initial SCC justification mirror Chain-MS. Deferred moves
    are then reconsidered in stable edge order and accepted only after an exact
    universal forced-win check. Future explicit quotient solvers only need to
    replace :func:`_solve_cq_quotient`.
    """
    possible_chains, possible_nodes = _chain_decomposition(
        frozenset(game.vertices),
        frozenset((v, w) for v in game.vertices for w in game.successors[v]),
        game.priority,
    )
    compressed, chain_state_to_id = _compress_to_chain_game(game, possible_chains)
    solved = _solve_cq_quotient(compressed)
    regions, selected = _expand_solution(
        game, possible_chains, compressed, chain_state_to_id, solved
    )
    strategies = {}
    initial_strategies = {}
    chains = {}
    chain_nodes = {}
    kernels = {}
    justification_edges = set()
    chain_bdds = {}
    chain_relations = {}
    chain_ids = {}
    for player in (0, 1):
        region = regions[player]
        allowed, player_justification, _ = _justify_from_cycles(
            game, player, region, selected
        )
        initial_strategies[player] = allowed

        allowed = _maximise_allowed_edges(game, player, region, allowed)

        strategies[player] = allowed
        justification_edges.update(player_justification)
        justification_edges.update(allowed)
        kernels[player] = _cycle_kernels(game, player, region, allowed)
        opponent_edges = frozenset(
            (v, w)
            for v in region
            if game.owner[v] != player
            for w in game.successors[v]
            if w in region
        )
        chains[player], chain_nodes[player] = _chain_decomposition(
            region, allowed | opponent_edges, game.priority
        )
        chain_bdds[player], chain_relations[player], chain_ids[player] = (
            _chain_transition_bdd(chains[player])
        )
    possible_bdd, possible_relation, possible_chain_ids = _chain_transition_bdd(possible_chains)
    node_count = len(possible_bdd) + len(chain_bdds[0]) + len(chain_bdds[1])
    maximized = ForcedMultiStrategyResult(
        even_wins=regions[0],
        odd_wins=regions[1],
        even_strategy=strategies[0],
        odd_strategy=strategies[1],
        even_chains=chains[0],
        odd_chains=chains[1],
        even_chain_nodes=chain_nodes[0],
        odd_chain_nodes=chain_nodes[1],
        even_cycle_kernels=kernels[0],
        odd_cycle_kernels=kernels[1],
        justification=frozenset(justification_edges),
        possible_cycle_bdd=possible_bdd,
        possible_cycle_relation=possible_relation,
        possible_chain_ids=possible_chain_ids,
        possible_chain_nodes=possible_nodes,
        even_chain_bdd=chain_bdds[0],
        odd_chain_bdd=chain_bdds[1],
        even_chain_relation=chain_relations[0],
        odd_chain_relation=chain_relations[1],
        even_chain_ids=chain_ids[0],
        odd_chain_ids=chain_ids[1],
        iterations=solved.iterations,
        bdd_nodes=node_count,
    )
    initial = CQStrategySnapshot(
        even_wins=regions[0],
        odd_wins=regions[1],
        even_strategy=initial_strategies[0],
        odd_strategy=initial_strategies[1],
    )
    return CQMultiStrategyStages(initial=initial, maximized=maximized)


def chain_quotient_multistrategy(game: ParityGame) -> ForcedMultiStrategyResult:
    """Solve ``game`` and return CQ-MS's greedily maximised strategy."""
    return chain_quotient_multistrategy_stages(game).maximized


def trace_chain_quotient_solver(game: ParityGame):
    """Return the exact CQ-MS quotient, solver output, and recursive trace."""
    possible_chains, _ = _chain_decomposition(
        frozenset(game.vertices),
        frozenset((v, w) for v in game.vertices for w in game.successors[v]),
        game.priority,
    )
    quotient, _ = _compress_to_chain_game(game, possible_chains)
    events: list[ExplicitSolverTraceEvent] = []
    solution = _solve_cq_quotient(quotient, trace=events)
    return quotient, solution, tuple(events)
