"""Independent validation and quality reports for parity-game multistrategies.

A forced-winning multistrategy must win under every resolution of its allowed
moves.  Validation therefore turns every choice in the restricted strategy
graph over to the opposing player and solves that universal-choice game.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10
from time import perf_counter
from typing import Callable, Hashable, Protocol

from .game import ParityGame
from .zielonka import ZielonkaResult, zielonka

Vertex = Hashable
Edge = tuple[Vertex, Vertex]


class MultiStrategyResult(Protocol):
    """Minimum solver-result interface accepted by :func:`validate_multistrategy`."""

    even_wins: frozenset[Vertex]
    odd_wins: frozenset[Vertex]
    even_strategy: frozenset[Edge]
    odd_strategy: frozenset[Edge]


@dataclass(frozen=True)
class LassoWitness:
    """An ultimately periodic play witnessing that a multistrategy can lose."""

    start: Vertex
    prefix: tuple[Vertex, ...]
    cycle: tuple[Vertex, ...]
    cycle_priorities: tuple[int, ...]
    greatest_recurring_priority: int
    winner: int


@dataclass(frozen=True)
class VertexChoiceReport:
    """Allowed, omitted, and safely addable choices at one controlled vertex."""

    vertex: Vertex
    priority: int
    legal_targets: tuple[Vertex, ...]
    allowed_targets: tuple[Vertex, ...]
    omitted_targets: tuple[Vertex, ...]
    addable_targets: tuple[Vertex, ...]


@dataclass(frozen=True)
class MaximalityProgress:
    """Progress event emitted while checking omitted edges."""

    player: int
    completed: int
    total: int
    edge: Edge
    elapsed_seconds: float


@dataclass(frozen=True)
class PlayerValidationReport:
    """Validation and permissiveness results for one player."""

    player: int
    region: frozenset[Vertex]
    reference_region: frozenset[Vertex]
    missing_reference_vertices: frozenset[Vertex]
    unexpected_vertices: frozenset[Vertex]
    unknown_region_vertices: frozenset[Vertex]
    controlled_vertices: frozenset[Vertex]
    allowed_edges: frozenset[Edge]
    malformed_edges: tuple[str, ...]
    illegal_edges: frozenset[Edge]
    wrong_owner_edges: frozenset[Edge]
    outside_region_sources: frozenset[Edge]
    escaping_edges: frozenset[Edge]
    opponent_escape_edges: frozenset[Edge]
    missing_choice_vertices: frozenset[Vertex]
    structurally_valid: bool
    forced_winning: bool | None
    losing_starts: frozenset[Vertex]
    counterexample: LassoWitness | None
    maximality_candidates: frozenset[Edge]
    maximality_checked: bool
    addable_edges: frozenset[Edge]
    inclusion_maximal: bool | None
    possible_edge_count: int
    retained_edge_count: int
    allowed_edge_coverage: float
    multi_choice_vertices: int
    multi_choice_coverage: float
    positional_refinements: int
    log10_positional_refinements: float
    validation_seconds: float
    maximality_seconds: float
    vertex_choices: tuple[VertexChoiceReport, ...]


@dataclass(frozen=True)
class MultiStrategyValidationReport:
    """Complete validation result for both players."""

    game_name: str
    vertices: int
    edges: int
    regions_disjoint: bool
    regions_exhaustive: bool
    regions_correct: bool
    structurally_valid: bool
    forced_winning: bool
    inclusion_maximal: bool | None
    valid: bool
    complete: bool | None
    reference: str
    players: tuple[PlayerValidationReport, PlayerValidationReport]
    total_seconds: float

    def for_player(self, player: int) -> PlayerValidationReport:
        """Return the per-player report."""
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return self.players[player]


def _ordered(values):
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=repr))


def _normalise_edges(values) -> tuple[frozenset[Edge], tuple[str, ...]]:
    edges: set[Edge] = set()
    malformed: list[str] = []
    try:
        raw_values = tuple(values)
    except TypeError:
        return frozenset(), (repr(values),)
    for raw in raw_values:
        try:
            source, target = raw
            hash(source)
            hash(target)
        except (TypeError, ValueError):
            malformed.append(repr(raw))
            continue
        edges.add((source, target))
    return frozenset(edges), tuple(malformed)


def _restricted_universal_game(
    game: ParityGame,
    player: int,
    region: frozenset[Vertex],
    allowed: frozenset[Edge],
) -> ParityGame:
    adversary = 1 - player
    successors = {}
    for vertex in region:
        if game.owner[vertex] == player:
            successors[vertex] = tuple(
                target for target in game.successors[vertex]
                if (vertex, target) in allowed and target in region
            )
        else:
            successors[vertex] = tuple(
                target for target in game.successors[vertex] if target in region
            )
    return ParityGame(
        owner={vertex: adversary for vertex in region},
        priority={vertex: game.priority[vertex] for vertex in region},
        successors=successors,
        name=f"{game.name} [player {player} universal validation]",
    )


def _lasso_from_solution(
    game: ParityGame,
    solution: ZielonkaResult,
    player: int,
    start: Vertex,
) -> LassoWitness:
    adversary = 1 - player
    losing_region = solution.even_wins if adversary == 0 else solution.odd_wins
    strategy = solution.even_strategy if adversary == 0 else solution.odd_strategy
    choices = dict(strategy)
    first_seen: dict[Vertex, int] = {}
    path: list[Vertex] = []
    vertex = start
    while vertex not in first_seen:
        first_seen[vertex] = len(path)
        path.append(vertex)
        target = choices.get(vertex)
        if target is None:
            target = next(w for w in game.successors[vertex] if w in losing_region)
        vertex = target
    cycle_start = first_seen[vertex]
    cycle = tuple(path[cycle_start:])
    greatest = max(game.priority[v] for v in cycle)
    return LassoWitness(
        start=start,
        prefix=tuple(path[:cycle_start]),
        cycle=cycle,
        cycle_priorities=tuple(game.priority[v] for v in cycle),
        greatest_recurring_priority=greatest,
        winner=greatest % 2,
    )


def _check_forced_win(
    game: ParityGame,
    player: int,
    region: frozenset[Vertex],
    allowed: frozenset[Edge],
    *,
    witness: bool,
) -> tuple[bool, frozenset[Vertex], LassoWitness | None]:
    if not region:
        return True, frozenset(), None
    restricted = _restricted_universal_game(game, player, region, allowed)
    solution = zielonka(restricted)
    wins = solution.even_wins if player == 0 else solution.odd_wins
    losing = region - wins
    counterexample = None
    if losing and witness:
        counterexample = _lasso_from_solution(
            restricted, solution, player, _ordered(losing)[0]
        )
    return not losing, frozenset(losing), counterexample


def _validate_player(
    game: ParityGame,
    result: MultiStrategyResult,
    reference: ZielonkaResult,
    player: int,
    *,
    check_maximality: bool,
    progress: Callable[[MaximalityProgress], None] | None,
) -> PlayerValidationReport:
    started = perf_counter()
    region = frozenset(result.even_wins if player == 0 else result.odd_wins)
    reference_region = reference.even_wins if player == 0 else reference.odd_wins
    raw_strategy = result.even_strategy if player == 0 else result.odd_strategy
    allowed, malformed = _normalise_edges(raw_strategy)
    vertices = set(game.vertices)
    game_edges = frozenset(
        (source, target)
        for source in game.vertices
        for target in game.successors[source]
    )
    controlled = frozenset(v for v in region if v in vertices and game.owner[v] == player)
    unknown_region_vertices = region - vertices

    illegal = frozenset(edge for edge in allowed if edge not in game_edges)
    wrong_owner = frozenset(
        edge for edge in allowed
        if edge[0] in vertices and game.owner[edge[0]] != player
    )
    outside_sources = frozenset(edge for edge in allowed if edge[0] not in region)
    escaping = frozenset(edge for edge in allowed if edge[1] not in region)
    usable = allowed - illegal - wrong_owner - outside_sources - escaping
    sources = {source for source, _ in usable}
    missing_choices = controlled - sources
    opponent_escapes = frozenset(
        (source, target)
        for source in region
        if source in vertices and game.owner[source] != player
        for target in game.successors[source]
        if target not in region
    )
    structurally_valid = not any((
        malformed, unknown_region_vertices, illegal, wrong_owner, outside_sources, escaping,
        missing_choices, opponent_escapes,
    ))

    forced_winning: bool | None = None
    losing_starts: frozenset[Vertex] = frozenset()
    counterexample = None
    if structurally_valid:
        forced_winning, losing_starts, counterexample = _check_forced_win(
            game, player, region, allowed, witness=True
        )
    validation_seconds = perf_counter() - started

    possible_edges = frozenset(
        (source, target)
        for source in controlled
        for target in game.successors[source]
        if target in region
    )
    candidates = possible_edges - allowed
    addable: set[Edge] = set()
    maximality_seconds = 0.0
    maximality_checked = False
    inclusion_maximal: bool | None = None
    if check_maximality and forced_winning is True:
        maximality_started = perf_counter()
        ordered_candidates = _ordered(candidates)
        total = len(ordered_candidates)
        for completed, edge in enumerate(ordered_candidates, 1):
            remains_winning, _, _ = _check_forced_win(
                game, player, region, allowed | {edge}, witness=False
            )
            if remains_winning:
                addable.add(edge)
            if progress is not None:
                progress(MaximalityProgress(
                    player=player,
                    completed=completed,
                    total=total,
                    edge=edge,
                    elapsed_seconds=perf_counter() - maximality_started,
                ))
        maximality_seconds = perf_counter() - maximality_started
        maximality_checked = True
        inclusion_maximal = not addable

    choice_counts = {
        vertex: sum((vertex, target) in allowed for target in game.successors[vertex])
        for vertex in controlled
    }
    refinements = 1
    refinement_log = 0.0
    for count in choice_counts.values():
        refinements *= count
        if count == 0:
            refinement_log = float("-inf")
        elif refinement_log != float("-inf"):
            refinement_log += log10(count)
    retained_count = len(allowed & possible_edges)
    possible_count = len(possible_edges)
    controlled_count = len(controlled)
    addable_frozen = frozenset(addable)
    vertex_choices = tuple(
        VertexChoiceReport(
            vertex=vertex,
            priority=game.priority[vertex],
            legal_targets=_ordered(
                target for source, target in possible_edges if source == vertex
            ),
            allowed_targets=_ordered(
                target for source, target in allowed & possible_edges if source == vertex
            ),
            omitted_targets=_ordered(
                target for source, target in candidates if source == vertex
            ),
            addable_targets=_ordered(
                target for source, target in addable_frozen if source == vertex
            ),
        )
        for vertex in _ordered(controlled)
    )
    return PlayerValidationReport(
        player=player,
        region=region,
        reference_region=reference_region,
        missing_reference_vertices=reference_region - region,
        unexpected_vertices=region - reference_region,
        unknown_region_vertices=unknown_region_vertices,
        controlled_vertices=controlled,
        allowed_edges=allowed,
        malformed_edges=malformed,
        illegal_edges=illegal,
        wrong_owner_edges=wrong_owner,
        outside_region_sources=outside_sources,
        escaping_edges=escaping,
        opponent_escape_edges=opponent_escapes,
        missing_choice_vertices=missing_choices,
        structurally_valid=structurally_valid,
        forced_winning=forced_winning,
        losing_starts=losing_starts,
        counterexample=counterexample,
        maximality_candidates=candidates,
        maximality_checked=maximality_checked,
        addable_edges=addable_frozen,
        inclusion_maximal=inclusion_maximal,
        possible_edge_count=possible_count,
        retained_edge_count=retained_count,
        allowed_edge_coverage=(retained_count / possible_count if possible_count else 1.0),
        multi_choice_vertices=sum(count > 1 for count in choice_counts.values()),
        multi_choice_coverage=(
            sum(count > 1 for count in choice_counts.values()) / controlled_count
            if controlled_count else 1.0
        ),
        positional_refinements=refinements,
        log10_positional_refinements=refinement_log,
        validation_seconds=validation_seconds,
        maximality_seconds=maximality_seconds,
        vertex_choices=vertex_choices,
    )


def validate_multistrategy(
    game: ParityGame,
    result: MultiStrategyResult,
    *,
    check_maximality: bool = True,
    progress: Callable[[MaximalityProgress], None] | None = None,
) -> MultiStrategyValidationReport:
    """Validate a two-player forced-winning multistrategy result.

    Completeness means inclusion-maximality on each player's true winning
    region.  It is exact: every omitted legal in-region edge is added in turn
    and the universal-choice game is solved again.  Set ``check_maximality`` to
    false to skip that potentially expensive phase; completeness is then
    reported as unknown.
    """
    started = perf_counter()
    even_region = frozenset(result.even_wins)
    odd_region = frozenset(result.odd_wins)
    vertices = frozenset(game.vertices)
    reference = zielonka(game)
    players = tuple(
        _validate_player(
            game, result, reference, player,
            check_maximality=check_maximality,
            progress=progress,
        )
        for player in (0, 1)
    )
    regions_disjoint = even_region.isdisjoint(odd_region)
    regions_exhaustive = even_region | odd_region == vertices
    regions_correct = (
        even_region == reference.even_wins
        and odd_region == reference.odd_wins
    )
    structurally_valid = regions_disjoint and regions_exhaustive and all(
        report.structurally_valid for report in players
    )
    forced_winning = all(report.forced_winning is True for report in players)
    if all(report.inclusion_maximal is not None for report in players):
        inclusion_maximal: bool | None = all(
            report.inclusion_maximal is True for report in players
        )
    else:
        inclusion_maximal = None
    valid = regions_correct and structurally_valid and forced_winning
    complete = inclusion_maximal if valid else False
    return MultiStrategyValidationReport(
        game_name=game.name,
        vertices=len(game.vertices),
        edges=sum(len(game.successors[v]) for v in game.vertices),
        regions_disjoint=regions_disjoint,
        regions_exhaustive=regions_exhaustive,
        regions_correct=regions_correct,
        structurally_valid=structurally_valid,
        forced_winning=forced_winning,
        inclusion_maximal=inclusion_maximal,
        valid=valid,
        complete=complete,
        reference="Zielonka",
        players=players,  # type: ignore[arg-type]
        total_seconds=perf_counter() - started,
    )
