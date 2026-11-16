"""Incremental, cached maximisation for Chain-Quotient multistrategies.

The conservative CQ seed is already forced-winning.  Adding one edge can only
introduce losing cycles that contain that edge.  This module therefore caches
transitive reachability in priority-threshold subgraphs and checks candidates
for newly closed opponent-parity cycles instead of solving a fresh parity game
for every candidate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable

from .cq_multistrategy import CQMultiStrategyStages, CQStrategySnapshot, _solve_cq_quotient
from .game import ParityGame
from .multistrategy import (
    ForcedMultiStrategyResult,
    _chain_decomposition,
    _chain_transition_bdd,
    _compress_to_chain_game,
    _cycle_kernels,
    _expand_solution,
    _justify_from_cycles,
    _strongly_connected_components,
)

Vertex = Hashable
Edge = tuple[Vertex, Vertex]


def _ordered_edges(edges):
    try:
        return tuple(sorted(edges))
    except TypeError:
        return tuple(sorted(edges, key=lambda edge: (repr(edge[0]), repr(edge[1]))))


def _iter_bits(mask: int):
    while mask:
        least = mask & -mask
        yield least.bit_length() - 1
        mask ^= least


@dataclass
class _ThresholdClosure:
    """Reflexive transitive closure for vertices with priority at most a limit."""

    priority: int
    eligible_mask: int
    at_priority_mask: int
    reachable: list[int]
    predecessors: list[int]

    def add_edge(self, source_id: int, target_id: int) -> None:
        source_bit = 1 << source_id
        target_bit = 1 << target_id
        if not (self.eligible_mask & source_bit and self.eligible_mask & target_bit):
            return
        if self.reachable[source_id] & target_bit:
            return

        sources = self.predecessors[source_id]
        targets = self.reachable[target_id]
        # For one inserted edge, every new path has the form
        # predecessor(source) -> source -> target -> successor(target).

        for predecessor_id in _iter_bits(sources):
            self.reachable[predecessor_id] |= targets
        for successor_id in _iter_bits(targets):
            self.predecessors[successor_id] |= sources


class IncrementalForcedWinCache:
    """Exact candidate-edge oracle for an already forced-winning strategy graph."""

    def __init__(
        self,
        game: ParityGame,
        player: int,
        region: frozenset[Vertex],
        allowed: frozenset[Edge],
    ) -> None:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        self.game = game
        self.player = player
        self.region = region
        self.allowed = set(allowed)
        self.vertices = tuple(sorted(region, key=lambda value: (type(value).__name__, repr(value))))
        self.ids = {vertex: index for index, vertex in enumerate(self.vertices)}
        self.adjacency = {
            vertex: set(
                target
                for target in game.successors[vertex]
                if target in region
                and (game.owner[vertex] != player or (vertex, target) in allowed)
            )
            for vertex in region
        }
        if any(not targets for targets in self.adjacency.values()):
            raise ValueError("initial multistrategy graph must be total on its winning region")

        losing_parity = 1 - player
        self.losing_priorities = tuple(
            priority
            for priority in sorted(set(game.priority[v] for v in region))
            if priority % 2 == losing_parity
        )
        self.closures = {
            priority: self._build_threshold_closure(priority)
            for priority in self.losing_priorities
        }

    def _build_threshold_closure(self, priority: int) -> _ThresholdClosure:
        eligible = {
            vertex for vertex in self.region if self.game.priority[vertex] <= priority
        }
        eligible_mask = sum(1 << self.ids[vertex] for vertex in eligible)
        at_priority_mask = sum(
            1 << self.ids[vertex]
            for vertex in eligible
            if self.game.priority[vertex] == priority
        )
        adjacency = {
            vertex: tuple(target for target in self.adjacency[vertex] if target in eligible)
            for vertex in eligible
        }
        components = _strongly_connected_components(frozenset(eligible), adjacency)
        component_of = {
            vertex: component_id
            for component_id, component in enumerate(components)
            for vertex in component
        }
        dag = [set() for _ in components]
        indegree = [0] * len(components)
        for source in eligible:
            source_component = component_of[source]
            for target in adjacency[source]:
                target_component = component_of[target]
                if source_component != target_component and target_component not in dag[source_component]:
                    dag[source_component].add(target_component)
                    indegree[target_component] += 1

        queue = deque(i for i, degree in enumerate(indegree) if degree == 0)
        topological = []
        while queue:
            component = queue.popleft()
            topological.append(component)
            for target in dag[component]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)

        component_vertices = [
            sum(1 << self.ids[vertex] for vertex in component)
            for component in components
        ]
        component_reachable = list(component_vertices)
        for component in reversed(topological):
            for target in dag[component]:
                component_reachable[component] |= component_reachable[target]

        component_predecessors = list(component_vertices)
        for component in topological:
            for target in dag[component]:
                component_predecessors[target] |= component_predecessors[component]

        count = len(self.vertices)
        reachable = [0] * count
        predecessors = [0] * count
        for vertex in eligible:
            identifier = self.ids[vertex]
            component = component_of[vertex]
            reachable[identifier] = component_reachable[component]
            predecessors[identifier] = component_predecessors[component]
        return _ThresholdClosure(
            priority=priority,
            eligible_mask=eligible_mask,
            at_priority_mask=at_priority_mask,
            reachable=reachable,
            predecessors=predecessors,
        )

    def can_add(self, edge: Edge) -> bool:
        """Return whether ``edge`` creates no opponent-winning cycle."""
        source, target = edge
        if source not in self.region or target not in self.region:
            return False
        if self.game.owner[source] != self.player or target not in self.game.successors[source]:
            return False
        if edge in self.allowed:
            return True

        endpoint_priority = max(self.game.priority[source], self.game.priority[target])
        source_id = self.ids[source]
        target_id = self.ids[target]
        for priority in self.losing_priorities:
            if priority < endpoint_priority:
                continue
            closure = self.closures[priority]
            losing_witnesses = (
                closure.reachable[target_id]
                & closure.predecessors[source_id]
                & closure.at_priority_mask
            )
            if losing_witnesses:
                return False
        return True

    def commit(self, edge: Edge) -> None:
        """Add a previously checked safe edge and update every cached closure."""
        if edge in self.allowed:
            return
        source, target = edge
        self.allowed.add(edge)
        self.adjacency[source].add(target)
        source_id = self.ids[source]
        target_id = self.ids[target]
        for closure in self.closures.values():
            closure.add_edge(source_id, target_id)

    def maximise(self) -> frozenset[Edge]:
        """Greedily produce an inclusion-maximal forced-winning edge set."""
        candidates = _ordered_edges(
            (vertex, target)
            for vertex in self.region
            if self.game.owner[vertex] == self.player
            for target in self.game.successors[vertex]
            if target in self.region and (vertex, target) not in self.allowed
        )
        for edge in candidates:
            if self.can_add(edge):
                self.commit(edge)
        return frozenset(self.allowed)


def cached_chain_quotient_multistrategy_stages(game: ParityGame) -> CQMultiStrategyStages:
    """Return conservative and cached-maximised CQ-MS strategies."""
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
    initial_strategies = {}
    strategies = {}
    chains = {}
    chain_nodes = {}
    kernels = {}
    justification_edges = set()
    chain_bdds = {}
    chain_relations = {}
    chain_ids = {}
    for player in (0, 1):
        region = regions[player]
        initial, player_justification, _ = _justify_from_cycles(
            game, player, region, selected
        )
        initial_strategies[player] = initial
        allowed = IncrementalForcedWinCache(game, player, region, initial).maximise()
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
        bdd_nodes=len(possible_bdd) + len(chain_bdds[0]) + len(chain_bdds[1]),
    )
    return CQMultiStrategyStages(
        initial=CQStrategySnapshot(
            even_wins=regions[0],
            odd_wins=regions[1],
            even_strategy=initial_strategies[0],
            odd_strategy=initial_strategies[1],
        ),
        maximized=maximized,
    )


def cached_chain_quotient_multistrategy(game: ParityGame) -> ForcedMultiStrategyResult:
    """Return the cached, incrementally maximised CQ-MS result."""
    return cached_chain_quotient_multistrategy_stages(game).maximized
