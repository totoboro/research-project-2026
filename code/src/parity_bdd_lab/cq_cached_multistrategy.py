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
from typing import Hashable, Mapping

import numpy as np

from .decision_diagrams import new_bdd
from .game import ParityGame

Vertex = Hashable
Edge = tuple[Vertex, Vertex]


@dataclass(frozen=True)
class PriorityChain:
    """A maximal 1-in/1-out path with its maximum vertex priority."""

    vertices: tuple[Vertex, ...]
    priority: int


@dataclass(frozen=True)
class CycleKernel:
    """An SCC encoding a family of recurrent cycles with one winner."""

    vertices: frozenset[Vertex]
    priority: int
    winner: int


@dataclass(frozen=True)
class _ChainState:
    chain_id: int


@dataclass(frozen=True)
class _ExplicitSolution:
    even_wins: frozenset[Vertex]
    odd_wins: frozenset[Vertex]
    even_choices: Mapping[Vertex, Vertex]
    odd_choices: Mapping[Vertex, Vertex]
    iterations: int


@dataclass(frozen=True)
class ExplicitSolverTraceEvent:
    """One observable step of the recursive explicit parity-game solver."""

    call_id: int
    depth: int
    action: str
    vertices: tuple[Vertex, ...]
    player: int | None = None
    priority: int | None = None
    target: tuple[Vertex, ...] = ()
    attracted: tuple[Vertex, ...] = ()
    even_wins: tuple[Vertex, ...] = ()
    odd_wins: tuple[Vertex, ...] = ()
    choices: tuple[Edge, ...] = ()


@dataclass(frozen=True)
class ForcedMultiStrategyResult:
    """Forced-winning move sets, cycle kernels, and symbolic chain relations."""

    even_wins: frozenset[Vertex]
    odd_wins: frozenset[Vertex]
    even_strategy: frozenset[Edge]
    odd_strategy: frozenset[Edge]
    even_chains: tuple[PriorityChain, ...]
    odd_chains: tuple[PriorityChain, ...]
    even_chain_nodes: frozenset[Vertex]
    odd_chain_nodes: frozenset[Vertex]
    even_cycle_kernels: tuple[CycleKernel, ...]
    odd_cycle_kernels: tuple[CycleKernel, ...]
    justification: frozenset[Edge]
    possible_cycle_bdd: object
    possible_cycle_relation: object
    possible_chain_ids: tuple[PriorityChain, ...]
    possible_chain_nodes: frozenset[Vertex]
    even_chain_bdd: object
    odd_chain_bdd: object
    even_chain_relation: object
    odd_chain_relation: object
    even_chain_ids: tuple[PriorityChain, ...]
    odd_chain_ids: tuple[PriorityChain, ...]
    iterations: int
    bdd_nodes: int


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


def _ordered(values):
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=lambda value: str(value)))


def _ordered_edges(edges):
    try:
        return tuple(sorted(edges))
    except TypeError:
        return tuple(sorted(edges, key=lambda edge: (repr(edge[0]), repr(edge[1]))))


def _cycle_kernels(
    game: ParityGame,
    player: int,
    region: frozenset[Vertex],
    allowed: frozenset[Edge],
) -> tuple[CycleKernel, ...]:
    """Group the strategy graph's possible recurrent cycles by SCC in O(V+E)."""
    adjacency = {}
    reverse = {v: [] for v in region}
    for vertex in region:
        targets = (
            tuple(w for w in game.successors[vertex] if (vertex, w) in allowed)
            if game.owner[vertex] == player else game.successors[vertex]
        )
        targets = tuple(w for w in targets if w in region)
        adjacency[vertex] = targets
        for target in targets:
            reverse[target].append(vertex)

    seen: set[Vertex] = set()
    finish: list[Vertex] = []
    for start in _ordered(region):
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, 0)]
        while stack:
            vertex, offset = stack[-1]
            targets = adjacency[vertex]
            if offset < len(targets):
                target = targets[offset]
                stack[-1] = (vertex, offset + 1)
                if target not in seen:
                    seen.add(target)
                    stack.append((target, 0))
            else:
                stack.pop()
                finish.append(vertex)

    seen.clear()
    kernels = []
    for start in reversed(finish):
        if start in seen:
            continue
        component = set()
        stack = [start]
        seen.add(start)
        while stack:
            vertex = stack.pop()
            component.add(vertex)
            for target in reverse[vertex]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        cyclic = len(component) > 1 or any(v in adjacency[v] for v in component)
        if not cyclic:
            continue
        priority = max(game.priority[v] for v in component)
        if priority % 2 != player:
            raise RuntimeError(
                f"justified strategy has a cycle kernel won by player {priority % 2}, "
                f"not player {player}"
            )
        kernels.append(CycleKernel(frozenset(component), priority, player))
    kernels.sort(key=lambda kernel: min((str(v) for v in kernel.vertices), default=""))
    return tuple(kernels)


def _chain_decomposition(
    vertices: frozenset[Vertex],
    edges: frozenset[Edge],
    priorities: Mapping[Vertex, int],
) -> tuple[tuple[PriorityChain, ...], frozenset[Vertex]]:
    """Partition edges into maximal 1-in/1-out chains without losing labels."""
    adjacency = {v: [] for v in vertices}
    indegree = {v: 0 for v in vertices}
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1
    for targets in adjacency.values():
        targets.sort(key=str)
    visited: set[Edge] = set()
    covered_nodes: set[Vertex] = set()
    chains: list[PriorityChain] = []

    def follow(source: Vertex, target: Vertex) -> PriorityChain:
        path = [source, target]
        chain_priority = max(priorities[source], priorities[target])
        visited.add((source, target))
        current = target
        while indegree[current] == 1 and len(adjacency[current]) == 1:
            next_vertex = adjacency[current][0]
            edge = (current, next_vertex)
            if edge in visited:
                break
            visited.add(edge)
            path.append(next_vertex)
            chain_priority = max(chain_priority, priorities[next_vertex])
            current = next_vertex
        covered_nodes.update(path)
        return PriorityChain(tuple(path), chain_priority)

    for source in _ordered(vertices):
        if indegree[source] == 1 and len(adjacency[source]) == 1:
            continue
        for target in adjacency[source]:
            if (source, target) not in visited:
                chains.append(follow(source, target))
    for source in _ordered(vertices):
        for target in adjacency[source]:
            if (source, target) not in visited:
                chains.append(follow(source, target))
    return tuple(chains), frozenset(covered_nodes)


def _chain_transition_bdd(chains: tuple[PriorityChain, ...]):
    """Encode chain adjacency as a symbolic transition relation."""
    count = len(chains)
    width = max(1, (max(2, count) - 1).bit_length())
    current = tuple(f"c{i}" for i in range(width))
    following = tuple(f"c{i}_next" for i in range(width))
    bdd = new_bdd(
        variable
        for pair in zip(reversed(current), reversed(following))
        for variable in pair
    )
    current_by_bit = {bit: current[bit] for bit in range(width)}
    next_by_bit = {bit: following[bit] for bit in range(width)}
    cache = {}

    def id_set(values, variables_by_bit):
        codes = tuple(sorted(set(values)))
        key = (tuple(variables_by_bit[bit] for bit in range(width)), codes)
        if key in cache:
            return cache[key]

        def build(bit, subset):
            if not subset:
                return bdd.false
            if bit < 0 or len(subset) == (1 << (bit + 1)):
                return bdd.true
            low_codes = [code for code in subset if not (code & (1 << bit))]
            high_codes = [code for code in subset if code & (1 << bit)]
            low = build(bit - 1, low_codes)
            high = build(bit - 1, high_codes)
            if low == high:
                return low
            return bdd.ite(bdd.var(variables_by_bit[bit]), high, low)

        result = build(width - 1, list(codes))
        cache[key] = result
        return result

    starts: dict[Vertex, list[int]] = {}
    ends: dict[Vertex, list[int]] = {}
    for chain_id, chain in enumerate(chains):
        starts.setdefault(chain.vertices[0], []).append(chain_id)
        ends.setdefault(chain.vertices[-1], []).append(chain_id)
    relation = bdd.false
    for vertex in starts.keys() & ends.keys():
        relation |= id_set(ends[vertex], current_by_bit) & id_set(
            starts[vertex], next_by_bit
        )
    return bdd, relation, chains


def _compress_to_chain_game(
    game: ParityGame,
    chains: tuple[PriorityChain, ...],
) -> tuple[ParityGame, dict[_ChainState, int]]:
    """Turn deterministic corridors into edge-priority states."""
    junctions = {chain.vertices[0] for chain in chains}
    owner = {v: game.owner[v] for v in junctions}
    priority = {v: 0 for v in junctions}
    successors = {v: [] for v in junctions}
    chain_state_to_id = {}
    for chain_id, chain in enumerate(chains):
        state = _ChainState(chain_id)
        chain_state_to_id[state] = chain_id
        owner[state] = 0
        priority[state] = chain.priority
        successors[state] = [chain.vertices[-1]]
        successors[chain.vertices[0]].append(state)
    compressed = ParityGame(
        owner=owner,
        priority=priority,
        successors={v: tuple(ws) for v, ws in successors.items()},
        name=f"{game.name} [chain quotient]",
    )
    return compressed, chain_state_to_id


def _solve_explicit_parity_game(game: ParityGame) -> _ExplicitSolution:
    """Solve a finite max-parity game using explicit attractor recursion."""
    predecessors = {v: [] for v in game.vertices}
    for source in game.vertices:
        for target in game.successors[source]:
            predecessors[target].append(source)

    solve_calls = 0

    def attractor(vertices: frozenset[Vertex], player: int, target: frozenset[Vertex]):
        attracted = set(target)
        choices = {}
        remaining = {
            v: sum(w in vertices for w in game.successors[v])
            for v in vertices
            if game.owner[v] != player
        }
        queue = deque(target)
        for vertex, count in remaining.items():
            if count == 0 and vertex not in attracted:
                attracted.add(vertex)
                queue.append(vertex)
        while queue:
            reached = queue.popleft()
            for source in predecessors[reached]:
                if source not in vertices or source in attracted:
                    continue
                if game.owner[source] == player:
                    attracted.add(source)
                    choices[source] = reached
                    queue.append(source)
                else:
                    remaining[source] -= 1
                    if remaining[source] == 0:
                        attracted.add(source)
                        queue.append(source)
        return frozenset(attracted), choices

    def solve(vertices: frozenset[Vertex]):
        nonlocal solve_calls
        solve_calls += 1
        if not vertices:
            return frozenset(), frozenset(), {0: {}, 1: {}}

        highest = max(game.priority[v] for v in vertices)
        alpha = highest % 2
        beta = 1 - alpha
        top = frozenset(v for v in vertices if game.priority[v] == highest)
        attract_alpha, strategy_alpha = attractor(vertices, alpha, top)
        sub_even, sub_odd, sub_strategy = solve(vertices - attract_alpha)
        sub_regions = {0: sub_even, 1: sub_odd}
        if not sub_regions[beta]:
            regions = {0: set(sub_even), 1: set(sub_odd)}
            regions[alpha].update(attract_alpha)
            strategies = {0: dict(sub_strategy[0]), 1: dict(sub_strategy[1])}
            strategies[alpha].update(strategy_alpha)
            return frozenset(regions[0]), frozenset(regions[1]), strategies

        attract_beta, strategy_beta = attractor(vertices, beta, sub_regions[beta])
        rest_even, rest_odd, rest_strategy = solve(vertices - attract_beta)
        regions = {0: set(rest_even), 1: set(rest_odd)}
        regions[beta].update(attract_beta)
        strategies = {0: dict(rest_strategy[0]), 1: dict(rest_strategy[1])}
        strategies[beta].update(sub_strategy[beta])
        strategies[beta].update(strategy_beta)
        return frozenset(regions[0]), frozenset(regions[1]), strategies

    even_wins, odd_wins, choices = solve(frozenset(game.vertices))
    regions = {0: even_wins, 1: odd_wins}
    for player in (0, 1):
        for vertex in regions[player]:
            if game.owner[vertex] != player or vertex in choices[player]:
                continue
            winning_successors = tuple(
                w for w in game.successors[vertex] if w in regions[player]
            )
            if not winning_successors:
                raise RuntimeError(
                    f"solver found no winning move for player {player} at {vertex!r}"
                )
            choices[player][vertex] = winning_successors[0]
    return _ExplicitSolution(
        even_wins, odd_wins, choices[0], choices[1], solve_calls
    )


def _solve_cq_quotient(game: ParityGame):
    """Solve CQ-MS's quotient using the local explicit parity-game solver."""
    return _solve_explicit_parity_game(game)


def _expand_solution(game, chains, compressed, chain_state_to_id, solution):
    """Map compressed winners and positional choices back to original vertices."""
    junctions = {chain.vertices[0] for chain in chains}
    regions = {0: set(), 1: set()}
    for vertex in junctions:
        winner = 0 if vertex in solution.even_wins else 1
        regions[winner].add(vertex)
    chain_winners = {}
    for chain_id, chain in enumerate(chains):
        state = _ChainState(chain_id)
        winner = 0 if state in solution.even_wins else 1
        chain_winners[chain_id] = winner
        regions[winner].update(chain.vertices[1:-1])

    selected = {0: {}, 1: {}}
    choices = {0: solution.even_choices, 1: solution.odd_choices}
    for player in (0, 1):
        for vertex in regions[player]:
            if game.owner[vertex] != player:
                continue
            if vertex in junctions:
                chain_state = choices[player].get(vertex)
                chain_id = chain_state_to_id.get(chain_state)
                if chain_id is None:
                    winning_chains = [
                        i
                        for i, chain in enumerate(chains)
                        if chain.vertices[0] == vertex
                        and chain.vertices[-1] in regions[player]
                    ]
                    if not winning_chains:
                        raise RuntimeError(f"no winning chain leaves {vertex!r}")
                    chain_id = winning_chains[0]
                selected[player][vertex] = chains[chain_id].vertices[1]

    for chain_id, chain in enumerate(chains):
        winner = chain_winners[chain_id]
        for source, target in zip(chain.vertices[1:-1], chain.vertices[2:]):
            if game.owner[source] == winner:
                selected[winner][source] = target
    return {p: frozenset(regions[p]) for p in (0, 1)}, selected


def _strongly_connected_components(vertices, adjacency):
    """Return SCCs using iterative Kosaraju traversal."""
    reverse = {v: [] for v in vertices}
    for source in vertices:
        for target in adjacency[source]:
            reverse[target].append(source)
    seen = set()
    finish = []
    for start in _ordered(vertices):
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, 0)]
        while stack:
            vertex, offset = stack[-1]
            targets = adjacency[vertex]
            if offset < len(targets):
                target = targets[offset]
                stack[-1] = (vertex, offset + 1)
                if target not in seen:
                    seen.add(target)
                    stack.append((target, 0))
            else:
                stack.pop()
                finish.append(vertex)
    seen.clear()
    components = []
    for start in reversed(finish):
        if start in seen:
            continue
        component = set()
        stack = [start]
        seen.add(start)
        while stack:
            vertex = stack.pop()
            component.add(vertex)
            for target in reverse[vertex]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        components.append(frozenset(component))
    return components


def _justify_from_cycles(game, player, region, selected):
    """Seed cyclic SCCs, then justify their condensation graph backwards."""
    if not region:
        return frozenset(), frozenset(), ()
    adjacency = {}
    for vertex in region:
        if game.owner[vertex] == player:
            target = selected[player].get(vertex)
            if target not in game.successors[vertex] or target not in region:
                raise RuntimeError(f"no positional winning move at {vertex!r}")
            adjacency[vertex] = (target,)
        else:
            targets = tuple(game.successors[vertex])
            if any(target not in region for target in targets):
                raise RuntimeError(
                    f"opponent can leave player {player}'s winning region at {vertex!r}"
                )
            adjacency[vertex] = targets

    components = _strongly_connected_components(region, adjacency)
    component_of = {
        vertex: component_id
        for component_id, component in enumerate(components)
        for vertex in component
    }
    dag = [set() for _ in components]
    indegree = [0] * len(components)
    for source in region:
        source_component = component_of[source]
        for target in adjacency[source]:
            target_component = component_of[target]
            if (
                source_component != target_component
                and target_component not in dag[source_component]
            ):
                dag[source_component].add(target_component)
                indegree[target_component] += 1

    queue = deque(i for i, count in enumerate(indegree) if count == 0)
    topological = []
    while queue:
        component = queue.popleft()
        topological.append(component)
        for target in dag[component]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(topological) != len(components):
        raise RuntimeError("SCC condensation unexpectedly contains a cycle")

    kernels = []
    allowed_edges = set()
    justification = set()
    processed = set()
    for component_id in reversed(topological):
        component = components[component_id]
        cyclic = len(component) > 1 or any(v in adjacency[v] for v in component)
        if cyclic:
            priority = max(game.priority[v] for v in component)
            if priority % 2 != player:
                raise RuntimeError(
                    f"player {player}'s strategy has a priority-{priority} cycle"
                )
            kernels.append(CycleKernel(component, priority, player))

        for vertex in component:
            if game.owner[vertex] == player:
                selected_target = selected[player][vertex]
                if (
                    component_of[selected_target] != component_id
                    and component_of[selected_target] not in processed
                ):
                    raise RuntimeError(
                        "selected strategy edge is not justified downstream"
                    )
                allowed_edges.add((vertex, selected_target))
                justification.add((vertex, selected_target))
                for target in game.successors[vertex]:
                    if target not in region:
                        continue
                    target_component = component_of[target]
                    if target_component != component_id:
                        if target_component not in processed:
                            continue
                        allowed_edges.add((vertex, target))
                        justification.add((vertex, target))
            else:
                for target in game.successors[vertex]:
                    target_component = component_of[target]
                    if (
                        target_component != component_id
                        and target_component not in processed
                    ):
                        raise RuntimeError(
                            "opponent successor was not justified first"
                        )
                    justification.add((vertex, target))
        processed.add(component_id)

    kernels.sort(
        key=lambda kernel: min((str(v) for v in kernel.vertices), default="")
    )
    return frozenset(allowed_edges), frozenset(justification), tuple(kernels)


@dataclass
class _ThresholdClosure:
    """Reflexive transitive closure for vertices with priority at most a limit."""

    priority: int
    eligible_mask: int
    at_priority_mask: np.ndarray
    reachable: np.ndarray
    predecessors: np.ndarray
    vertex_count: int

    def _indices(self, mask: np.ndarray) -> np.ndarray:
        """Decode a packed uint64 mask in native NumPy code."""
        bits = np.unpackbits(mask.view(np.uint8), bitorder="little")
        return np.flatnonzero(bits[: self.vertex_count])

    def edge_delta(
        self, source_id: int, target_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return the predecessor/successor rectangle introduced by an edge."""
        source_bit = 1 << source_id
        target_bit = 1 << target_id
        if not (self.eligible_mask & source_bit and self.eligible_mask & target_bit):
            return None
        target_word = target_id >> 6
        packed_target_bit = np.uint64(1) << np.uint64(target_id & 63)
        if self.reachable[source_id, target_word] & packed_target_bit:
            return None
        # Copy both rows: the vectorised update below can modify their backing
        # matrices, while this rank-one update must use the pre-insertion sets.
        sources = self.predecessors[source_id].copy()
        targets = self.reachable[target_id].copy()
        return sources, targets, self._indices(sources), self._indices(targets)

    def apply_delta(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        predecessor_ids: np.ndarray,
        successor_ids: np.ndarray,
    ) -> None:
        """Materialise one Boolean rank-one transitive-closure update."""
        self.reachable[predecessor_ids] |= targets
        self.predecessors[successor_ids] |= sources

    def add_edge(self, source_id: int, target_id: int) -> None:
        delta = self.edge_delta(source_id, target_id)
        if delta is None:
            return
        # For one inserted edge, every new path has the form
        # predecessor(source) -> source -> target -> successor(target).
        self.apply_delta(*delta)


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
        self.commits = 0
        self.closure_edges_added = 0
        self.game = game
        self.player = player
        self.region = region
        self.allowed = set(allowed)
        self.vertices = tuple(sorted(region, key=lambda value: (type(value).__name__, repr(value))))
        self.ids = {vertex: index for index, vertex in enumerate(self.vertices)}
        # Rebuilding uses the SCC condensation and is preferable when one
        # incremental rectangle would touch nearly the whole region.  The
        # floor avoids rebuilding small closures for moderate fan-out.
        self.rebuild_fanout = max(1024, len(self.vertices))
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
        block_count = (count + 63) // 64

        def packed(mask: int) -> np.ndarray:
            return np.frombuffer(
                mask.to_bytes(block_count * 8, "little"), dtype=np.uint64
            ).copy()

        return _ThresholdClosure(
            priority=priority,
            eligible_mask=eligible_mask,
            at_priority_mask=packed(at_priority_mask),
            reachable=np.vstack(tuple(packed(mask) for mask in reachable)),
            predecessors=np.vstack(tuple(packed(mask) for mask in predecessors)),
            vertex_count=count,
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
            losing_witnesses = np.any(
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
        endpoint_priority = max(
            self.game.priority[source], self.game.priority[target]
        )

        self.commits += 1
        for priority in self.losing_priorities:
            if priority < endpoint_priority:
                continue
            closure = self.closures[priority]
            delta = closure.edge_delta(source_id, target_id)
            if delta is None:
                continue
            self.closure_edges_added += 1
            sources, targets, predecessor_ids, successor_ids = delta
            if len(predecessor_ids) + len(successor_ids) >= self.rebuild_fanout:
                self.closures[priority] = self._build_threshold_closure(priority)
            else:
                closure.apply_delta(
                    sources, targets, predecessor_ids, successor_ids
                )

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
        forced_win_cache =  IncrementalForcedWinCache(game, player, region, initial)
        allowed = forced_win_cache.maximise()
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
