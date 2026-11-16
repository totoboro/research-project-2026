"""Forced-winning multistrategies and lossless chain views.

For each player, this constructs a forced-winning set of allowed moves on that
player's winning region. The chain-SCC construction is deliberately
conservative: it does not claim global inclusion-maximality among all safe
multistrategies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Mapping

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


def _ordered(values):
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=lambda value: str(value)))


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

    # Iterative Kosaraju: each vertex and edge is visited a constant number of times.
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
    # Remaining edges are disjoint directed cycles of 1-in/1-out vertices.
    for source in _ordered(vertices):
        for target in adjacency[source]:
            if (source, target) not in visited:
                chains.append(follow(source, target))
    return tuple(chains), frozenset(covered_nodes)


def _chain_transition_bdd(chains: tuple[PriorityChain, ...]):
    """Encode chain adjacency; paths in this relation are chain sequences.

    A transition joins two chains when the first ends at the vertex where the
    second begins. Consequently, every cycle in the graph is represented by a
    closed path in this BDD without listing every simple cycle.
    """
    count = len(chains)
    width = max(1, (max(2, count) - 1).bit_length())
    current = tuple(f"c{i}" for i in range(width))
    following = tuple(f"c{i}_next" for i in range(width))
    bdd = new_bdd(variable for pair in zip(reversed(current), reversed(following)) for variable in pair)
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
        predecessors = id_set(ends[vertex], current_by_bit)
        successors = id_set(starts[vertex], next_by_bit)
        relation |= predecessors & successors
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
        owner[state] = 0  # One successor, so this owner has no choice.
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


def _solve_explicit_parity_game(
    game: ParityGame,
    *,
    trace: list[ExplicitSolverTraceEvent] | None = None,
) -> _ExplicitSolution:
    """Solve a finite max-parity game using an explicit attractor recursion.

    When ``trace`` is supplied, append structured events describing every
    recursive call, attractor, intermediate solution, and return value.  The
    optional tracing does not change the solver's decisions.
    """
    predecessors = {v: [] for v in game.vertices}
    for source in game.vertices:
        for target in game.successors[source]:
            predecessors[target].append(source)

    solve_calls = 0

    def ordered_edges(choices):
        return tuple(sorted(choices.items(), key=lambda edge: (str(edge[0]), str(edge[1]))))

    def record(call_id, depth, action, vertices, **details):
        if trace is not None:
            trace.append(ExplicitSolverTraceEvent(
                call_id=call_id,
                depth=depth,
                action=action,
                vertices=_ordered(vertices),
                **details,
            ))

    def attractor(vertices: frozenset[Vertex], player: int, target: frozenset[Vertex]):
        attracted = set(target)
        choices = {}
        remaining = {
            v: sum(w in vertices for w in game.successors[v])
            for v in vertices if game.owner[v] != player
        }
        queue = deque(target)
        for v, count in remaining.items():
            if count == 0 and v not in attracted:
                attracted.add(v)
                queue.append(v)
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

    def solve(vertices: frozenset[Vertex], depth: int = 0):
        nonlocal solve_calls
        solve_calls += 1
        call_id = solve_calls
        record(call_id, depth, "enter", vertices)
        if not vertices:
            record(call_id, depth, "return-empty", vertices)
            return frozenset(), frozenset(), {0: {}, 1: {}}

        highest = max(game.priority[v] for v in vertices)
        alpha = highest % 2
        beta = 1 - alpha
        top = frozenset(v for v in vertices if game.priority[v] == highest)
        record(
            call_id, depth, "select-highest-priority", vertices,
            player=alpha, priority=highest, target=_ordered(top),
        )
        attract_alpha, strategy_alpha = attractor(vertices, alpha, top)
        record(
            call_id, depth, "first-attractor", vertices,
            player=alpha, priority=highest, target=_ordered(top),
            attracted=_ordered(attract_alpha), choices=ordered_edges(strategy_alpha),
        )
        sub_even, sub_odd, sub_strategy = solve(vertices - attract_alpha, depth + 1)
        record(
            call_id, depth, "first-recursion-result", vertices,
            even_wins=_ordered(sub_even), odd_wins=_ordered(sub_odd),
        )
        sub_regions = {0: sub_even, 1: sub_odd}
        if not sub_regions[beta]:
            regions = {0: set(sub_even), 1: set(sub_odd)}
            regions[alpha].update(attract_alpha)
            strategies = {0: dict(sub_strategy[0]), 1: dict(sub_strategy[1])}
            strategies[alpha].update(strategy_alpha)
            record(
                call_id, depth, "return-after-first-recursion", vertices,
                even_wins=_ordered(regions[0]), odd_wins=_ordered(regions[1]),
                choices=ordered_edges(strategies[alpha]),
            )
            return frozenset(regions[0]), frozenset(regions[1]), strategies

        attract_beta, strategy_beta = attractor(vertices, beta, sub_regions[beta])
        record(
            call_id, depth, "second-attractor", vertices,
            player=beta, target=_ordered(sub_regions[beta]),
            attracted=_ordered(attract_beta), choices=ordered_edges(strategy_beta),
        )
        rest_even, rest_odd, rest_strategy = solve(vertices - attract_beta, depth + 1)
        record(
            call_id, depth, "second-recursion-result", vertices,
            even_wins=_ordered(rest_even), odd_wins=_ordered(rest_odd),
        )
        regions = {0: set(rest_even), 1: set(rest_odd)}
        regions[beta].update(attract_beta)
        strategies = {0: dict(rest_strategy[0]), 1: dict(rest_strategy[1])}
        strategies[beta].update(sub_strategy[beta])
        strategies[beta].update(strategy_beta)
        record(
            call_id, depth, "return-after-second-recursion", vertices,
            even_wins=_ordered(regions[0]), odd_wins=_ordered(regions[1]),
            choices=ordered_edges(strategies[beta]),
        )
        return frozenset(regions[0]), frozenset(regions[1]), strategies

    even_wins, odd_wins, choices = solve(frozenset(game.vertices))
    regions = {0: even_wins, 1: odd_wins}
    # Attractor targets are seeded directly; complete their positional choice
    # with an edge that remains in the player's solved region.
    for player in (0, 1):
        for vertex in regions[player]:
            if game.owner[vertex] != player or vertex in choices[player]:
                continue
            winning_successors = tuple(w for w in game.successors[vertex] if w in regions[player])
            if not winning_successors:
                raise RuntimeError(f"solver found no winning move for player {player} at {vertex!r}")
            choices[player][vertex] = winning_successors[0]
    result = _ExplicitSolution(even_wins, odd_wins, choices[0], choices[1], solve_calls)
    record(
        1, 0, "complete-positional-strategies", frozenset(game.vertices),
        even_wins=_ordered(even_wins), odd_wins=_ordered(odd_wins),
        choices=ordered_edges({**choices[0], **choices[1]}),
    )
    return result


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
                        i for i, chain in enumerate(chains)
                        if chain.vertices[0] == vertex and chain.vertices[-1] in regions[player]
                    ]
                    if not winning_chains:
                        raise RuntimeError(f"no winning chain leaves {vertex!r}")
                    chain_id = winning_chains[0]
                selected[player][vertex] = chains[chain_id].vertices[1]

    # Interior vertices have exactly one outgoing edge; it is the only move
    # needed at those positions, regardless of the quotient solver's state IDs.
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
    selected_edges = frozenset((v, w) for v, w in selected[player].items())
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
                raise RuntimeError(f"opponent can leave player {player}'s winning region at {vertex!r}")
            adjacency[vertex] = targets

    components = _strongly_connected_components(region, adjacency)
    component_of = {v: i for i, component in enumerate(components) for v in component}
    dag = [set() for _ in components]
    indegree = [0] * len(components)
    for source in region:
        source_component = component_of[source]
        for target in adjacency[source]:
            target_component = component_of[target]
            if source_component != target_component and target_component not in dag[source_component]:
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
                raise RuntimeError(f"player {player}'s strategy has a priority-{priority} cycle")
            kernels.append(CycleKernel(component, priority, player))

        for vertex in component:
            if game.owner[vertex] == player:
                selected_target = selected[player][vertex]
                if (component_of[selected_target] != component_id
                        and component_of[selected_target] not in processed):
                    raise RuntimeError("selected strategy edge is not justified downstream")
                allowed_edges.add((vertex, selected_target))
                justification.add((vertex, selected_target))
                for target in game.successors[vertex]:
                    if target not in region:
                        continue
                    target_component = component_of[target]
                    if target_component != component_id:
                        if target_component not in processed:
                            continue
                        # Exiting a chain-SCC cannot return to it. These extra
                        # moves preserve the winning cycle reached downstream.
                        allowed_edges.add((vertex, target))
                        justification.add((vertex, target))
            else:
                for target in game.successors[vertex]:
                    target_component = component_of[target]
                    if target_component != component_id and target_component not in processed:
                        raise RuntimeError("opponent successor was not justified first")
                    justification.add((vertex, target))
        processed.add(component_id)

    kernels.sort(key=lambda kernel: min((str(v) for v in kernel.vertices), default=""))
    return frozenset(allowed_edges), frozenset(justification), tuple(kernels)


def chain_multistrategy(game: ParityGame) -> ForcedMultiStrategyResult:
    """Solve and justify forced multistrategies using chains and cycle kernels.

    This solver does not call FPJM or Zielonka. It compresses deterministic
    corridors to priority-carrying chain transitions, solves that quotient with
    an explicit attractor recursion, then constructs its own justification
    graph by propagating backwards from winning cycle kernels. Vertices outside
    cycle kernels may retain every move that decreases the justification rank.
    """
    possible_chains, possible_nodes = _chain_decomposition(
        frozenset(game.vertices),
        frozenset((v, w) for v in game.vertices for w in game.successors[v]),
        game.priority,
    )
    compressed, chain_state_to_id = _compress_to_chain_game(game, possible_chains)
    solved = _solve_explicit_parity_game(compressed)
    regions, selected = _expand_solution(game, possible_chains, compressed, chain_state_to_id, solved)
    strategies: dict[int, frozenset[Edge]] = {}
    chains: dict[int, tuple[PriorityChain, ...]] = {}
    chain_nodes: dict[int, frozenset[Vertex]] = {}
    kernels: dict[int, tuple[CycleKernel, ...]] = {}
    justification_edges = set()
    chain_bdds = {}
    chain_relations = {}
    chain_ids = {}
    for player in (0, 1):
        region = regions[player]
        allowed, player_justification, kernels[player] = _justify_from_cycles(
            game, player, region, selected
        )
        strategies[player] = allowed
        justification_edges.update(player_justification)
        chains[player], chain_nodes[player] = _chain_decomposition(region, allowed | frozenset(
            (v, w) for v in region if game.owner[v] != player for w in game.successors[v] if w in region
        ), game.priority)
        chain_bdds[player], chain_relations[player], chain_ids[player] = _chain_transition_bdd(chains[player])
    possible_bdd, possible_relation, possible_chain_ids = _chain_transition_bdd(possible_chains)
    node_count = len(possible_bdd) + len(chain_bdds[0]) + len(chain_bdds[1])
    return ForcedMultiStrategyResult(
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
