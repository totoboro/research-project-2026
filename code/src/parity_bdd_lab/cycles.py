"""Enumerate unique directed simple parity-game cycles without strategy profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .game import ParityGame

Vertex = Hashable


@dataclass(frozen=True)
class ParityCycle:
    """A directed simple cycle, canonicalized up to rotation."""

    vertices: tuple[Vertex, ...]
    priorities: tuple[int, ...]
    greatest_recurring_priority: int
    winner: int


def _strongly_connected_components(vertices: set[Vertex], adjacency: dict[Vertex, tuple[Vertex, ...]]):
    """Tarjan SCC decomposition over the selected induced subgraph."""
    index = 0
    indices: dict[Vertex, int] = {}
    lowlink: dict[Vertex, int] = {}
    stack: list[Vertex] = []
    on_stack: set[Vertex] = set()
    components: list[set[Vertex]] = []

    def visit(vertex: Vertex) -> None:
        nonlocal index
        indices[vertex] = lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)

        for target in adjacency[vertex]:
            if target not in vertices:
                continue
            if target not in indices:
                visit(target)
                lowlink[vertex] = min(lowlink[vertex], lowlink[target])
            elif target in on_stack:
                lowlink[vertex] = min(lowlink[vertex], indices[target])

        if lowlink[vertex] == indices[vertex]:
            component = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == vertex:
                    break
            components.append(component)

    for vertex in sorted(vertices, key=lambda v: _order_key(v)):
        if vertex not in indices:
            visit(vertex)
    return components


def _order_key(vertex: Vertex):
    return (type(vertex).__name__, repr(vertex))


def _canonical_rotation(cycle: tuple[Vertex, ...], order: dict[Vertex, int]) -> tuple[Vertex, ...]:
    rotations = (cycle[i:] + cycle[:i] for i in range(len(cycle)))
    return min(rotations, key=lambda rotation: tuple(order[v] for v in rotation))


def enumerate_unique_cycles(game: ParityGame) -> tuple[ParityCycle, ...]:
    """Return every unique directed simple cycle in ``game``.

    Cycles that differ only by their starting vertex are returned once; reverse
    edge directions remain distinct. Johnson's algorithm enumerates elementary
    cycles directly, avoiding the product of out-degrees used by positional
    strategy-profile enumeration.
    """
    vertices = game.vertices
    order = {vertex: index for index, vertex in enumerate(vertices)}
    # Parallel duplicate edges do not make distinct vertex cycles.
    adjacency = {vertex: tuple(dict.fromkeys(game.successors[vertex])) for vertex in vertices}
    remaining = set(vertices)
    found: set[tuple[Vertex, ...]] = set()

    while remaining:
        components = _strongly_connected_components(remaining, adjacency)
        cyclic = [
            component for component in components
            if len(component) > 1
            or any(vertex in adjacency[vertex] for vertex in component)
        ]
        if not cyclic:
            break
        component = min(cyclic, key=lambda c: min(order[v] for v in c))
        start = min(component, key=order.__getitem__)
        blocked: set[Vertex] = set()
        dependents: dict[Vertex, set[Vertex]] = {vertex: set() for vertex in component}
        path: list[Vertex] = []

        def unblock(vertex: Vertex) -> None:
            blocked.discard(vertex)
            while dependents[vertex]:
                dependent = dependents[vertex].pop()
                if dependent in blocked:
                    unblock(dependent)

        def circuit(vertex: Vertex) -> bool:
            found_cycle = False
            path.append(vertex)
            blocked.add(vertex)
            for target in adjacency[vertex]:
                if target not in component:
                    continue
                if target == start:
                    found.add(_canonical_rotation(tuple(path), order))
                    found_cycle = True
                elif target not in blocked and circuit(target):
                    found_cycle = True
            if found_cycle:
                unblock(vertex)
            else:
                for target in adjacency[vertex]:
                    if target in component:
                        dependents[target].add(vertex)
            path.pop()
            return found_cycle

        circuit(start)
        remaining.remove(start)

    result = []
    for cycle in sorted(found, key=lambda c: tuple(order[v] for v in c)):
        priorities = tuple(game.priority[vertex] for vertex in cycle)
        greatest = max(priorities)
        result.append(ParityCycle(cycle, priorities, greatest, greatest % 2))
    return tuple(result)


@dataclass(frozen=True)
class ForcedParityCycle:
    """A cycle compatible with a winning strategy that guarantees its winner."""

    cycle: ParityCycle
    starts: tuple[Vertex, ...]
    winning_strategy: tuple[tuple[Vertex, Vertex], ...]


def enumerate_forced_winning_cycles(game: ParityGame) -> tuple[ForcedParityCycle, ...]:
    """Find cycles whose winner can guarantee a win while committing to the cycle.

    For each cycle, restrict the cycle winner's choices at their owned cycle
    vertices to the corresponding cycle edges, then solve the restricted game.
    The cycle is returned when the winner can still win from at least one cycle
    vertex. The opponent retains all choices, including the cycle edge, so the
    cycle remains a possible outcome. Zielonka is used here to avoid enumerating
    positional strategy profiles.
    """
    from .zielonka import zielonka

    forced = []
    for cycle in enumerate_unique_cycles(game):
        player = cycle.winner
        vertices = cycle.vertices
        successor_on_cycle = {
            vertex: vertices[(index + 1) % len(vertices)]
            for index, vertex in enumerate(vertices)
        }
        restricted_successors = {
            vertex: (
                (successor_on_cycle[vertex],)
                if vertex in successor_on_cycle and game.owner[vertex] == player
                else game.successors[vertex]
            )
            for vertex in game.vertices
        }
        restricted_game = ParityGame(
            owner=game.owner,
            priority=game.priority,
            successors=restricted_successors,
            name=f"{game.name}:cycle-{vertices}",
        )
        result = zielonka(restricted_game)
        winning_region = result.even_wins if player == 0 else result.odd_wins
        starts = tuple(vertex for vertex in vertices if vertex in winning_region)
        if starts:
            strategy = result.even_strategy if player == 0 else result.odd_strategy
            # Keep the complete witness strategy over the winner's region; on
            # this cycle it is constrained to the cycle's outgoing edges.
            forced.append(ForcedParityCycle(cycle, starts, tuple(sorted(strategy, key=lambda edge: (str(edge[0]), str(edge[1]))))))
    return tuple(forced)
