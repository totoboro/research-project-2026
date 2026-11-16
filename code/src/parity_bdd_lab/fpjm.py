"""Symbolic FPJM parity-game solver using BDDs from :mod:`dd`.

The implementation follows Algorithm 2 (symbolic FPJ) from Lijzenga and
van Dijk, *Symbolic Parity Game Solvers that Yield Winning Strategies*.
Vertices are encoded as binary IDs; both vertex sets and edge relations are BDDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Hashable

from .decision_diagrams import collect_bdd_garbage, get_bdd_class
from .game import ParityGame

Vertex = Hashable
Edge = tuple[Vertex, Vertex]
_BDD_GC_INTERVAL = 256
_BDD_SOLVER_GC_INTERVAL = 8


@dataclass(frozen=True)
class FPJMResult:
    """Winning regions and symbolic justification multi-strategies returned by FPJM."""

    even_wins: frozenset[Vertex]
    odd_wins: frozenset[Vertex]
    even_strategy: frozenset[Edge]
    odd_strategy: frozenset[Edge]
    justification: frozenset[Edge]
    iterations: int
    bdd_nodes: int


class _Encoding:
    """BDD encoding of vertices, owners, priorities, and transition relation."""

    def __init__(self, game: ParityGame):
        self.game = game
        self.vertices = game.vertices
        self.ids = {vertex: index for index, vertex in enumerate(self.vertices)}
        self.labels = {index: vertex for vertex, index in self.ids.items()}
        self.width = max(1, ceil(log2(max(2, len(self.vertices)))))
        self.x = tuple(f"v{i}" for i in range(self.width))
        self.xp = tuple(f"v{i}_next" for i in range(self.width))
        self.bdd = get_bdd_class()()
        # Keep each source bit adjacent to its corresponding destination bit.
        # Grouping all current-state bits before all next-state bits can make
        # sparse transition relations exponentially larger as BDDs.
        interleaved = tuple(
            variable for pair in zip(reversed(self.x), reversed(self.xp)) for variable in pair
        )
        self.bdd.declare(*interleaved)
        self.V = self._set(self.vertices)
        self.owner = {
            player: self._set(v for v in self.vertices if game.owner[v] == player)
            for player in (0, 1)
        }
        max_priority = max(game.priority.values())
        self.priorities = {
            p: self._set(v for v in self.vertices if game.priority[v] == p)
            for p in range(max_priority + 1)
        }
        self.priority_even = self._set(v for v in self.vertices if game.priority[v] % 2 == 0)
        self.priority_odd = self._set(v for v in self.vertices if game.priority[v] % 2 == 1)
        self.E = self.bdd.false
        self._state_cubes = {}
        self._next_cubes = {}
        self.edges: tuple[Edge, ...] = tuple(
            (v, w) for v in self.vertices for w in game.successors[v]
        )
        # Build the transition relation one source row at a time. OR-ing one
        # complete source/target cube per edge creates large intermediate
        # BDDs on sparse games with many vertices. Sharing the source prefix
        # and grouping its target set is substantially more compact.
        for row_number, source in enumerate(self.vertices, 1):
            targets = self.bdd.false
            for target in game.successors[source]:
                targets |= self._next_cube(target)
            self.E |= self._cube(source) & targets
            del targets
            if row_number % _BDD_GC_INTERVAL == 0:
                collect_bdd_garbage(self.bdd)
        self._state_cubes.clear()
        self._next_cubes.clear()
        collect_bdd_garbage(self.bdd)

    def _bits(self, identifier: int, variables: tuple[str, ...]) -> dict[str, bool]:
        return {variable: bool((identifier >> bit) & 1) for bit, variable in enumerate(variables)}

    def _cube(self, source: Vertex, target: Vertex | None = None):
        if target is not None:
            assignment = self._bits(self.ids[source], self.x)
            assignment.update(self._bits(self.ids[target], self.xp))
            return self.bdd.cube(assignment)
        if source not in self._state_cubes:
            self._state_cubes[source] = self.bdd.cube(self._bits(self.ids[source], self.x))
        return self._state_cubes[source]

    def _next_cube(self, vertex: Vertex):
        if vertex not in self._next_cubes:
            self._next_cubes[vertex] = self.bdd.cube(self._bits(self.ids[vertex], self.xp))
        return self._next_cubes[vertex]

    def _set(self, vertices):
        """Build a vertex set as a bit trie, sharing common binary prefixes."""
        identifiers = sorted({self.ids[vertex] for vertex in vertices})
        if not identifiers:
            return self.bdd.false

        def build(bit: int, codes: list[int]):
            if not codes:
                return self.bdd.false
            if bit < 0 or len(codes) == (1 << (bit + 1)):
                return self.bdd.true
            low_codes = [code for code in codes if not (code & (1 << bit))]
            high_codes = [code for code in codes if code & (1 << bit)]
            low = build(bit - 1, low_codes)
            high = build(bit - 1, high_codes)
            if low == high:
                return low
            return self.bdd.ite(self.bdd.var(self.x[bit]), high, low)

        return build(self.width - 1, identifiers)

    def prime(self, vertex_set):
        return self.bdd.let(dict(zip(self.x, self.xp)), vertex_set)

    def exists_next(self, relation):
        return self.bdd.exist(self.xp, relation)

    def exists_next_edge(self, edge_relation, targets):
        return self.V & self.exists_next(edge_relation & self.prime(targets))

    def forall_next(self, edge_relation, targets):
        return self.V & self.bdd.forall(self.xp, ~edge_relation | self.prime(targets))

    def vertices_in(self, vertex_set) -> frozenset[Vertex]:
        result = set()
        valid_vertices = vertex_set & self.V
        for assignment in self.bdd.pick_iter(valid_vertices, care_vars=set(self.x)):
            identifier = self._identifier(assignment, self.x)
            if identifier in self.labels:
                result.add(self.labels[identifier])
        return frozenset(result)

    def edges_in(self, edge_set) -> frozenset[Edge]:
        result = set()
        valid_edges = edge_set & self.E
        care_vars = set(self.x + self.xp)
        for assignment in self.bdd.pick_iter(valid_edges, care_vars=care_vars):
            source_id = self._identifier(assignment, self.x)
            target_id = self._identifier(assignment, self.xp)
            if source_id in self.labels and target_id in self.labels:
                result.add((self.labels[source_id], self.labels[target_id]))
        return frozenset(result)

    @staticmethod
    def _identifier(assignment, variables: tuple[str, ...]) -> int:
        return sum(1 << bit for bit, variable in enumerate(variables)
                   if assignment.get(variable, False))


def fpjm(game: ParityGame, *, max_iterations: int = 1_000_000) -> FPJMResult:
    """Solve ``game`` using symbolic FPJM.

    Returns the even and odd winning regions plus the justification edges used
    as each player's strategy. Variables are binary-coded vertex IDs in the
    selected ``dd`` BDD backend. The iteration limit guards against accidental
    nontermination if the implementation or input is changed incorrectly.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    enc = _Encoding(game)
    bdd = enc.bdd

    # Initially, Even is assumed to win precisely the even-priority vertices.
    Z = enc.priority_even
    J = bdd.false

    def phi(region):
        # One step in which Even-owned vertices can choose, and Odd-owned
        # vertices must have every successor, inside Even's estimated region.
        return (
            enc.owner[0] & enc.exists_next_edge(enc.E, region)
        ) | (
            enc.owner[1] & enc.forall_next(enc.E, region)
        )

    def reaches(justification, target):
        reached = target
        while True:
            previous = reached
            reached = enc.V & (reached | enc.exists_next_edge(justification, reached))
            if reached == previous:
                return reached

    def strategy_diamond(region, unjustified):
        even_winning_owner_vertices = unjustified & enc.owner[0] & region
        odd_winning_owner_vertices = unjustified & enc.owner[1] & ~region
        losing_for_owner = unjustified & ~(
            (enc.owner[0] & region) | (enc.owner[1] & ~region)
        ) & enc.V

        good_even_edges = enc.E & even_winning_owner_vertices & enc.prime(region)
        good_odd_edges = enc.E & odd_winning_owner_vertices & enc.prime(~region & enc.V)
        all_losing_edges = enc.E & losing_for_owner
        return good_even_edges | good_odd_edges | all_losing_edges

    def next_step(region, justification, unjustified):
        priority = next(
            p for p in range(max(game.priority.values()) + 1)
            if (enc.priorities[p] & unjustified) != bdd.false
        )
        at_priority = unjustified & enc.priorities[priority]
        changed = at_priority & bdd.apply("xor", region, phi(region))

        if changed != bdd.false:
            reset = reaches(justification, changed)
            lower = bdd.false
            for p in range(priority):
                lower |= enc.priorities[p]
            higher = bdd.false
            for p in range(priority + 1, max(game.priority.values()) + 1):
                higher |= enc.priorities[p]

            if priority % 2 == 0:
                reset_lower = (region & ~(reset & enc.priority_odd)) & lower
            else:
                reset_lower = (region | (reset & enc.priority_even)) & lower

            updated_region = (region & higher) | bdd.apply("xor", region & enc.priorities[priority], changed) | reset_lower
            updated_justification = (justification & ~reset) | strategy_diamond(updated_region, changed)
        else:
            updated_region = region
            updated_justification = justification | strategy_diamond(updated_region, at_priority)
        return updated_region, updated_justification

    unjustified = enc.V & ~enc.exists_next(J)
    iterations = 0
    while unjustified != bdd.false:
        if iterations >= max_iterations:
            raise RuntimeError(f"FPJM did not converge within {max_iterations:,} iterations")
        Z, J = next_step(Z, J, unjustified)
        unjustified = enc.V & ~enc.exists_next(J)
        iterations += 1
        if iterations % _BDD_SOLVER_GC_INTERVAL == 0:
            collect_bdd_garbage(bdd)

    even_wins = Z & enc.V
    odd_wins = enc.V & ~Z

    # The symbolic algorithm can retain every winning edge. Restrict to edges
    # whose source is controlled by the corresponding winner and in its region.
    even_strategy = J & enc.owner[0] & even_wins & enc.prime(even_wins)
    odd_strategy = J & enc.owner[1] & odd_wins & enc.prime(odd_wins)
    collect_bdd_garbage(bdd)
    return FPJMResult(
        even_wins=enc.vertices_in(even_wins),
        odd_wins=enc.vertices_in(odd_wins),
        even_strategy=enc.edges_in(even_strategy),
        odd_strategy=enc.edges_in(odd_strategy),
        justification=enc.edges_in(J),
        iterations=iterations,
        bdd_nodes=len(bdd),
    )
