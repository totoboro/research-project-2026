"""Symbolic DFI and DFI-no-strategy parity-game solvers using ``dd`` BDDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .decision_diagrams import collect_bdd_garbage
from .fpjm import _BDD_SOLVER_GC_INTERVAL, _Encoding
from .game import ParityGame

Vertex = Hashable
Edge = tuple[Vertex, Vertex]


@dataclass(frozen=True)
class DFIResult:
    """Winning regions and, for DFI, strategy edges."""

    even_wins: frozenset[Vertex]
    odd_wins: frozenset[Vertex]
    even_strategy: frozenset[Edge]
    odd_strategy: frozenset[Edge]
    iterations: int
    bdd_nodes: int


def _solve_dfi(game: ParityGame, *, compute_strategy: bool, max_iterations: int) -> DFIResult:
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    enc = _Encoding(game)
    bdd = enc.bdd
    d = max(game.priority.values())
    Z = bdd.false  # current distraction set
    frozen = {p: bdd.false for p in range(d + 1)}
    even_edges = bdd.false
    odd_edges = bdd.false
    p = 0
    iterations = 0

    def even_estimate(distractions):
        # Vertices currently estimated to be won by Even.
        return (enc.priority_even & ~distractions) | (enc.priority_odd & distractions)

    def odd_estimate(distractions):
        return enc.V & ~even_estimate(distractions)

    def one_step_even(region, candidates):
        target_even = even_estimate(region)
        even_pre = enc.exists_next_edge(enc.E, target_even)
        odd_pre = enc.forall_next(enc.E, target_even)
        return (candidates & enc.owner[0] & even_pre) | (candidates & enc.owner[1] & odd_pre)

    while p <= d:
        if iterations >= max_iterations:
            raise RuntimeError(f"DFI did not converge within {max_iterations:,} iterations")

        alpha = p % 2
        candidates = enc.priorities[p] & ~Z
        for frozen_set in frozen.values():
            candidates &= ~frozen_set

        one_step = one_step_even(Z, candidates)
        if alpha == 0:
            new_distractions = candidates & ~one_step
        else:
            new_distractions = one_step
        Z |= new_distractions

        if compute_strategy:
            # Rebuild choices at the currently processed vertices. A winning
            # owner chooses an edge into its current estimated winning region.
            even_edges &= ~candidates
            odd_edges &= ~candidates
            even_edges |= enc.E & candidates & enc.owner[0] & enc.prime(even_estimate(Z))
            odd_edges |= enc.E & candidates & enc.owner[1] & enc.prime(odd_estimate(Z))

        iterations += 1
        if iterations % _BDD_SOLVER_GC_INTERVAL == 0:
            collect_bdd_garbage(bdd)
        if new_distractions != bdd.false:
            lower = bdd.false
            for lower_priority in range(p):
                lower |= enc.priorities[lower_priority]
            lower_unfrozen = lower
            for frozen_set in frozen.values():
                lower_unfrozen &= ~frozen_set

            current_even = even_estimate(Z)
            current_odd = odd_estimate(Z)
            alpha_wins = lower_unfrozen & (current_even if alpha == 0 else current_odd)
            frozen[p] |= lower_unfrozen & ~alpha_wins
            Z &= ~alpha_wins
            p = 0
        else:
            frozen[p] = bdd.false
            p += 1

    even_wins = enc.V & even_estimate(Z)
    odd_wins = enc.V & ~even_wins

    if compute_strategy:
        even_strategy_bdd = even_edges & enc.owner[0] & even_wins & enc.prime(even_wins)
        odd_strategy_bdd = odd_edges & enc.owner[1] & odd_wins & enc.prime(odd_wins)
        even_strategy = enc.edges_in(even_strategy_bdd)
        odd_strategy = enc.edges_in(odd_strategy_bdd)
    else:
        even_strategy = frozenset()
        odd_strategy = frozenset()

    collect_bdd_garbage(bdd)
    return DFIResult(
        even_wins=enc.vertices_in(even_wins),
        odd_wins=enc.vertices_in(odd_wins),
        even_strategy=even_strategy,
        odd_strategy=odd_strategy,
        iterations=iterations,
        bdd_nodes=len(bdd),
    )


def dfi(game: ParityGame, *, max_iterations: int = 1_000_000) -> DFIResult:
    """Solve a game with symbolic DFI and return winning strategy edges."""
    return _solve_dfi(game, compute_strategy=True, max_iterations=max_iterations)


def dfi_ns(game: ParityGame, *, max_iterations: int = 1_000_000) -> DFIResult:
    """Solve a game with symbolic DFI-no-strategy (skip strategy construction)."""
    return _solve_dfi(game, compute_strategy=False, max_iterations=max_iterations)
