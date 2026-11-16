from pathlib import Path

import pytest

from parity_bdd_lab import (
    CycleEnumerationLimitError,
    ParityGame,
    chain_multistrategy,
    chain_quotient_multistrategy,
    chain_quotient_multistrategy_stages,
    enumerate_multistrategy_cycles,
    enumerate_unique_cycles,
    read_pg,
    trace_chain_quotient_solver,
    validate_multistrategy,
)


GAMES = Path(__file__).parents[1] / "data" / "games"
BENCHMARKS = Path(__file__).parents[1] / "data" / "benchmarks" / "syntcomp2020-pgame"


def test_cq_ms_matches_chain_ms_baseline_on_image_example() -> None:
    labels = dict(enumerate("abcdefghi"))
    game = read_pg(GAMES / "image_example.pg", vertex_labels=labels)

    original = chain_multistrategy(game)
    copied = chain_quotient_multistrategy(game)

    assert copied.even_wins == original.even_wins
    assert copied.odd_wins == original.odd_wins
    assert copied.even_strategy == original.even_strategy
    assert copied.odd_strategy == original.odd_strategy
    assert copied.justification == original.justification


def test_cq_ms_revisits_deferred_edges_and_is_maximal_on_lilydemo01() -> None:
    game = read_pg(BENCHMARKS / "lilydemo01.tlsf.ehoa.oink")

    conservative = chain_multistrategy(game)
    maximised = chain_quotient_multistrategy(game)
    report = validate_multistrategy(game, maximised)

    assert (0, 10) not in conservative.odd_strategy
    assert (0, 10) in maximised.odd_strategy
    assert (0, 7) not in maximised.odd_strategy
    assert maximised.odd_strategy > conservative.odd_strategy
    assert report.valid
    assert report.complete is True
    assert report.for_player(1).addable_edges == frozenset()


def test_cq_stages_expose_all_permitted_recurrent_cycles() -> None:
    game = read_pg(BENCHMARKS / "lilydemo01.tlsf.ehoa.oink")

    stages = chain_quotient_multistrategy_stages(game)
    initial_cycles = enumerate_multistrategy_cycles(game, stages.initial, 1)
    maximized_cycles = enumerate_multistrategy_cycles(game, stages.maximized, 1)

    assert stages.initial.odd_strategy < stages.maximized.odd_strategy
    assert tuple(cycle.vertices for cycle in initial_cycles) == ((5, 35, 34),)
    assert tuple(cycle.vertices for cycle in maximized_cycles) == ((5, 35, 34),)
    assert all(cycle.winner == 1 for cycle in initial_cycles + maximized_cycles)


def test_cycle_enumeration_limit_stops_exponential_output() -> None:
    game = ParityGame(
        owner={0: 0, 1: 0},
        priority={0: 0, 1: 0},
        successors={0: (0, 1), 1: (0, 1)},
    )

    with pytest.raises(CycleEnumerationLimitError, match="max_cycles=1"):
        enumerate_unique_cycles(game, max_cycles=1)


def test_image_example_explicit_solver_trace_is_complete() -> None:
    labels = dict(enumerate("abcdefghi"))
    game = read_pg(GAMES / "image_example.pg", vertex_labels=labels)

    quotient, solution, events = trace_chain_quotient_solver(game)

    assert len(quotient.vertices) == 20
    assert solution.iterations == 21
    assert not solution.even_wins
    assert solution.odd_wins == frozenset(quotient.vertices)
    assert events[0].action == "enter"
    assert events[-1].action == "complete-positional-strategies"
    assert sum(event.action == "enter" for event in events) == solution.iterations
    assert any(event.action == "second-attractor" for event in events)
