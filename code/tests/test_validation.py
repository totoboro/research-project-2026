from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from parity_bdd_lab import (
    ParityGame,
    chain_multistrategy,
    enumerate_unique_cycles,
    read_pg,
    validate_multistrategy,
    zielonka,
)


GAMES = Path(__file__).parents[1] / "data" / "games"


@pytest.mark.parametrize("path", sorted(GAMES.glob("*.pg")), ids=lambda p: p.name)
def test_chain_multistrategy_passes_bundled_games(path: Path) -> None:
    game = read_pg(path)
    report = validate_multistrategy(game, chain_multistrategy(game))

    assert report.regions_correct
    assert report.structurally_valid
    assert report.forced_winning
    assert report.valid
    assert report.complete is True


def test_nonmaximal_strategy_reports_safely_addable_edge() -> None:
    game = ParityGame(
        owner={"a": 0, "b": 1},
        priority={"a": 0, "b": 0},
        successors={"a": ("a", "b"), "b": ("b",)},
        name="nonmaximal",
    )
    result = SimpleNamespace(
        even_wins=frozenset({"a", "b"}),
        odd_wins=frozenset(),
        even_strategy=frozenset({("a", "a")}),
        odd_strategy=frozenset(),
    )

    report = validate_multistrategy(game, result)
    even = report.for_player(0)

    assert report.valid
    assert report.complete is False
    assert even.inclusion_maximal is False
    assert even.addable_edges == frozenset({("a", "b")})
    assert even.positional_refinements == 1
    assert even.allowed_edge_coverage == 0.5


def test_losing_multistrategy_returns_concrete_odd_cycle() -> None:
    game = ParityGame(
        owner={"a": 0, "b": 0},
        priority={"a": 0, "b": 1},
        successors={"a": ("a", "b"), "b": ("a", "b")},
        name="losing-union",
    )
    result = SimpleNamespace(
        even_wins=frozenset({"a", "b"}),
        odd_wins=frozenset(),
        even_strategy=frozenset({("a", "a"), ("a", "b"), ("b", "b")}),
        odd_strategy=frozenset(),
    )

    report = validate_multistrategy(game, result)
    witness = report.for_player(0).counterexample

    assert report.structurally_valid
    assert not report.forced_winning
    assert not report.valid
    assert witness is not None
    assert witness.winner == 1
    assert witness.greatest_recurring_priority % 2 == 1
    assert witness.cycle
    assert all(
        witness.cycle[(index + 1) % len(witness.cycle)]
        in game.successors[vertex]
        for index, vertex in enumerate(witness.cycle)
    )


def test_structural_errors_are_reported_without_solving_restricted_game() -> None:
    game = ParityGame(
        owner={0: 0, 1: 1},
        priority={0: 0, 1: 1},
        successors={0: (0,), 1: (1,)},
    )
    result = SimpleNamespace(
        even_wins=frozenset({0}),
        odd_wins=frozenset({1}),
        even_strategy=[(0, 1), ("malformed",)],
        odd_strategy=frozenset({(0, 0)}),
    )

    report = validate_multistrategy(game, result)

    assert not report.structurally_valid
    assert report.for_player(0).malformed_edges
    assert report.for_player(0).escaping_edges == frozenset({(0, 1)})
    assert report.for_player(0).missing_choice_vertices == frozenset({0})
    assert report.for_player(1).wrong_owner_edges == frozenset({(0, 0)})
    assert report.for_player(0).forced_winning is None


def test_unknown_region_vertex_is_reported_without_crashing() -> None:
    game = ParityGame(owner={0: 0}, priority={0: 0}, successors={0: (0,)})
    result = SimpleNamespace(
        even_wins=frozenset({0, "unknown"}),
        odd_wins=frozenset(),
        even_strategy=frozenset({(0, 0)}),
        odd_strategy=frozenset(),
    )

    report = validate_multistrategy(game, result)

    assert not report.structurally_valid
    assert report.for_player(0).unknown_region_vertices == frozenset({"unknown"})
    assert report.for_player(0).forced_winning is None


def test_skipping_maximality_reports_completeness_unknown() -> None:
    game = read_pg(GAMES / "image_example.pg")
    report = validate_multistrategy(
        game, chain_multistrategy(game), check_maximality=False
    )

    assert report.valid
    assert report.inclusion_maximal is None
    assert report.complete is None
    assert all(not player.maximality_checked for player in report.players)


def test_progress_reports_every_candidate() -> None:
    game = ParityGame(
        owner={"a": 0, "b": 1},
        priority={"a": 0, "b": 0},
        successors={"a": ("a", "b"), "b": ("b",)},
    )
    result = SimpleNamespace(
        even_wins=frozenset({"a", "b"}), odd_wins=frozenset(),
        even_strategy=frozenset({("a", "a")}), odd_strategy=frozenset(),
    )
    events = []

    validate_multistrategy(game, result, progress=events.append)

    assert len(events) == 1
    assert events[0].completed == events[0].total == 1
    assert events[0].edge == ("a", "b")


def test_universal_result_agrees_with_direct_cycle_enumeration() -> None:
    rng = random.Random(20261115)
    for game_number in range(20):
        vertices = tuple(range(4))
        game = ParityGame(
            owner={v: rng.randrange(2) for v in vertices},
            priority={v: rng.randrange(5) for v in vertices},
            successors={
                v: tuple(w for w in vertices if rng.random() < 0.45) or (rng.choice(vertices),)
                for v in vertices
            },
            name=f"random-{game_number}",
        )
        solved = zielonka(game)
        strategies = {}
        for player, region in ((0, solved.even_wins), (1, solved.odd_wins)):
            strategies[player] = frozenset(
                (v, w)
                for v in region if game.owner[v] == player
                for w in game.successors[v] if w in region
            )
        result = SimpleNamespace(
            even_wins=solved.even_wins,
            odd_wins=solved.odd_wins,
            even_strategy=strategies[0],
            odd_strategy=strategies[1],
        )
        report = validate_multistrategy(game, result, check_maximality=False)

        for player, region in ((0, solved.even_wins), (1, solved.odd_wins)):
            if not region:
                expected = True
            else:
                restricted = ParityGame(
                    owner={v: 0 for v in region},
                    priority={v: game.priority[v] for v in region},
                    successors={
                        v: tuple(
                            w for w in game.successors[v]
                            if w in region and (
                                game.owner[v] != player or (v, w) in strategies[player]
                            )
                        )
                        for v in region
                    },
                )
                expected = all(
                    cycle.winner == player
                    for cycle in enumerate_unique_cycles(restricted)
                )
            assert report.for_player(player).forced_winning is expected
