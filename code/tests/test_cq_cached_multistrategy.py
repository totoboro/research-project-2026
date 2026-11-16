from __future__ import annotations

import random
from pathlib import Path

import pytest

from parity_bdd_lab import (
    ParityGame,
    cached_chain_quotient_multistrategy,
    chain_quotient_multistrategy,
    read_pg,
    validate_multistrategy,
)


GAMES = Path(__file__).parents[1] / "data" / "games"
BENCHMARKS = Path(__file__).parents[1] / "data" / "benchmarks" / "syntcomp2020-pgame"


@pytest.mark.parametrize("path", sorted(GAMES.glob("*.pg")), ids=lambda path: path.name)
def test_cached_cq_matches_zielonka_checked_cq_on_examples(path: Path) -> None:
    game = read_pg(path)

    reference = chain_quotient_multistrategy(game)
    cached = cached_chain_quotient_multistrategy(game)

    assert cached.even_wins == reference.even_wins
    assert cached.odd_wins == reference.odd_wins
    assert cached.even_strategy == reference.even_strategy
    assert cached.odd_strategy == reference.odd_strategy


def test_cached_cq_matches_reference_and_is_complete_on_lilydemo01() -> None:
    game = read_pg(BENCHMARKS / "lilydemo01.tlsf.ehoa.oink")

    reference = chain_quotient_multistrategy(game)
    cached = cached_chain_quotient_multistrategy(game)
    report = validate_multistrategy(game, cached)

    assert cached.even_strategy == reference.even_strategy
    assert cached.odd_strategy == reference.odd_strategy
    assert report.valid
    assert report.complete is True


def test_cached_candidate_oracle_matches_reference_on_seeded_random_games() -> None:
    rng = random.Random(20261116)
    for game_number in range(20):
        vertices = tuple(range(6))
        game = ParityGame(
            owner={vertex: rng.randrange(2) for vertex in vertices},
            priority={vertex: rng.randrange(6) for vertex in vertices},
            successors={
                vertex: tuple(
                    target for target in vertices if rng.random() < 0.4
                ) or (rng.choice(vertices),)
                for vertex in vertices
            },
            name=f"cached-random-{game_number}",
        )

        reference = chain_quotient_multistrategy(game)
        cached = cached_chain_quotient_multistrategy(game)

        assert cached.even_wins == reference.even_wins
        assert cached.odd_wins == reference.odd_wins
        assert cached.even_strategy == reference.even_strategy
        assert cached.odd_strategy == reference.odd_strategy
