from __future__ import annotations

import random
from pathlib import Path

import pytest

from parity_bdd_lab import (
    ParityGame,
    oink_fpi,
    oink_fpim,
    oink_fpj,
    oink_fpjm,
    read_pg,
    validate_multistrategy,
    zielonka,
)


GAMES = Path(__file__).parents[1] / "data" / "games"
SOLVERS = (oink_fpi, oink_fpim, oink_fpj, oink_fpjm)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("path", sorted(GAMES.glob("*.pg")), ids=lambda path: path.name)
def test_oink_solver_matches_zielonka_on_bundled_games(solver, path: Path) -> None:
    game = read_pg(path)
    expected = zielonka(game)
    actual = solver(game)

    assert actual.even_wins == expected.even_wins
    assert actual.odd_wins == expected.odd_wins
    assert validate_multistrategy(game, actual, check_maximality=False).valid


@pytest.mark.parametrize("solver", SOLVERS)
def test_oink_solver_matches_zielonka_on_seeded_random_games(solver) -> None:
    randomizer = random.Random(20261116)
    for size in range(1, 8):
        vertices = tuple(range(size))
        for game_number in range(25):
            game = ParityGame(
                owner={vertex: randomizer.randrange(2) for vertex in vertices},
                priority={vertex: randomizer.randrange(8) for vertex in vertices},
                successors={
                    vertex: tuple(
                        target for target in vertices if randomizer.random() < 0.35
                    )
                    or (randomizer.choice(vertices),)
                    for vertex in vertices
                },
                name=f"random-{size}-{game_number}",
            )
            expected = zielonka(game)
            actual = solver(game)

            assert actual.even_wins == expected.even_wins
            assert actual.odd_wins == expected.odd_wins
            assert validate_multistrategy(
                game, actual, check_maximality=False
            ).valid


@pytest.mark.parametrize("solver", SOLVERS)
def test_oink_solver_rejects_nonpositive_iteration_limit(solver) -> None:
    game = ParityGame(owner={0: 0}, priority={0: 0}, successors={0: (0,)})

    with pytest.raises(ValueError, match="positive"):
        solver(game, max_iterations=0)


def test_multi_strategy_variants_retain_all_immediate_winning_moves() -> None:
    game = ParityGame(
        owner={"choice": 0, "left": 1, "right": 1},
        priority={"choice": 0, "left": 0, "right": 2},
        successors={
            "choice": ("left", "right"),
            "left": ("left",),
            "right": ("right",),
        },
    )

    expected = frozenset({("choice", "left"), ("choice", "right")})
    assert oink_fpim(game).even_strategy >= expected
    assert oink_fpjm(game).even_strategy >= expected
    assert len(oink_fpi(game).even_strategy & expected) == 1
    assert len(oink_fpj(game).even_strategy & expected) == 1
