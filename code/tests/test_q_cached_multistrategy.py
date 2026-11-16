from __future__ import annotations

from pathlib import Path

import pytest

from parity_bdd_lab import (
    q_cached_multistrategy,
    read_pg,
    validate_multistrategy,
)
from parity_bdd_lab.q_cached_multistrategy import _vertex_transition_bdd


GAMES = Path(__file__).parents[1] / "data" / "games"


@pytest.mark.parametrize("path", sorted(GAMES.glob("*.pg")), ids=lambda path: path.name)
def test_q_cached_multistrategy_is_complete_on_examples(path: Path) -> None:
    game = read_pg(path)
    result = q_cached_multistrategy(game)
    report = validate_multistrategy(game, result)

    assert report.valid
    assert report.complete is True
    assert result.possible_vertex_ids == result.even_vertex_ids
    assert result.possible_vertex_ids == result.odd_vertex_ids


def test_vertex_transition_bdd_encodes_exact_edges() -> None:
    vertices = ("a", "b", "c")
    edges = {("a", "b"), ("a", "c"), ("c", "a")}
    bdd, relation, _ = _vertex_transition_bdd(vertices, edges)
    width = 2

    for source_id, source in enumerate(vertices):
        for target_id, target in enumerate(vertices):
            assignment = {
                **{
                    f"v{bit}": bool(source_id & (1 << bit))
                    for bit in range(width)
                },
                **{
                    f"v{bit}_next": bool(target_id & (1 << bit))
                    for bit in range(width)
                },
            }
            assert (bdd.let(assignment, relation) == bdd.true) is (
                (source, target) in edges
            )
