from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from parity_bdd_lab.game import ParityGame
from parity_bdd_lab.visualize import draw_game


@pytest.fixture
def routing_game() -> ParityGame:
    return ParityGame(
        owner={"a": 0, "b": 1, "c": 0},
        priority={"a": 0, "b": 1, "c": 2},
        successors={"a": ("b",), "b": ("a", "c"), "c": ("c",)},
        name="routing",
    )


def test_draw_game_routes_each_edge_kind_separately(monkeypatch, routing_game):
    calls = []
    original = nx.draw_networkx_edges

    def record_edges(*args, **kwargs):
        calls.append((tuple(kwargs["edgelist"]), kwargs["connectionstyle"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(nx, "draw_networkx_edges", record_edges)
    fig, _ = draw_game(
        routing_game,
        positions={"a": (0, 1), "b": (1, 1), "c": (2, 0)},
        show=False,
    )
    try:
        assert calls == [
            ((("b", "c"), ("c", "c")), "arc3,rad=0.0"),
            ((("a", "b"), ("b", "a")), "arc3,rad=0.12"),
        ]
    finally:
        plt.close(fig)


def test_graphviz_is_the_default_layout(monkeypatch, routing_game):
    observed = {}

    def fake_layout(graph, *, prog):
        observed["prog"] = prog
        observed["graph"] = graph
        return {node: (index, index * 2) for index, node in enumerate(graph)}

    monkeypatch.setattr(nx.nx_pydot, "pydot_layout", fake_layout)
    fig, _ = draw_game(routing_game, show=False)
    try:
        assert observed["prog"] == "dot"
        assert observed["graph"].graph["graph"]["nodesep"] == "0.8"
        assert observed["graph"].graph["graph"]["ranksep"] == "1.0"
    finally:
        plt.close(fig)


def test_explicit_positions_bypass_graphviz(monkeypatch, routing_game):
    def fail_layout(*args, **kwargs):
        raise AssertionError("automatic layout should not be called")

    monkeypatch.setattr(nx.nx_pydot, "pydot_layout", fail_layout)
    fig, _ = draw_game(
        routing_game,
        positions={"a": (0, 1), "b": (1, 1), "c": (2, 0)},
        show=False,
    )
    plt.close(fig)


def test_graphviz_failure_has_actionable_message(monkeypatch, routing_game):
    def fail_layout(*args, **kwargs):
        raise FileNotFoundError("dot")

    monkeypatch.setattr(nx.nx_pydot, "pydot_layout", fail_layout)
    with pytest.raises(RuntimeError, match="pydot.*Graphviz.*dot"):
        draw_game(routing_game, show=False)
