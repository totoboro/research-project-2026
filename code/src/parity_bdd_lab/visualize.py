"""Graph plotting helpers for parity games and PGSolver ``.pg`` files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from parity_bdd_lab.game import ParityGame


def draw_game(
    game: "ParityGame",
    *,
    positions: Mapping[object, tuple[float, float]] | None = None,
    vertex_names: Mapping[object, str] | None = None,
    layout: str = "spring",
    ax: "Axes | None" = None,
    title: str | None = None,
    show: bool = True,
):
    """Draw a parity game using NetworkX and Matplotlib.

    Player 0 vertices are blue circles; player 1 vertices are orange pentagons.
    Each vertex is labelled with its name/ID and priority. Returns ``(fig, ax)``.
    Provide ``positions`` for exact control, or choose ``spring``, ``circular``,
    ``shell``, or ``kamada_kawai`` for a deterministic automatic layout.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Graph drawing requires the optional dependencies. Install with "
            "`python -m pip install -e '.[notebooks]'`."
        ) from exc

    graph = nx.DiGraph()
    graph.add_nodes_from(game.vertices)
    for source in game.vertices:
        for target in game.successors[source]:
            graph.add_edge(source, target)

    if positions is None:
        layouts = {
            "spring": lambda: nx.spring_layout(graph, seed=7, k=1.4),
            "circular": lambda: nx.circular_layout(graph),
            "shell": lambda: nx.shell_layout(graph),
            "kamada_kawai": lambda: nx.kamada_kawai_layout(graph),
        }
        if layout not in layouts:
            raise ValueError(f"unknown layout {layout!r}; choose one of {', '.join(layouts)}")
        pos = layouts[layout]()
    else:
        missing = set(game.vertices) - set(positions)
        if missing:
            raise ValueError(f"positions are missing vertices: {sorted(missing)}")
        pos = positions

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    else:
        fig = ax.figure

    node_colors = {0: "#4C78A8", 1: "#F2A541"}
    node_shapes = {0: "o", 1: "p"}
    labels = {
        v: f"{(vertex_names or {}).get(v, str(v))}\n{game.priority[v]}"
        for v in game.vertices
    }

    for owner in (0, 1):
        nodes = [v for v in game.vertices if game.owner[v] == owner]
        if nodes:
            nx.draw_networkx_nodes(
                graph, pos, nodelist=nodes, node_shape=node_shapes[owner],
                node_color=node_colors[owner], node_size=1500,
                edgecolors="#222222", linewidths=1.2, ax=ax,
            )
    nx.draw_networkx_labels(
        graph, pos, labels=labels, font_size=11, font_weight="bold", ax=ax
    )
    nx.draw_networkx_edges(
        graph, pos, arrows=True, arrowstyle="-|>", arrowsize=18,
        width=1.5, edge_color="#333333", connectionstyle="arc3,rad=0.08",
        min_source_margin=18, min_target_margin=20, ax=ax,
    )
    ax.set_title(title or game.name)
    ax.margins(0.18)
    ax.set_axis_off()
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker=node_shapes[player], color="w",
                       markerfacecolor=node_colors[player], markeredgecolor="#222222",
                       markersize=11, label=f"Player {player}")
            for player in (0, 1)
        ],
        loc="best", frameon=False,
    )
    if show:
        plt.show()
    return fig, ax


def draw_pg(
    path: str | Path,
    *,
    vertex_labels: Mapping[int, object] | None = None,
    **kwargs,
):
    """Read a PGSolver-format file and draw it, optionally mapping IDs to labels."""
    from .game import read_pg

    return draw_game(read_pg(path, vertex_labels=vertex_labels), **kwargs)
