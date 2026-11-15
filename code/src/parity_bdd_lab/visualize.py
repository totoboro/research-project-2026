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
    layout: str = "graphviz",
    ax: "Axes | None" = None,
    title: str | None = None,
    show: bool = True,
):
    """Draw a parity game using NetworkX and Matplotlib.

    Player 0 vertices are blue circles; player 1 vertices are orange pentagons.
    Each vertex is labelled with its name/ID and priority. Returns ``(fig, ax)``.
    Provide ``positions`` for exact control, or choose ``graphviz`` (the
    default), ``spring``, ``circular``, ``shell``, or ``kamada_kawai`` for a
    deterministic automatic layout. One-way edges are straight; reciprocal
    edges are curved away from one another so both directions remain visible.
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
        def graphviz_layout():
            # Relabel before passing the graph to pydot: distinct Python
            # objects can have identical string representations, which
            # Graphviz would otherwise collapse into one node.
            layout_graph = nx.convert_node_labels_to_integers(
                graph, label_attribute="vertex"
            )
            layout_graph.graph["graph"] = {
                "nodesep": "0.8",
                "ranksep": "1.0",
                "overlap": "false",
            }
            layout_graph.graph["node"] = {
                "fixedsize": "true",
                "height": "0.75",
                "shape": "circle",
                "width": "0.75",
            }
            try:
                layout_positions = nx.nx_pydot.pydot_layout(
                    layout_graph, prog="dot"
                )
            except Exception as exc:
                raise RuntimeError(
                    "the graphviz layout requires the `pydot` Python package "
                    "and a working Graphviz `dot` executable; install the "
                    "project's `notebooks` extra and Graphviz"
                ) from exc
            if layout_positions is None or set(layout_positions) != set(layout_graph):
                raise RuntimeError("Graphviz did not return a position for every vertex")
            return {
                data["vertex"]: layout_positions[node]
                for node, data in layout_graph.nodes(data=True)
            }

        layouts = {
            "graphviz": graphviz_layout,
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

    vertex_count = len(game.vertices)
    if ax is None:
        figure_scale = min(1.75, max(1.0, (vertex_count / 16) ** 0.5))
        fig, ax = plt.subplots(
            figsize=(9 * figure_scale, 6 * figure_scale), constrained_layout=True
        )
    else:
        fig = ax.figure

    node_colors = {0: "#4C78A8", 1: "#F2A541"}
    node_shapes = {0: "o", 1: "p"}
    node_size = min(1500, max(300, 4500 / vertex_count**0.5))
    font_size = min(11, max(7, 15 - vertex_count**0.5))
    labels = {
        v: f"{(vertex_names or {}).get(v, str(v))}\n{game.priority[v]}"
        for v in game.vertices
    }

    for owner in (0, 1):
        nodes = [v for v in game.vertices if game.owner[v] == owner]
        if nodes:
            nx.draw_networkx_nodes(
                graph, pos, nodelist=nodes, node_shape=node_shapes[owner],
                node_color=node_colors[owner], node_size=node_size,
                edgecolors="#222222", linewidths=1.2, ax=ax,
            )
    nx.draw_networkx_labels(
        graph, pos, labels=labels, font_size=font_size, font_weight="bold", ax=ax
    )
    self_loops = []
    reciprocal_edges = []
    one_way_edges = []
    for source, target in graph.edges:
        if source == target:
            self_loops.append((source, target))
        elif graph.has_edge(target, source):
            reciprocal_edges.append((source, target))
        else:
            one_way_edges.append((source, target))

    edge_style = {
        "arrows": True,
        "arrowstyle": "-|>",
        "arrowsize": 18,
        "width": 1.5,
        "edge_color": "#333333",
        "min_source_margin": 18,
        "min_target_margin": 20,
        "node_size": node_size,
        "ax": ax,
    }
    # NetworkX sizes self-loops from the vertical span of the edgelist in a
    # draw call. Include them with a non-loop group when possible so loops do
    # not collapse behind their nodes in a larger layout. NetworkX still uses
    # its dedicated self-loop path, regardless of that group's arc style.
    straight_edges = one_way_edges + (self_loops if one_way_edges else [])
    curved_edges = reciprocal_edges + (
        self_loops if not one_way_edges and reciprocal_edges else []
    )
    loops_only = self_loops if not one_way_edges and not reciprocal_edges else []
    if straight_edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=straight_edges,
            connectionstyle="arc3,rad=0.0", **edge_style,
        )
    if curved_edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=curved_edges,
            connectionstyle="arc3,rad=0.12", **edge_style,
        )
    if loops_only:
        nx.draw_networkx_edges(
            graph, pos, edgelist=loops_only,
            connectionstyle="arc3,rad=0.35", **edge_style,
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
