"""CPU flamegraph profiling helpers for parity-game solver runs."""

from __future__ import annotations

import cProfile
import gc
import html
import json
import pstats
import re
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .benchmark import DEFAULT_METHODS
from .datasets import benchmark_dataset_names, iter_benchmark_files
from .game import ParityGame, read_pg


@dataclass(frozen=True)
class ProfileArtifact:
    dataset: str
    game: str
    method: str
    bdd_backend: str | None
    runtime_seconds: float
    peak_traced_memory_bytes: int | None
    profile_file: str
    flamegraph_file: str
    stats_file: str


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "unnamed"


def _function_label(function_key) -> str:
    filename, line, function = function_key
    if filename == "~":
        return function
    return f"{Path(filename).name}:{line} {function}"


def _folded_stacks(stats: pstats.Stats) -> dict[tuple[str, ...], float]:
    """Convert cProfile's caller graph to weighted stacks for a flamegraph.

    cProfile aggregates identical functions across call sites. Caller edge
    counts are used to distribute each function's self time through its call
    contexts, producing a useful folded-stack approximation without external
    profiling packages.
    """
    entries = stats.stats
    callers_by_function = {function: entry[4] for function, entry in entries.items()}
    graph = {function: [] for function in entries}
    edge_counts = {}
    for function, callers in callers_by_function.items():
        for caller, call_data in callers.items():
            if caller not in entries or caller[0] == "~" and "Profiler" in caller[2]:
                continue
            calls = call_data[1] if isinstance(call_data, tuple) else 1
            calls = max(int(calls), 1)
            graph[caller].append(function)
            edge_counts[(caller, function)] = calls

    # Collapse recursion by removing edges internal to strongly connected
    # components. cProfile aggregates recursive calls at function level; keeping
    # those edges would require infinitely expanding recursive stack paths.
    index = 0
    indices = {}
    lowlink = {}
    stack = []
    on_stack = set()
    component = {}
    component_count = 0

    def visit(vertex):
        nonlocal index, component_count
        indices[vertex] = lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)
        for child in graph[vertex]:
            if child not in indices:
                visit(child)
                lowlink[vertex] = min(lowlink[vertex], lowlink[child])
            elif child in on_stack:
                lowlink[vertex] = min(lowlink[vertex], indices[child])
        if lowlink[vertex] == indices[vertex]:
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component[member] = component_count
                if member == vertex:
                    break
            component_count += 1

    for function in graph:
        if function not in indices:
            visit(function)

    total_incoming = {function: 0 for function in entries}
    children_by_caller = defaultdict(list)
    for (caller, function), calls in edge_counts.items():
        if component[caller] == component[function]:
            continue
        total_incoming[function] += calls
        children_by_caller[caller].append((function, calls))

    roots = [
        function for function in entries
        if total_incoming[function] == 0
        and not (function[0] == "~" and "Profiler" in function[2])
    ]
    folded: dict[tuple[str, ...], float] = defaultdict(float)

    def visit(function, stack: tuple[str, ...], share: float, ancestors: frozenset, depth: int):
        primitive_calls, calls, own_seconds, cumulative_seconds, _ = entries[function]
        del primitive_calls, calls, cumulative_seconds
        label = _function_label(function)
        current = stack + (label,)
        if own_seconds > 0:
            folded[current] += own_seconds * share
        if depth >= 80:
            return
        descendants = children_by_caller.get(function, ())
        for child, edge_calls in descendants:
            denominator = total_incoming.get(child, 0)
            if denominator:
                visit(child, current, share * edge_calls / denominator,
                      ancestors | {function}, depth + 1)

    for root in roots:
        visit(root, (), 1.0, frozenset(), 0)
    return dict(folded)


def _render_flamegraph(folded: dict[tuple[str, ...], float], target: Path, title: str) -> None:
    """Render folded cProfile stacks as a standalone, hoverable SVG flamegraph."""
    tree = {"label": "root", "self": 0.0, "children": {}}
    for stack, weight in folded.items():
        if weight <= 0:
            continue
        node = tree
        for label in stack:
            node = node["children"].setdefault(
                label, {"label": label, "self": 0.0, "children": {}}
            )
        node["self"] += weight

    def total(node):
        node["total"] = node["self"] + sum(total(child) for child in node["children"].values())
        return node["total"]

    total(tree)
    width = 1500
    bar_height = 22
    depth = max((len(stack) for stack in folded), default=1)
    height = max(100, (depth + 2) * bar_height + 45)
    total_weight = tree["total"] or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:ui-monospace,monospace;font-size:11px} .frame:hover rect{stroke:#111;stroke-width:1.5}</style>',
        f'<text x="10" y="20" font-size="15">{html.escape(title)} — self time (cProfile)</text>',
        f'<text x="10" y="{height - 8}" fill="#555">Total profiled self time: {total_weight:.3f} s; hover a frame for details</text>',
    ]

    palette = ("#f6bd60", "#f28482", "#84a59d", "#f7ede2", "#a8dadc", "#cdb4db", "#ffafcc")

    def draw(node, x, y, allotted_width):
        children = sorted(node["children"].values(), key=lambda child: child["total"], reverse=True)
        cursor = x
        for child in children:
            child_width = allotted_width * child["total"] / node["total"] if node["total"] else 0
            if child_width >= 0.7:
                color = palette[sum(child["label"].encode("utf-8")) % len(palette)]
                label = html.escape(child["label"])
                tooltip = html.escape(f"{child['label']} — {child['total']:.6f}s cumulative")
                parts.append(
                    f'<g class="frame"><title>{tooltip}</title>'
                    f'<rect x="{cursor:.2f}" y="{y}" width="{max(0, child_width - 0.7):.2f}" '
                    f'height="{bar_height - 1}" rx="2" fill="{color}"/>'
                )
                if child_width > 45:
                    chars = max(1, int((child_width - 8) / 6.4))
                    shown = label if len(child["label"]) <= chars else html.escape(child["label"][:chars - 1]) + "…"
                    parts.append(f'<text x="{cursor + 4:.2f}" y="{y + 15}">{shown}</text>')
                parts.append('</g>')
                draw(child, cursor, y - bar_height, child_width)
            cursor += child_width

    draw(tree, 0, height - bar_height - 25, width)
    parts.append('</svg>')
    target.write_text("\n".join(parts), encoding="utf-8")


def profile_solver(
    game: ParityGame,
    method: str,
    output_directory: str | Path,
    *,
    dataset: str = "custom",
    game_name: str | None = None,
    measure_memory: bool = False,
) -> ProfileArtifact:
    """Profile one solver run and write cProfile, text stats, and SVG flamegraph."""
    if method not in DEFAULT_METHODS:
        raise ValueError(f"unknown method {method!r}; available: {', '.join(DEFAULT_METHODS)}")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    resolved_game_name = game_name or game.name
    stem = "__".join(map(_safe_name, (dataset, resolved_game_name, method)))
    profile_path = output / f"{stem}.prof"
    flamegraph_path = output / f"{stem}.svg"
    stats_path = output / f"{stem}.txt"

    # Import the BDD backend before starting the profiler so its one-time
    # dependency imports do not dominate the first solver's flamegraph.
    from .decision_diagrams import get_bdd_class
    get_bdd_class()

    profiler = cProfile.Profile()
    was_tracing = tracemalloc.is_tracing()
    if measure_memory and not was_tracing:
        tracemalloc.start()
    elif measure_memory:
        tracemalloc.reset_peak()
    try:
        result = profiler.runcall(DEFAULT_METHODS[method], game)
    finally:
        peak_memory = tracemalloc.get_traced_memory()[1] if measure_memory else None
        if measure_memory and not was_tracing:
            tracemalloc.stop()
    del result
    gc.collect()

    profiler.dump_stats(profile_path)
    stats = pstats.Stats(profiler)
    with stats_path.open("w", encoding="utf-8") as stream:
        stats.stream = stream
        stats.sort_stats("cumulative").print_stats()
        stream.write("\n--- sorted by self time ---\n")
        stats.sort_stats("tottime").print_stats()
    _render_flamegraph(
        _folded_stacks(stats), flamegraph_path,
        f"{dataset} / {resolved_game_name} / {method}",
    )
    runtime = stats.total_tt
    artifact = ProfileArtifact(
        dataset=dataset,
        game=resolved_game_name,
        method=method,
        bdd_backend=(get_bdd_class().__module__ if method in {"DFI", "DFI-ns", "FPJM"} else None),
        runtime_seconds=runtime,
        peak_traced_memory_bytes=peak_memory,
        profile_file=str(profile_path),
        flamegraph_file=str(flamegraph_path),
        stats_file=str(stats_path),
    )
    (output / f"{stem}.json").write_text(
        json.dumps(asdict(artifact), indent=2) + "\n", encoding="utf-8"
    )
    return artifact


def profile_dataset(
    dataset: str,
    output_directory: str | Path,
    *,
    methods: Iterable[str] = ("DFI", "DFI-ns", "FPJM", "Zielonka"),
    directory: str | Path | None = None,
    measure_memory: bool = False,
    progress: bool = True,
) -> list[ProfileArtifact]:
    """Profile every game/method pair in a named benchmark dataset sequentially."""
    files = tuple(
        iter_benchmark_files(dataset, directory=directory)
        if directory is not None else iter_benchmark_files(dataset)
    )
    if not files:
        raise FileNotFoundError(f"no benchmark games found for {dataset!r}")
    selected_methods = tuple(methods)
    artifacts = []
    total = len(files) * len(selected_methods)
    for game_index, game_path in enumerate(files, 1):
        game = read_pg(game_path)
        for method_index, method in enumerate(selected_methods, 1):
            if progress:
                completed = (game_index - 1) * len(selected_methods) + method_index
                print(f"[{completed}/{total}] profiling {dataset} / {game_path.name} / {method}", flush=True)
            artifacts.append(profile_solver(
                game, method, output_directory, dataset=dataset,
                game_name=game_path.name, measure_memory=measure_memory,
            ))
        del game
        gc.collect()
    return artifacts


def find_benchmark_game(name: str, *, directory: str | Path | None = None) -> tuple[str, Path]:
    """Find a benchmark game by full filename or unique filename stem prefix."""
    files = tuple(
        iter_benchmark_files(directory=directory)
        if directory is not None else iter_benchmark_files()
    )
    exact = [path for path in files if path.name == name]
    matches = exact or [path for path in files if path.name.startswith(name)]
    if len(matches) != 1:
        if not matches:
            raise FileNotFoundError(f"no benchmark game matches {name!r}")
        raise ValueError(f"{name!r} is ambiguous; matches: {', '.join(path.name for path in matches)}")
    path = matches[0]
    dataset = next(
        (group for group in benchmark_dataset_names()
         if path in (iter_benchmark_files(group, directory=directory)
                     if directory is not None else iter_benchmark_files(group))),
        "custom",
    )
    return dataset, path
