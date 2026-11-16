"""Access and convert the parity-game benchmark files used by symbolic FPJM."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .game import ParityGame, read_pg

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "syntcomp2020-pgame"

# Prefix mapping from the authors' v1.1 empirical_evaluation.py. The two LTL
# automaton families were reported together as the ltl2dba benchmark set.
BENCHMARK_PREFIXES: dict[str, tuple[str, ...]] = {
    "Lily": ("lilydemo",),
    "AMBA": ("amba_decomposed",),
    "ltl2dba": ("ltl2dba", "ltl2dpa"),
    "Arbiters": ("full_arbiter", "prioritized_arbiter", "round_robin_arbiter", "simple_arbiter"),
    "Detector": ("detector",),
    "Buffer": ("starve",),
    "Load balancer": ("load",),
}


def benchmark_dataset_names() -> tuple[str, ...]:
    """Return the seven benchmark group names reported in the paper."""
    return tuple(BENCHMARK_PREFIXES)


def iter_benchmark_files(dataset: str | None = None, *, directory: str | Path = BENCHMARK_DIR) -> Iterator[Path]:
    """Yield preserved ``.oink`` files, optionally filtered to one paper group."""
    root = Path(directory)
    if dataset is None:
        yield from sorted(root.glob("*.oink"))
        return
    normalized = dataset.casefold()
    key = next((name for name in BENCHMARK_PREFIXES if name.casefold() == normalized), None)
    if key is None:
        valid = ", ".join(BENCHMARK_PREFIXES)
        raise ValueError(f"unknown benchmark set {dataset!r}; choose one of: {valid}")
    prefixes = BENCHMARK_PREFIXES[key]
    yield from (path for path in sorted(root.glob("*.oink")) if path.name.startswith(prefixes))


def load_benchmark_dataset(dataset: str, *, directory: str | Path = BENCHMARK_DIR) -> dict[str, ParityGame]:
    """Load one paper benchmark group as ``{filename: ParityGame}``."""
    files = tuple(iter_benchmark_files(dataset, directory=directory))
    if not files:
        raise FileNotFoundError(f"no .oink games found for {dataset!r} in {directory}")
    return {path.name: read_pg(path) for path in files}
