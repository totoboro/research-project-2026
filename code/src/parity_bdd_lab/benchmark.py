"""Reusable benchmark runners for parity-game solvers and benchmark datasets."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import multiprocessing
import pickle
import sys
import time
import tracemalloc
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Hashable, Iterable, Mapping

from .datasets import benchmark_dataset_names, iter_benchmark_files
from .dfi import dfi, dfi_ns
from .fpjm import fpjm
from .game import ParityGame, read_pg
from .oink_fpi import oink_fpi
from .oink_fpim import oink_fpim
from .oink_fpj import oink_fpj
from .oink_fpjm import oink_fpjm
from .multistrategy import chain_multistrategy
from .cq_multistrategy import chain_quotient_multistrategy
from .cq_cached_multistrategy import cached_chain_quotient_multistrategy
from .q_cached_multistrategy import q_cached_multistrategy
from .zielonka import zielonka

Solver = Callable[[ParityGame], object]
DEFAULT_METHODS: dict[str, Solver] = {
    "DFI": dfi,
    "DFI-ns": dfi_ns,
    "FPJM": fpjm,
    "Oink-FPI": oink_fpi,
    "Oink-FPIM": oink_fpim,
    "Oink-FPJ": oink_fpj,
    "Oink-FPJM": oink_fpjm,
    "Chain-MS": chain_multistrategy,
    "Zielonka": zielonka,
    "CQ-MS": chain_quotient_multistrategy,
    "CQ-MS-Cached": cached_chain_quotient_multistrategy,
    "Q-MS-Cached": q_cached_multistrategy,
}


@dataclass(frozen=True)
class BenchmarkRecord:
    dataset: str
    game: str
    method: str
    run: int
    runtime_seconds: float
    peak_traced_memory_bytes: int | None
    even_winning_vertices: int | None
    odd_winning_vertices: int | None
    solution_fingerprint: str | None
    strategy_edges: int | None
    iterations: int | None
    bdd_nodes: int | None


@dataclass(frozen=True)
class BenchmarkProgress:
    """Progress notification emitted after each solver run."""

    dataset: str
    game: str
    method: str
    run: int
    runs_per_method: int
    game_number: int
    total_games: int
    completed_runs: int
    total_runs: int


ProgressCallback = Callable[[BenchmarkProgress], None]


def _report_progress(progress: ProgressCallback | bool | None, event: BenchmarkProgress) -> None:
    if progress is True:
        print(
            f"[{event.completed_runs}/{event.total_runs} runs; "
            f"game {event.game_number}/{event.total_games}] "
            f"{event.dataset} / {event.game} / {event.method} "
            f"run {event.run}/{event.runs_per_method} complete",
            flush=True,
        )
    elif progress:
        progress(event)


def _normalize_methods(methods: Mapping[str, Solver] | Iterable[str] | None) -> dict[str, Solver]:
    if methods is None:
        return dict(DEFAULT_METHODS)
    if isinstance(methods, Mapping):
        if not methods:
            raise ValueError("at least one benchmark method is required")
        return dict(methods)
    selected = {}
    for name in methods:
        if name not in DEFAULT_METHODS:
            raise ValueError(f"unknown built-in method {name!r}; available: {', '.join(DEFAULT_METHODS)}")
        selected[name] = DEFAULT_METHODS[name]
    if not selected:
        raise ValueError("at least one benchmark method is required")
    return selected


def _stable_vertices(values) -> tuple[Hashable, ...] | None:
    if values is None:
        return None
    return tuple(sorted(values, key=lambda value: (type(value).__name__, repr(value))))


def _solution_fingerprint(even_wins, odd_wins) -> str | None:
    if even_wins is None or odd_wins is None:
        return None
    payload = {
        "even": [(type(v).__name__, repr(v)) for v in _stable_vertices(even_wins)],
        "odd": [(type(v).__name__, repr(v)) for v in _stable_vertices(odd_wins)],
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _validate_workers(workers: int) -> None:
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")


def _validate_picklable_methods(methods: Mapping[str, Solver]) -> None:
    for name, solver in methods.items():
        try:
            pickle.dumps(solver)
        except Exception as error:
            raise TypeError(
                f"solver {name!r} cannot be sent to worker processes; use a picklable "
                "top-level function or set workers=1"
            ) from error


def _execute_task(task):
    """Run one solver repetition in a worker process and return compact data."""
    (dataset, game_name, source, method_name, solver, run, measure_memory,
     game_number, repeats, total_games, task_index) = task
    game = read_pg(source) if isinstance(source, Path) else source
    was_tracing = tracemalloc.is_tracing()
    if measure_memory and not was_tracing:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        result = solver(game)
    finally:
        elapsed = time.perf_counter() - started
        peak = tracemalloc.get_traced_memory()[1] if measure_memory and not was_tracing else None
        if measure_memory and not was_tracing:
            tracemalloc.stop()

    even_wins = getattr(result, "even_wins", None)
    odd_wins = getattr(result, "odd_wins", None)
    even_strategy = getattr(result, "even_strategy", None)
    odd_strategy = getattr(result, "odd_strategy", None)
    record = BenchmarkRecord(
        dataset=dataset,
        game=game_name,
        method=method_name,
        run=run,
        runtime_seconds=elapsed,
        peak_traced_memory_bytes=peak,
        even_winning_vertices=len(even_wins) if even_wins is not None else None,
        odd_winning_vertices=len(odd_wins) if odd_wins is not None else None,
        solution_fingerprint=_solution_fingerprint(even_wins, odd_wins),
        strategy_edges=(len(even_strategy) + len(odd_strategy)
                        if even_strategy is not None and odd_strategy is not None else None),
        iterations=getattr(result, "iterations", None),
        bdd_nodes=getattr(result, "bdd_nodes", None),
    )
    del result, game
    gc.collect()
    return task_index, record, (dataset, game_name, method_name, run, repeats, game_number, total_games)


def _run_parallel(tasks, *, workers: int, total_runs: int, total_games: int,
                  progress: ProgressCallback | bool | None) -> list[BenchmarkRecord]:
    """Execute a bounded number of process tasks to cap queued game memory."""
    pending = {}
    completed_records = []
    completed = 0
    task_iter = iter(tasks)
    max_pending = max(workers, workers * 2)
    # VS Code/Jupyter is an interactive parent process on the main supported
    # macOS setup. Fork makes module-level worker functions usable there; on
    # other platforms use spawn and recycle workers periodically.
    if sys.platform == "darwin":
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        )
    else:
        pool = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=25)
    with pool:
        exhausted = False
        while pending or not exhausted:
            while len(pending) < max_pending and not exhausted:
                try:
                    task = next(task_iter)
                except StopIteration:
                    exhausted = True
                    break
                future = pool.submit(_execute_task, task)
                pending[future] = task
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                task_index, record, metadata = future.result()
                completed_records.append((task_index, record))
                completed += 1
                dataset, game_name, method_name, run, repeats, game_number, _ = metadata
                _report_progress(progress, BenchmarkProgress(
                    dataset, game_name, method_name, run, repeats,
                    game_number, total_games, completed, total_runs,
                ))
    completed_records.sort(key=lambda pair: pair[0])
    return [record for _, record in completed_records]


def _parallel_tasks(game_sources, methods, repeats, measure_memory):
    index = 0
    for dataset, game_name, source, game_number, total_games in game_sources:
        for method_name, solver in methods.items():
            for run in range(1, repeats + 1):
                yield (dataset, game_name, source, method_name, solver, run,
                       measure_memory, game_number, repeats, total_games, index)
                index += 1


def _run_one(dataset: str, game_name: str, game: ParityGame, methods: Mapping[str, Solver], repeats: int, measure_memory: bool,
             *, progress: ProgressCallback | bool | None = None, game_number: int = 1,
             total_games: int = 1, completed_runs: int = 0, total_runs: int | None = None):
    records = []
    total_runs = total_runs if total_runs is not None else len(methods) * repeats
    for method_name, solver in methods.items():
        for run in range(1, repeats + 1):
            was_tracing = tracemalloc.is_tracing()
            if measure_memory and not was_tracing:
                tracemalloc.start()
            started = time.perf_counter()
            try:
                result = solver(game)
            finally:
                elapsed = time.perf_counter() - started
                peak = tracemalloc.get_traced_memory()[1] if measure_memory and not was_tracing else None
                if measure_memory and not was_tracing:
                    tracemalloc.stop()

            even_wins = getattr(result, "even_wins", None)
            odd_wins = getattr(result, "odd_wins", None)
            even_strategy = getattr(result, "even_strategy", None)
            odd_strategy = getattr(result, "odd_strategy", None)
            strategies_size = (
                len(even_strategy) + len(odd_strategy)
                if even_strategy is not None and odd_strategy is not None else None
            )
            records.append(BenchmarkRecord(
                dataset=dataset,
                game=game_name,
                method=method_name,
                run=run,
                runtime_seconds=elapsed,
                peak_traced_memory_bytes=peak,
                even_winning_vertices=len(even_wins) if even_wins is not None else None,
                odd_winning_vertices=len(odd_wins) if odd_wins is not None else None,
                solution_fingerprint=_solution_fingerprint(even_wins, odd_wins),
                strategy_edges=strategies_size,
                iterations=getattr(result, "iterations", None),
                bdd_nodes=getattr(result, "bdd_nodes", None),
            ))
            del result
            completed_runs += 1
            _report_progress(progress, BenchmarkProgress(
                dataset, game_name, method_name, run, repeats,
                game_number, total_games, completed_runs, total_runs,
            ))
            # BDD wrappers can leave unreachable nodes in manager tables.
            # Collect after each solve so long runs release them promptly.
            gc.collect()
    return records


def benchmark_game(
    game: ParityGame,
    *,
    game_name: str | None = None,
    dataset: str = "custom",
    methods: Mapping[str, Solver] | Iterable[str] | None = None,
    repeats: int = 1,
    measure_memory: bool = False,
    progress: ProgressCallback | bool | None = None,
    workers: int = 4,
) -> list[BenchmarkRecord]:
    """Benchmark selected methods on one game using up to ``workers`` processes.

    Use ``workers=1`` for sequential execution. Parallel workers require
    picklable solver callables; memory tracing is off by default.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    _validate_workers(workers)
    selected = _normalize_methods(methods)
    if workers > 1:
        _validate_picklable_methods(selected)
        total_runs = len(selected) * repeats
        tasks = _parallel_tasks(
            ((dataset, game_name or game.name, game, 1, 1),),
            selected, repeats, measure_memory,
        )
        return _run_parallel(tasks, workers=workers, total_runs=total_runs,
                             total_games=1, progress=progress)
    return _run_one(dataset, game_name or game.name, game, selected, repeats, measure_memory,
                    progress=progress)


def benchmark_dataset(
    dataset: str,
    *,
    methods: Mapping[str, Solver] | Iterable[str] | None = None,
    repeats: int = 1,
    measure_memory: bool = False,
    directory: str | Path | None = None,
    progress: ProgressCallback | bool | None = True,
    workers: int = 4,
) -> list[BenchmarkRecord]:
    """Benchmark a dataset, reporting completed runs and using up to four workers by default."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    _validate_workers(workers)
    selected_methods = _normalize_methods(methods)
    paths = tuple(iter_benchmark_files(dataset, directory=directory) if directory is not None else iter_benchmark_files(dataset))
    if not paths:
        raise FileNotFoundError(f"no benchmark games found for {dataset!r}")
    if workers > 1:
        _validate_picklable_methods(selected_methods)
        tasks = _parallel_tasks(
            ((dataset, path.name, path, index, len(paths)) for index, path in enumerate(paths, 1)),
            selected_methods, repeats, measure_memory,
        )
        total_runs = len(paths) * len(selected_methods) * repeats
        return _run_parallel(tasks, workers=workers, total_runs=total_runs,
                             total_games=len(paths), progress=progress)
    records = []
    completed_runs = 0
    total_runs = len(paths) * len(selected_methods) * repeats
    for game_index, path in enumerate(paths, 1):
        game = read_pg(path)
        game_records = _run_one(
            dataset, path.name, game, selected_methods, repeats, measure_memory,
            progress=progress, game_number=game_index, total_games=len(paths),
            completed_runs=completed_runs, total_runs=total_runs,
        )
        records.extend(game_records)
        completed_runs += len(game_records)
        del game_records, game
        gc.collect()
    return records


def benchmark_all_datasets(
    *,
    methods: Mapping[str, Solver] | Iterable[str] | None = None,
    repeats: int = 1,
    measure_memory: bool = False,
    directory: str | Path | None = None,
    datasets: str | Iterable[str] | None = None,
    progress: ProgressCallback | bool | None = True,
    workers: int = 4,
) -> list[BenchmarkRecord]:
    """Benchmark selected paper datasets, reporting completed runs across workers.

    Set ``progress=False`` to silence output or pass a callback receiving
    :class:`BenchmarkProgress` events. Games and solver results are released
    between games; only the compact benchmark records are retained. Set
    ``workers=1`` to run sequentially. Parallel solver callables must pickle.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    _validate_workers(workers)
    available = benchmark_dataset_names()
    if datasets is None:
        datasets = available
    elif isinstance(datasets, str):
        datasets = (datasets,)
    canonical = {name.casefold(): name for name in available}
    selected_datasets = []
    for requested in datasets:
        name = canonical.get(requested.casefold())
        if name is None:
            raise ValueError(f"unknown benchmark dataset {requested!r}; choose from {', '.join(available)}")
        if name not in selected_datasets:
            selected_datasets.append(name)
    datasets = tuple(selected_datasets)
    if not datasets:
        raise ValueError("at least one benchmark dataset is required")
    files_by_dataset = {
        dataset: tuple(iter_benchmark_files(dataset, directory=directory) if directory is not None
                       else iter_benchmark_files(dataset))
        for dataset in datasets
    }
    total_games = sum(map(len, files_by_dataset.values()))
    if total_games == 0:
        raise FileNotFoundError("no benchmark games found")
    selected_methods = _normalize_methods(methods)
    if workers > 1:
        _validate_picklable_methods(selected_methods)
        game_sources = (
            (dataset, path.name, path, game_number, total_games)
            for game_number, (dataset, path) in enumerate(
                ((dataset, path) for dataset, paths in files_by_dataset.items() for path in paths), 1
            )
        )
        tasks = _parallel_tasks(game_sources, selected_methods, repeats, measure_memory)
        total_runs = total_games * len(selected_methods) * repeats
        return _run_parallel(tasks, workers=workers, total_runs=total_runs,
                             total_games=total_games, progress=progress)
    total_runs = total_games * len(selected_methods) * repeats
    records = []
    completed_games = 0
    completed_runs = 0
    for dataset, paths in files_by_dataset.items():
        for path in paths:
            game = read_pg(path)
            game_records = _run_one(
                dataset, path.name, game, selected_methods, repeats, measure_memory,
                progress=progress, game_number=completed_games + 1,
                total_games=total_games, completed_runs=completed_runs,
                total_runs=total_runs,
            )
            records.extend(game_records)
            completed_runs += len(game_records)
            completed_games += 1
            del game_records, game
            gc.collect()
    return records


def benchmark_selection(
    datasets: str | Iterable[str] | None = None,
    *,
    methods: Mapping[str, Solver] | Iterable[str] | None = None,
    repeats: int = 1,
    measure_memory: bool = False,
    directory: str | Path | None = None,
    progress: ProgressCallback | bool | None = True,
    workers: int = 4,
) -> list[BenchmarkRecord]:
    """Benchmark any chosen method and dataset subset with the shared runner.

    ``datasets=None`` selects all paper datasets; a string selects one dataset,
    and an iterable selects several. ``methods`` accepts built-in method names
    or a mapping of display names to top-level solver callables.
    """
    return benchmark_all_datasets(
        datasets=datasets,
        methods=methods,
        repeats=repeats,
        measure_memory=measure_memory,
        directory=directory,
        progress=progress,
        workers=workers,
    )


def records_to_dataframe(records: Iterable[BenchmarkRecord]):
    """Convert benchmark records to a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(asdict(record) for record in records)


def write_benchmark_csv(records: Iterable[BenchmarkRecord], path: str | Path) -> Path:
    """Write records to CSV using only the Python standard library."""
    records = tuple(records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkRecord.__dataclass_fields__)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return target
