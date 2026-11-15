"""Utilities for experimenting with parity games and decision diagrams."""

from .game import (
    ParityGame,
    StrategyProfile,
    enumerate_strategy_profiles,
    play,
    read_pg,
    strategy_profile_count,
    write_pg,
)
from .benchmark import (
    BenchmarkRecord,
    BenchmarkProgress,
    DEFAULT_METHODS,
    benchmark_all_datasets,
    benchmark_dataset,
    benchmark_game,
    benchmark_selection,
    records_to_dataframe,
    write_benchmark_csv,
)
from .datasets import (
    benchmark_dataset_names,
    iter_benchmark_files,
    load_benchmark_dataset,
)
from .profiling import ProfileArtifact, profile_dataset, profile_solver
from .cycles import (
    ForcedParityCycle,
    ParityCycle,
    enumerate_forced_winning_cycles,
    enumerate_unique_cycles,
)
from .decision_diagrams import collect_bdd_garbage, get_bdd_class, new_bdd, new_mdd
from .dfi import DFIResult, dfi, dfi_ns
from .fpj import FPJResult, fpj
from .multistrategy import CycleKernel, ForcedMultiStrategyResult, PriorityChain, chain_multistrategy
from .visualize import draw_game, draw_pg
from .zielonka import ZielonkaResult, zielonka

__all__ = [
    "FPJResult",
    "ForcedMultiStrategyResult",
    "CycleKernel",
    "PriorityChain",
    "DFIResult",
    "BenchmarkRecord",
    "BenchmarkProgress",
    "DEFAULT_METHODS",
    "ProfileArtifact",
    "profile_dataset",
    "profile_solver",
    "ParityGame",
    "ParityCycle",
    "ForcedParityCycle",
    "enumerate_forced_winning_cycles",
    "enumerate_unique_cycles",
    "StrategyProfile",
    "enumerate_strategy_profiles",
    "draw_game",
    "fpj",
    "chain_multistrategy",
    "dfi",
    "dfi_ns",
    "zielonka",
    "ZielonkaResult",
    "benchmark_game",
    "benchmark_dataset",
    "benchmark_all_datasets",
    "benchmark_selection",
    "records_to_dataframe",
    "write_benchmark_csv",
    "draw_pg",
    "benchmark_dataset_names",
    "iter_benchmark_files",
    "load_benchmark_dataset",
    "new_bdd",
    "get_bdd_class",
    "collect_bdd_garbage",
    "new_mdd",
    "play",
    "read_pg",
    "strategy_profile_count",
    "write_pg",
]
