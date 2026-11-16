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
    CycleEnumerationLimitError,
    ForcedParityCycle,
    MultiStrategyLike,
    ParityCycle,
    enumerate_forced_winning_cycles,
    enumerate_multistrategy_cycles,
    enumerate_unique_cycles,
    multistrategy_subgame,
)
from .decision_diagrams import collect_bdd_garbage, get_bdd_class, new_bdd, new_mdd
from .dfi import DFIResult, dfi, dfi_ns
from .fpjm import FPJMResult, fpjm
from ._oink_common import OinkResult
from .oink_fpi import oink_fpi
from .oink_fpim import oink_fpim
from .oink_fpj import oink_fpj
from .oink_fpjm import oink_fpjm
from .multistrategy import CycleKernel, ForcedMultiStrategyResult, PriorityChain, chain_multistrategy
from .cq_multistrategy import (
    CQMultiStrategyStages,
    CQStrategySnapshot,
    chain_quotient_multistrategy,
    chain_quotient_multistrategy_stages,
    trace_chain_quotient_solver,
)
from .cq_cached_multistrategy import (
    IncrementalForcedWinCache,
    cached_chain_quotient_multistrategy,
    cached_chain_quotient_multistrategy_stages,
)
from .validation import (
    LassoWitness,
    MaximalityProgress,
    MultiStrategyResult,
    MultiStrategyValidationReport,
    PlayerValidationReport,
    VertexChoiceReport,
    validate_multistrategy,
)
from .visualize import draw_game, draw_pg
from .zielonka import ZielonkaResult, zielonka

__all__ = [
    "FPJMResult",
    "OinkResult",
    "ForcedMultiStrategyResult",
    "CycleKernel",
    "CycleEnumerationLimitError",
    "MultiStrategyLike",
    "PriorityChain",
    "CQMultiStrategyStages",
    "CQStrategySnapshot",
    "IncrementalForcedWinCache",
    "LassoWitness",
    "MaximalityProgress",
    "MultiStrategyResult",
    "MultiStrategyValidationReport",
    "PlayerValidationReport",
    "VertexChoiceReport",
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
    "enumerate_multistrategy_cycles",
    "enumerate_unique_cycles",
    "multistrategy_subgame",
    "StrategyProfile",
    "enumerate_strategy_profiles",
    "draw_game",
    "fpjm",
    "oink_fpi",
    "oink_fpim",
    "oink_fpj",
    "oink_fpjm",
    "chain_multistrategy",
    "chain_quotient_multistrategy",
    "chain_quotient_multistrategy_stages",
    "cached_chain_quotient_multistrategy",
    "cached_chain_quotient_multistrategy_stages",
    "trace_chain_quotient_solver",
    "validate_multistrategy",
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
