#!/usr/bin/env python3
"""Profile parity-game solvers and generate standalone SVG flamegraphs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from parity_bdd_lab import DEFAULT_METHODS, profile_dataset, profile_solver, read_pg
from parity_bdd_lab.profiling import find_benchmark_game


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game", action="append", metavar="PATH_OR_NAME",
        help="game file path, or a unique benchmark filename/prefix; may be repeated",
    )
    parser.add_argument(
        "--dataset", action="append", metavar="NAME",
        help="benchmark dataset name, for example AMBA or Lily; may be repeated",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(DEFAULT_METHODS),
        default=list(DEFAULT_METHODS), help="solver methods to profile (default: all)",
    )
    parser.add_argument(
        "--bdd-backend", choices=("auto", "python", "cudd"),
        default=os.environ.get("PARITY_BDD_BACKEND", "auto"),
        help="BDD backend for DFI/FPJM (default: auto; use cudd to require native CUDD)",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "profiles",
        help="directory for .prof, .txt, .json, and .svg outputs",
    )
    parser.add_argument(
        "--benchmark-directory", type=Path,
        help="override the directory used to find benchmark datasets",
    )
    parser.add_argument(
        "--memory", action="store_true",
        help="also record tracemalloc's peak Python allocation during each run",
    )
    args = parser.parse_args()

    if not args.dataset and not args.game:
        parser.error("provide at least one --game or --dataset")
    os.environ["PARITY_BDD_BACKEND"] = args.bdd_backend

    all_artifacts = []
    for dataset in args.dataset or ():
        all_artifacts.extend(profile_dataset(
            dataset,
            args.output,
            methods=args.methods,
            directory=args.benchmark_directory,
            measure_memory=args.memory,
        ))

    if args.game:
        total = len(args.game) * len(args.methods)
        for game_number, game_spec in enumerate(args.game, 1):
            game_path = Path(game_spec)
            if game_path.is_file():
                game = read_pg(game_path)
                dataset = "custom"
                game_name = game_path.name
            else:
                dataset, game_path = find_benchmark_game(
                    game_spec, directory=args.benchmark_directory,
                )
                game = read_pg(game_path)
                game_name = game_path.name
            for method_index, method in enumerate(args.methods, 1):
                completed = (game_number - 1) * len(args.methods) + method_index
                print(f"[{completed}/{total}] profiling {dataset} / {game_name} / {method}", flush=True)
                all_artifacts.append(profile_solver(
                    game, method, args.output,
                    dataset=dataset, game_name=game_name,
                    measure_memory=args.memory,
                ))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    previous = []
    if manifest.exists():
        try:
            previous = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = []
    by_profile = {entry.get("profile_file"): entry for entry in previous}
    by_profile.update({artifact.profile_file: asdict(artifact) for artifact in all_artifacts})
    manifest.write_text(
        json.dumps(list(by_profile.values()), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(all_artifacts)} profile sets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
