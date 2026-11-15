# SYNTCOMP 2020 pgame benchmark snapshot used by the FPJ paper

This directory contains the 119 benchmark game files from the authors' tagged `v1.1` release. The files have been copied byte-for-byte from `experiments/syntcomp/` in [olijzenga/bdd-parity-game-solver](https://github.com/olijzenga/bdd-parity-game-solver/tree/v1.1/experiments/syntcomp). The source commit is `22195bfb303986c872fce7ee710d0fe6284ab9a3`; `manifest.json` records each filename, benchmark group, byte size, and SHA-256 digest. The upstream `LICENSE` is included.

The paper says its evaluated automata came from the SYNTCOMP 2020 pgame track and were converted by Knor into explicit parity games. These preserved `.oink` files are the exact explicit game inputs consumed by the authors' v1.1 evaluation script. Their contents use PGSolver's `parity N;` text syntax. The `.oink` suffix is the authors' file naming convention; no normalization or conversion was applied to these source copies.

## Benchmark groups

The prefix mapping below follows `scripts/empirical_evaluation.py` in the tagged source repository, matching the paper's reported counts:

| Paper group | Files | File prefixes |
| --- | ---: | --- |
| Lily | 23 | `lilydemo` |
| AMBA | 8 | `amba_decomposed` |
| ltl2dba | 58 | `ltl2dba`, `ltl2dpa` |
| Arbiters | 15 | `full_arbiter`, `prioritized_arbiter`, `round_robin_arbiter`, `simple_arbiter` |
| Detector | 2 | `detector` |
| Buffer | 2 | `starve` |
| Load balancer | 11 | `load` |

## Use from the project

```python
from parity_bdd_lab import load_benchmark_dataset

lily = load_benchmark_dataset("Lily")  # filename -> ParityGame
```

The original `.oink` files remain in this directory. `read_pg()` and `load_benchmark_dataset("Lily")` load them directly. `.oink` and `.pg` use the same PGSolver text grammar, so no format conversion is needed.
