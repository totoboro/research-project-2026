# Benchmark datasets

The paper's SYNTCOMP 2020 pgame inputs used for its FPJ/DFI evaluation are preserved byte-for-byte in [`syntcomp2020-pgame/`](syntcomp2020-pgame/). That snapshot has all 119 `.oink` games, the upstream license, provenance, and a checksum manifest. See its README for the seven benchmark groups and their counts.

The files use PGSolver's `parity N;` text grammar despite their `.oink` suffix. The `.oink` suffix does not indicate a different grammar: these files use the same PGSolver text format as `.pg` files, so `read_pg()` and the benchmark loader read them directly. No conversion step is needed.
