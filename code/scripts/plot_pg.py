#!/usr/bin/env python3
"""Render a PGSolver .pg file as a graph image."""

import argparse
from pathlib import Path

from parity_bdd_lab import draw_pg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input PGSolver .pg file")
    parser.add_argument("-o", "--output", type=Path, help="image output path (default: beside input, .png)")
    args = parser.parse_args()

    output = args.output or args.input.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = draw_pg(args.input, show=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
