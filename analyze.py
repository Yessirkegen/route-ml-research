#!/usr/bin/env python3
"""CLI entry-point: analyse routing experiment results."""

import sys

from route_risk.analysis import load_results, summarize_best, plot_pareto


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py results.csv")
        sys.exit(1)

    df = load_results(sys.argv[1])

    best = summarize_best(df, eta_limit_min=2.0)
    print("\n=== Best delta per scenario (ΔETA <= +2 min) ===")
    print(best.to_string(index=False))

    best.to_csv("summary_best.tsv", sep="\t", index=False)
    print("Saved summary_best.tsv")

    plot_pareto(df, outdir="figs")


if __name__ == "__main__":
    main()
