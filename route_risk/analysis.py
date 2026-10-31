"""Post-hoc analysis of routing experiment results."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["deta_min"] = (df["eta_risk_s"] - df["eta_fast_s"]) / 60.0
    df["ddist_km"] = df["dist_risk_km"] - df["dist_fast_km"]
    df["risk_drop_abs"] = df["drisk"]
    df["risk_drop_pct"] = (df["drisk"] / df["risk_fast"].clip(lower=1e-9)) * 100.0
    df["scenario"] = df["depart"].astype(str) + " | " + df["roads"]
    return df


def summarize_best(df: pd.DataFrame, eta_limit_min: float = 2.0) -> pd.DataFrame:
    rows = []
    for (depart, roads), g in df.groupby(["depart", "roads"]):
        g_ok = g[g["deta_min"] <= eta_limit_min]
        if g_ok.empty:
            rows.append({"depart": depart, "roads": roads, "delta": None,
                         "deta_min": None, "risk_drop_abs": None,
                         "risk_drop_pct": None, "ddist_km": None})
            continue
        g_ok = g_ok.sort_values(["risk_drop_abs", "deta_min", "ddist_km"],
                                ascending=[False, True, True])
        top = g_ok.iloc[0]
        rows.append({
            "depart": depart, "roads": roads,
            "delta": int(top["delta"]),
            "deta_min": round(top["deta_min"], 2),
            "risk_drop_abs": round(top["risk_drop_abs"], 4),
            "risk_drop_pct": round(top["risk_drop_pct"], 1),
            "ddist_km": round(top["ddist_km"], 2),
        })
    return pd.DataFrame(rows).sort_values(["depart", "roads"])


def plot_pareto(df: pd.DataFrame, outdir: str = "figs"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for (depart, roads), g in df.groupby(["depart", "roads"]):
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(g["deta_min"], g["risk_drop_abs"])
        for _, r in g.iterrows():
            plt.annotate(
                str(int(r["delta"])),
                (r["deta_min"], r["risk_drop_abs"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points",
            )
        plt.axvline(0.0, lw=1, ls="--")
        plt.axhline(0.0, lw=1, ls="--")
        plt.xlabel("ΔETA (min)")
        plt.ylabel("Risk reduction (abs)")
        plt.title(f"Pareto: ΔETA vs Risk Drop\n{depart} | {roads}")
        plt.tight_layout()
        fname = out / f"pareto__{depart.replace(':', '-')}__{roads.replace(',', '_')}.png"
        fig.savefig(fname, dpi=160)
        plt.close(fig)
        print(f"Saved {fname}")
