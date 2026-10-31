#!/usr/bin/env python3
"""Generate all figures and formatted tables for the IEEE SIST paper.

Reads experiment outputs from experiments/ and produces figures in
experiments/figs_paper/.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = ROOT / "experiments" / "figs_paper"
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = ROOT / "experiments" / "routing_results.csv"
BASELINE = ROOT / "experiments" / "baseline_comparison.csv"
ABLATION = ROOT / "experiments" / "ablation_study.csv"
AADF_CMP = ROOT / "experiments" / "aadf_comparison.csv"

# =========================================================================
# Table 1: Routing results summary
# =========================================================================
def table_routing_results():
    if not RESULTS.exists():
        print("[SKIP] routing_results.csv not found")
        return
    df = pd.read_csv(RESULTS)
    df["deta_min"] = (df["eta_risk_s"] - df["eta_fast_s"]) / 60.0
    df["risk_drop_pct"] = (df["drisk"] / df["risk_fast"].clip(lower=1e-9)) * 100.0

    summary = df.groupby(["orig", "dest", "depart", "delta"]).agg(
        dist_fast_km=("dist_fast_km", "first"),
        eta_fast_min=("eta_fast_s", lambda x: round(x.iloc[0] / 60, 1)),
        risk_fast=("risk_fast", "first"),
        dist_risk_km=("dist_risk_km", "first"),
        eta_risk_min=("eta_risk_s", lambda x: round(x.iloc[0] / 60, 1)),
        risk_risk=("risk_risk", "first"),
        deta_min=("deta_min", "first"),
        risk_drop_pct=("risk_drop_pct", "first"),
    ).reset_index()

    out_path = OUT / "table_routing_results.csv"
    summary.to_csv(out_path, index=False, float_format="%.2f")
    print(f"  Routing results table: {out_path} ({len(summary)} rows)")

    latex_path = OUT / "table_routing_results.tex"
    with open(latex_path, "w") as f:
        f.write(summary.to_latex(index=False, float_format="%.2f"))
    print(f"  LaTeX: {latex_path}")


# =========================================================================
# Table 2: Baseline comparison
# =========================================================================
def table_baselines():
    if not BASELINE.exists():
        print("[SKIP] baseline_comparison.csv not found")
        return
    df = pd.read_csv(BASELINE)
    out_path = OUT / "table_baseline_comparison.csv"
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Baseline table: {out_path}")


# =========================================================================
# Table 3: Ablation study
# =========================================================================
def table_ablation():
    if not ABLATION.exists():
        print("[SKIP] ablation_study.csv not found")
        return
    df = pd.read_csv(ABLATION)
    out_path = OUT / "table_ablation.csv"
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Ablation table: {out_path}")


# =========================================================================
# Figure 1: Pareto frontiers (ETA vs Risk for different deltas)
# =========================================================================
def fig_pareto_frontiers():
    if not RESULTS.exists():
        print("[SKIP] routing_results.csv not found")
        return
    df = pd.read_csv(RESULTS)
    df["eta_risk_min"] = df["eta_risk_s"] / 60.0
    df["scenario"] = df["orig"].str[:8] + " -> " + df["dest"].str[:8]

    scenarios = df.groupby(["orig", "dest"]).size().nlargest(4).index
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (orig, dest) in zip(axes.flat, scenarios):
        sub = df[(df["orig"] == orig) & (df["dest"] == dest)]
        for depart, g in sub.groupby("depart"):
            g = g.sort_values("delta")
            ax.plot(g["eta_risk_min"], g["risk_risk"], "o-",
                    label=depart.split("T")[1][:5], markersize=5)
            for _, r in g.iterrows():
                ax.annotate(f"d={int(r['delta'])}", (r["eta_risk_min"], r["risk_risk"]),
                            fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("ETA (min)")
        ax.set_ylabel("Risk sum")
        ax.set_title(f"{orig[:8]}... -> {dest[:8]}...", fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle("Pareto Frontiers: ETA vs Risk for Different Dial Settings", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "fig_pareto_frontiers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto frontiers: {OUT / 'fig_pareto_frontiers.png'}")


# =========================================================================
# Figure 2: Feature importance
# =========================================================================
def fig_feature_importance():
    model_path = ROOT / "models" / "risk_model_v2.joblib"
    if not model_path.exists():
        model_path = ROOT / "models" / "risk_model.joblib"
    if not model_path.exists():
        print("[SKIP] No model found for feature importance")
        return

    from joblib import load as jl_load
    pipe = jl_load(str(model_path))
    try:
        lgbm_model = pipe.named_steps["clf"]
        importances = lgbm_model.feature_importances_
        pre = pipe.named_steps["pre"]
        feature_names = []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                feature_names.extend(cols)
        feature_names = feature_names[:len(importances)]

        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=True).tail(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(fi["feature"], fi["importance"], color="#2196F3", edgecolor="white")
        ax.set_xlabel("Feature Importance (split count)")
        ax.set_title("Top Feature Importances (LightGBM Poisson)")
        plt.tight_layout()
        fig.savefig(OUT / "fig_feature_importance.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Feature importance: {OUT / 'fig_feature_importance.png'}")
    except Exception as e:
        print(f"  [SKIP] Feature importance: {e}")


# =========================================================================
# Figure 3: Spatial risk heatmap
# =========================================================================
def fig_spatial_heatmap():
    edge_path = ROOT / "models" / "risk_model_v2_edge_features.csv"
    if not edge_path.exists():
        edge_path = ROOT / "models" / "risk_model_edge_features.csv"
    if not edge_path.exists():
        print("[SKIP] No edge features for heatmap")
        return

    model_path = ROOT / "models" / "risk_model_v2.joblib"
    if not model_path.exists():
        model_path = ROOT / "models" / "risk_model.joblib"

    from joblib import load as jl_load
    from route_risk.model import NUM_COLS, CAT_COLS, _available_weather_cols

    data = pd.read_csv(str(edge_path))
    pipe = jl_load(str(model_path))
    weather_cols = _available_weather_cols(data)
    feature_cols = [c for c in NUM_COLS if c in data.columns] + weather_cols + CAT_COLS
    X = data[feature_cols].copy()
    data["predicted"] = pipe.predict(X)

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(data["centroid_lon"], data["centroid_lat"],
                    c=data["predicted"], cmap="YlOrRd", s=3, alpha=0.6,
                    edgecolors="none", vmin=0,
                    vmax=data["predicted"].quantile(0.95))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Predicted Crash Risk (Greater London)")
    ax.set_aspect("equal")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Predicted crash count")
    plt.tight_layout()
    fig.savefig(OUT / "fig_spatial_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Spatial heatmap: {OUT / 'fig_spatial_heatmap.png'}")


# =========================================================================
# Figure 4: Ablation bar chart
# =========================================================================
def fig_ablation_bars():
    if not ABLATION.exists():
        print("[SKIP] ablation_study.csv not found")
        return
    df = pd.read_csv(ABLATION)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))

    ax = axes[0]
    ax.barh(df["config"], df["deviance"], color=colors)
    ax.set_xlabel("Poisson Deviance")
    ax.set_title("Deviance by Feature Set")
    ax.invert_yaxis()

    ax = axes[1]
    ax.barh(df["config"], df["mae"], color=colors)
    ax.set_xlabel("MAE")
    ax.set_title("MAE by Feature Set")
    ax.invert_yaxis()

    ax = axes[2]
    ax.barh(df["config"], df["spearman"], color=colors)
    ax.set_xlabel("Spearman r")
    ax.set_title("Spearman Correlation by Feature Set")
    ax.invert_yaxis()

    plt.suptitle("Feature Group Ablation Study", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT / "fig_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Ablation bars: {OUT / 'fig_ablation.png'}")


# =========================================================================
# Figure 5: Scenario comparison (ETA and Risk side by side)
# =========================================================================
def fig_scenario_comparison():
    if not RESULTS.exists():
        print("[SKIP] routing_results.csv not found")
        return
    df = pd.read_csv(RESULTS)
    sub = df[df["delta"] == 100].head(12)
    if sub.empty:
        sub = df.head(12)

    sub = sub.copy()
    sub["label"] = sub.apply(
        lambda r: f"{r['orig'][:7]}...\n{r['depart'].split('T')[1][:5]}", axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = range(len(sub))
    w = 0.35

    ax = axes[0]
    ax.bar([i - w / 2 for i in x], sub["eta_fast_s"] / 60, w,
           label="Fastest", color="#1f77b4")
    ax.bar([i + w / 2 for i in x], sub["eta_risk_s"] / 60, w,
           label="Risk-aware", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["label"], fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("ETA (minutes)")
    ax.set_title("Travel Time Comparison (delta=100)")
    ax.legend()

    ax = axes[1]
    ax.bar([i - w / 2 for i in x], sub["risk_fast"], w,
           label="Fastest", color="#1f77b4")
    ax.bar([i + w / 2 for i in x], sub["risk_risk"], w,
           label="Risk-aware", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["label"], fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Risk sum")
    ax.set_title("Risk Comparison (delta=100)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUT / "fig_scenario_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scenario comparison: {OUT / 'fig_scenario_comparison.png'}")


def main():
    print("Generating paper figures and tables...")
    print()
    table_routing_results()
    table_baselines()
    table_ablation()
    fig_pareto_frontiers()
    fig_feature_importance()
    fig_spatial_heatmap()
    fig_ablation_bars()
    fig_scenario_comparison()
    print(f"\nDone! Outputs in {OUT}/")


if __name__ == "__main__":
    main()
