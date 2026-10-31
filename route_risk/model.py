"""Model training, validation, and persistence."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import dump, load as jl_load

from .config import WEATHER_VARS

warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning,
    module="sklearn.impute._base",
)


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

NUM_COLS = [
    "curvature", "length_km", "speed_limit", "lanes_n",
    "traffic_proxy", "junction_density", "log_exposure",
    "pct_night", "pct_wet", "pct_junction", "mean_severity",
    "betweenness", "avg_neighbor_crashes", "road_continuity",
]

NUM_COLS_EH = [
    "curvature", "length_km", "speed_limit", "lanes_n",
    "traffic_proxy", "junction_density", "log_exposure",
    "hour", "dow", "month", "season",
    "weather_conditions", "light_conditions", "road_surface_conditions",
    "pct_night", "pct_wet", "pct_junction", "mean_severity",
    "betweenness", "avg_neighbor_crashes", "road_continuity",
]

CAT_COLS = ["highway"]
TARGET_COL = "crash_count"


def _available_weather_cols(df: pd.DataFrame) -> list[str]:
    """Detect weather columns: both plain (``temperature_2m``) and
    seasonal (``temperature_2m_winter``) forms."""
    cols = []
    for wv in WEATHER_VARS:
        if wv in df.columns and df[wv].notna().any():
            cols.append(wv)
        for suffix in ("_winter", "_spring", "_summer", "_autumn"):
            c = wv + suffix
            if c in df.columns and df[c].notna().any():
                cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# LightGBM model builder
# ---------------------------------------------------------------------------

def _build_lgbm(num_cols: list[str], cat_cols: list[str],
                lgbm_params: dict | None = None):
    """Build a LightGBM Poisson model wrapped in a preprocessing pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    import lightgbm as lgb

    num_transform = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    cat_transform = Pipeline([
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                  unknown_value=-1)),
    ])
    pre = ColumnTransformer([
        ("num", num_transform, num_cols),
        ("cat", cat_transform, cat_cols),
    ])
    defaults = dict(
        objective="poisson",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbose=-1,
    )
    if lgbm_params:
        defaults.update(lgbm_params)
    defaults["verbose"] = -1
    clf = lgb.LGBMRegressor(**defaults)
    return Pipeline([("pre", pre), ("clf", clf)])


def _optuna_tune(X_tr, y_tr, num_cols, cat_cols, n_trials=40):
    """Run Optuna hyperparameter search for LightGBM Poisson."""
    import optuna
    from sklearn.model_selection import cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "poisson",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        pipe = _build_lgbm(num_cols, cat_cols, lgbm_params=params)
        scores = cross_val_score(pipe, X_tr, y_tr, cv=3,
                                 scoring="neg_mean_poisson_deviance")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best["objective"] = "poisson"
    print(f"  Optuna best (n={n_trials}): {best}")
    return best


# Fallback: sklearn PoissonRegressor (no lightgbm dependency)
def _build_poisson(num_cols: list[str], cat_cols: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import PoissonRegressor

    num_transform = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_transform, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    clf = PoissonRegressor(alpha=1.0, max_iter=300)
    return Pipeline([("pre", pre), ("clf", clf)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _detect_edge_hour(data: pd.DataFrame) -> bool:
    """Return True if the dataset uses edge-hour format (has condition codes)."""
    return "weather_conditions" in data.columns and "hour" in data.columns


def _get_feature_cols(data: pd.DataFrame,
                      num_cols: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Return (num_cols, feature_cols) respecting available columns."""
    if num_cols is None:
        if _detect_edge_hour(data):
            num_cols = [c for c in NUM_COLS_EH if c in data.columns]
        else:
            weather_cols = _available_weather_cols(data)
            num_cols = [c for c in NUM_COLS if c in data.columns] + weather_cols
    else:
        num_cols = [c for c in num_cols if c in data.columns]
    feature_cols = num_cols + CAT_COLS
    return num_cols, feature_cols


def train(data: pd.DataFrame,
          target: str = TARGET_COL,
          model_out: str = "risk_model.joblib",
          test_size: float = 0.2,
          random_state: int = 42,
          use_lgbm: bool = True,
          test_year: int = 0,
          df_acc_test: "pd.DataFrame | None" = None,
          gdf_edges: "gpd.GeoDataFrame | None" = None) -> dict:
    """Train a Poisson model on a count-based dataset.

    Uses LightGBM by default; falls back to sklearn PoissonRegressor if
    ``use_lgbm=False`` or if lightgbm is not installed.

    When *test_year* > 0 and *df_acc_test* is provided, the held-out year is
    used for temporal validation in addition to the random split.
    """
    num_cols, feature_cols = _get_feature_cols(data)

    y = data[target].values.astype(float)
    X = data[feature_cols].copy()

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    if use_lgbm:
        try:
            import lightgbm as _lgb_check
            best_params = _optuna_tune(X_tr, y_tr, num_cols, CAT_COLS, n_trials=40)
            pipe = _build_lgbm(num_cols, CAT_COLS, lgbm_params=best_params)
            pipe.fit(X_tr, y_tr)
            print(f"  Model: LightGBM Poisson (tuned, {best_params.get('n_estimators', '?')} trees)")
        except ImportError:
            print("  lightgbm not installed, falling back to PoissonRegressor")
            pipe = _build_poisson(num_cols, CAT_COLS)
            pipe.fit(X_tr, y_tr)
    else:
        pipe = _build_poisson(num_cols, CAT_COLS)
        pipe.fit(X_tr, y_tr)
        print("  Model: sklearn PoissonRegressor")

    y_pred = pipe.predict(X_te)
    dev = _poisson_deviance(y_te, y_pred)
    mae = float(np.mean(np.abs(y_te - y_pred)))
    spearman = _spearman(y_te, y_pred)

    print(f"  Random-split validation (n={len(y_te)}): deviance={dev:.4f}  "
          f"MAE={mae:.4f}  Spearman r={spearman:.3f}")

    result = {"deviance": dev, "mae": mae, "spearman": spearman, "n_test": len(y_te)}

    if test_year > 0 and df_acc_test is not None and gdf_edges is not None:
        result.update(_temporal_validation(pipe, data, df_acc_test,
                                          gdf_edges, feature_cols, target, test_year))

    dump(pipe, model_out)
    print(f"  Saved model to {model_out}")
    return result


def _temporal_validation(pipe, train_data, df_acc_test, gdf_edges,
                         feature_cols, target, test_year) -> dict:
    """Validate by counting how well predicted risk correlates with held-out
    year crash counts on the same edge set."""
    from .data import _snap_accidents_to_edges

    test_agg = _snap_accidents_to_edges(df_acc_test, gdf_edges)
    test_counts = test_agg["crash_count"]

    y_actual = np.array([test_counts.get(i, 0) for i in range(len(train_data))])
    X_all = train_data[feature_cols].copy()
    y_pred = pipe.predict(X_all)

    dev = _poisson_deviance(y_actual, y_pred)
    mae = float(np.mean(np.abs(y_actual - y_pred)))
    sp = _spearman(y_actual, y_pred)

    print(f"  Temporal validation ({test_year}, n={len(y_actual)}): "
          f"deviance={dev:.4f}  MAE={mae:.4f}  Spearman r={sp:.3f}")

    return {
        "temporal_deviance": dev, "temporal_mae": mae,
        "temporal_spearman": sp, "temporal_year": test_year,
    }


# ---------------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------------

def cross_validate_spatial(data: pd.DataFrame,
                           target: str = TARGET_COL,
                           n_splits: int = 5,
                           use_lgbm: bool = True) -> dict:
    """K-fold CV with spatial splits (latitude bands)."""
    num_cols, feature_cols = _get_feature_cols(data)

    y = data[target].values.astype(float)
    X = data[feature_cols].copy()

    lat_col = "centroid_lat" if "centroid_lat" in data.columns else None
    if lat_col:
        bands = pd.qcut(data[lat_col], q=n_splits, labels=False, duplicates="drop")
    else:
        bands = np.arange(len(data)) % n_splits

    deviances, maes, spearmans = [], [], []
    for fold in range(int(bands.max()) + 1):
        mask_te = bands == fold
        mask_tr = ~mask_te
        if mask_te.sum() == 0 or mask_tr.sum() == 0:
            continue
        if use_lgbm:
            try:
                pipe = _build_lgbm(num_cols, CAT_COLS)
            except ImportError:
                pipe = _build_poisson(num_cols, CAT_COLS)
        else:
            pipe = _build_poisson(num_cols, CAT_COLS)
        pipe.fit(X[mask_tr], y[mask_tr])
        y_pred = pipe.predict(X[mask_te])
        deviances.append(_poisson_deviance(y[mask_te], y_pred))
        maes.append(float(np.mean(np.abs(y[mask_te] - y_pred))))
        spearmans.append(_spearman(y[mask_te], y_pred))

    result = {
        "mean_deviance": float(np.mean(deviances)),
        "mean_mae": float(np.mean(maes)),
        "mean_spearman": float(np.mean(spearmans)),
        "n_folds": len(deviances),
    }
    print(f"  Spatial CV ({result['n_folds']} folds): "
          f"dev={result['mean_deviance']:.4f}  "
          f"MAE={result['mean_mae']:.4f}  "
          f"Spearman={result['mean_spearman']:.3f}")
    return result


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check_top_edges(data: pd.DataFrame, pipe, n: int = 10):
    """Print the top-n riskiest edges and check they aren't just the busiest."""
    num_cols, feature_cols = _get_feature_cols(data)
    X = data[feature_cols].copy()
    data = data.copy()
    data["predicted_risk"] = pipe.predict(X)
    top = data.nlargest(n, "predicted_risk")
    print(f"\n  Top-{n} predicted-risk edges:")
    cols = ["edge_idx", "highway", "traffic_proxy", "crash_count",
            "severity_rate", "predicted_risk", "length_km"]
    cols = [c for c in cols if c in top.columns]
    print(top[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _poisson_deviance(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-12)
    y_true = np.asarray(y_true, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(2 * np.mean(term - y_true + y_pred))


def _spearman(a, b):
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r) if np.isfinite(r) else 0.0


# ---------------------------------------------------------------------------
# Loading a saved model
# ---------------------------------------------------------------------------

def load_model(path: str = "risk_model.joblib"):
    return jl_load(path)


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

LEAKY_COLS = ["pct_night", "pct_wet", "pct_junction", "mean_severity",
              "avg_neighbor_crashes"]


def compare_baselines(data: pd.DataFrame,
                      target: str = TARGET_COL,
                      test_size: float = 0.2,
                      random_state: int = 42) -> pd.DataFrame:
    """Train four models and return a comparison DataFrame.

    Models: simple heuristic, PoissonRegressor, LightGBM-default, LightGBM-tuned.
    """
    from sklearn.model_selection import train_test_split

    num_cols, feature_cols = _get_feature_cols(data)
    y = data[target].values.astype(float)
    X = data[feature_cols].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    results = []

    # 1) Heuristic: predict crash_count proportional to length * traffic
    heur_pred_tr = (X_tr["length_km"] * X_tr["traffic_proxy"]).values
    heur_pred_te = (X_te["length_km"] * X_te["traffic_proxy"]).values
    scale = y_tr.sum() / max(heur_pred_tr.sum(), 1e-9)
    heur_pred_te = heur_pred_te * scale
    results.append({
        "model": "Heuristic (length*traffic)",
        "deviance": _poisson_deviance(y_te, np.maximum(heur_pred_te, 1e-12)),
        "mae": float(np.mean(np.abs(y_te - heur_pred_te))),
        "spearman": _spearman(y_te, heur_pred_te),
    })
    print(f"  Heuristic: dev={results[-1]['deviance']:.4f} "
          f"MAE={results[-1]['mae']:.4f} Spearman={results[-1]['spearman']:.3f}")

    # 2) PoissonRegressor (linear)
    pipe_pr = _build_poisson(num_cols, CAT_COLS)
    pipe_pr.fit(X_tr, y_tr)
    y_pr = pipe_pr.predict(X_te)
    results.append({
        "model": "PoissonRegressor (linear)",
        "deviance": _poisson_deviance(y_te, y_pr),
        "mae": float(np.mean(np.abs(y_te - y_pr))),
        "spearman": _spearman(y_te, y_pr),
    })
    print(f"  PoissonRegressor: dev={results[-1]['deviance']:.4f} "
          f"MAE={results[-1]['mae']:.4f} Spearman={results[-1]['spearman']:.3f}")

    # 3) LightGBM with default params
    try:
        pipe_def = _build_lgbm(num_cols, CAT_COLS)
        pipe_def.fit(X_tr, y_tr)
        y_def = pipe_def.predict(X_te)
        results.append({
            "model": "LightGBM-default",
            "deviance": _poisson_deviance(y_te, y_def),
            "mae": float(np.mean(np.abs(y_te - y_def))),
            "spearman": _spearman(y_te, y_def),
        })
        print(f"  LightGBM-default: dev={results[-1]['deviance']:.4f} "
              f"MAE={results[-1]['mae']:.4f} Spearman={results[-1]['spearman']:.3f}")

        # 4) LightGBM tuned
        best_params = _optuna_tune(X_tr, y_tr, num_cols, CAT_COLS, n_trials=40)
        pipe_tuned = _build_lgbm(num_cols, CAT_COLS, lgbm_params=best_params)
        pipe_tuned.fit(X_tr, y_tr)
        y_tuned = pipe_tuned.predict(X_te)
        results.append({
            "model": "LightGBM-tuned (Optuna)",
            "deviance": _poisson_deviance(y_te, y_tuned),
            "mae": float(np.mean(np.abs(y_te - y_tuned))),
            "spearman": _spearman(y_te, y_tuned),
        })
        print(f"  LightGBM-tuned: dev={results[-1]['deviance']:.4f} "
              f"MAE={results[-1]['mae']:.4f} Spearman={results[-1]['spearman']:.3f}")
    except ImportError:
        print("  lightgbm not available, skipping LightGBM baselines")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Feature-group ablation
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "road_geometry": ["curvature", "length_km"],
    "traffic": ["speed_limit", "lanes_n", "traffic_proxy", "junction_density", "log_exposure"],
    "graph_topology": ["betweenness", "road_continuity"],
    "stats19_descriptors": ["pct_night", "pct_wet", "pct_junction",
                            "mean_severity", "avg_neighbor_crashes"],
}


def ablation_study(data: pd.DataFrame,
                   target: str = TARGET_COL,
                   test_size: float = 0.2,
                   random_state: int = 42) -> pd.DataFrame:
    """Progressive feature-group ablation.

    Returns a DataFrame with one row per configuration showing how each
    feature group contributes to model performance.
    """
    from sklearn.model_selection import train_test_split

    weather_cols = _available_weather_cols(data)
    y = data[target].values.astype(float)

    configs = [
        ("road_geometry + highway", FEATURE_GROUPS["road_geometry"]),
        ("+ traffic", FEATURE_GROUPS["road_geometry"] + FEATURE_GROUPS["traffic"]),
        ("+ weather", FEATURE_GROUPS["road_geometry"] + FEATURE_GROUPS["traffic"] + weather_cols),
        ("+ graph_topology",
         FEATURE_GROUPS["road_geometry"] + FEATURE_GROUPS["traffic"]
         + weather_cols + FEATURE_GROUPS["graph_topology"]),
        ("+ stats19_descriptors (leaky)",
         FEATURE_GROUPS["road_geometry"] + FEATURE_GROUPS["traffic"]
         + weather_cols + FEATURE_GROUPS["graph_topology"]
         + FEATURE_GROUPS["stats19_descriptors"]),
    ]

    results = []
    for name, num_cols_cfg in configs:
        available = [c for c in num_cols_cfg if c in data.columns]
        feat_cols = available + CAT_COLS
        X = data[feat_cols].copy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )
        try:
            pipe = _build_lgbm(available, CAT_COLS)
        except ImportError:
            pipe = _build_poisson(available, CAT_COLS)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        row = {
            "config": name,
            "n_features": len(available) + len(CAT_COLS),
            "deviance": _poisson_deviance(y_te, y_pred),
            "mae": float(np.mean(np.abs(y_te - y_pred))),
            "spearman": _spearman(y_te, y_pred),
        }
        results.append(row)
        print(f"  {name}: dev={row['deviance']:.4f} MAE={row['mae']:.4f} "
              f"Spearman={row['spearman']:.3f}")

    return pd.DataFrame(results)
