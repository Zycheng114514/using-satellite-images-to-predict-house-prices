"""
Part 3d: Train GBM model on concatenated image embeddings + tabular features

Instead of an MLP fusion head, this script concatenates the pre-extracted
image embeddings (2048-dim from Inception v3) directly with the tabular
features (~26 columns) and trains a LightGBM regressor on the combined
~2074-dim feature vector.

Reads:
  - data/processed/merged_<year>.geojson
  - data/embeddings/image_embeddings_<year>.npy
  - data/embeddings/ssl_order_<year>.npy

Produces:
  - models/model_gbm_all.txt          : trained LightGBM model
  - models/results_gbm_all.json       : evaluation metrics
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBED_DIR     = DATA_DIR / "embeddings"
MODEL_DIR     = ROOT / "models"

AVAILABLE_YEARS = [2025, 2023, 2021, 2019, 2017, 2015]

# ── Feature columns ──────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "BATHRM", "HF_BATHRM", "ROOMS", "BEDRM", "AYB", "YR_RMDL", "EYB",
    "STORIES", "GBA", "LANDAREA", "NUM_UNITS", "KITCHENS", "FIREPLACES",
    "SALE_YEAR",
]
CATEGORICAL_FEATURES = [
    "HEAT", "AC", "STYLE", "STRUCT", "GRADE", "CNDTN", "EXTWALL",
    "ROOF", "INTWALL", "USECODE", "QUALIFIED", "PROP_TYPE",
]


def paths_for_year(year: int):
    return {
        "merged": PROCESSED_DIR / f"merged_{year}.geojson",
        "embed":  EMBED_DIR / f"image_embeddings_{year}.npy",
        "ssl":    EMBED_DIR / f"ssl_order_{year}.npy",
    }


def output_paths(years: list[int]):
    tag = str(years[0]) if len(years) == 1 else "all"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "model":   MODEL_DIR / f"model_gbm_{tag}.txt",
        "results": MODEL_DIR / f"results_gbm_{tag}.json",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_ssl(ssl: str) -> str:
    return re.sub(r"\s+", " ", str(ssl).strip())


def load_data(year: int) -> gpd.GeoDataFrame:
    p = paths_for_year(year)
    print(f"Loading merged dataset for {year} …")
    gdf = gpd.read_file(p["merged"])
    gdf["SSL"] = gdf["SSL"].apply(normalise_ssl)
    gdf["SALEDATE"] = pd.to_datetime(gdf["SALEDATE"], errors="coerce")
    gdf["PRICE"] = pd.to_numeric(gdf["PRICE"], errors="coerce")
    gdf = gdf[gdf["QUALIFIED"].str.strip() == "Q"]
    gdf = gdf[gdf["PRICE"] > 0]
    print(f"  {len(gdf):,} rows ({year}, qualified, price > 0)")
    return gdf


def load_embeddings(year: int):
    p = paths_for_year(year)
    if not p["embed"].exists() or not p["ssl"].exists():
        raise FileNotFoundError(
            f"Embeddings not found for {year}. Run train_model.py first to extract them."
        )
    return np.load(p["embed"]), np.load(p["ssl"], allow_pickle=True)


def prepare_tabular(gdf, ssl_order):
    """Prepare tabular features aligned to ssl_order. Returns raw (unscaled) for GBM."""
    df = gdf.drop_duplicates(subset="SSL").set_index("SSL").loc[ssl_order].reset_index()

    # Derive sale year from SALEDATE as a temporal market feature
    df["SALE_YEAR"] = pd.to_datetime(df["SALEDATE"], errors="coerce").dt.year

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Fill missing with 0 — missingness is structural (e.g. commercial has no BATHRM)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).astype("category").cat.codes

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_tabular = df[feature_cols].values.astype(np.float32)
    y = df["PRICE"].values.astype(np.float32)

    # No StandardScaler — GBM is tree-based and doesn't need feature scaling
    y_log = np.log1p(y)

    print(f"  Tabular features: {X_tabular.shape[1]} columns")
    print(f"  Price range: ${y.min():,.0f} – ${y.max():,.0f}")
    return X_tabular, y_log, y


# ── Training ─────────────────────────────────────────────────────────────���───

def train_gbm(X, y_log, y_raw, years: list[int]):
    """Train LightGBM on concatenated embeddings + tabular features."""
    n_emb = X.shape[1] - len(NUMERIC_FEATURES) - len(CATEGORICAL_FEATURES)
    print(f"\n  Total features: {X.shape[1]} "
          f"({n_emb} embedding + {len(NUMERIC_FEATURES)} numeric + "
          f"{len(CATEGORICAL_FEATURES)} categorical)")

    # Train / val / test split (90% / 5% / 5%) — same as neural net scripts
    idx = np.arange(len(y_log))
    idx_train, idx_test = train_test_split(idx, test_size=0.10, random_state=42)
    idx_val, idx_test = train_test_split(idx_test, test_size=0.50, random_state=42)

    print(f"  Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")

    train_set = lgb.Dataset(X[idx_train], label=y_log[idx_train])
    val_set   = lgb.Dataset(X[idx_val],   label=y_log[idx_val], reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 10,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    print("\nTraining LightGBM …")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    # ── Test evaluation ──
    pred_log = model.predict(X[idx_test])
    y_test_log = y_log[idx_test]

    # R² in log-price space
    ss_res = np.sum((y_test_log - pred_log) ** 2)
    ss_tot = np.sum((y_test_log - y_test_log.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # R² in dollar space
    y_test_dollars = np.expm1(y_test_log)
    pred_dollars   = np.expm1(pred_log)
    ss_res_d = np.sum((y_test_dollars - pred_dollars) ** 2)
    ss_tot_d = np.sum((y_test_dollars - y_test_dollars.mean()) ** 2)
    r2_dollars = 1 - ss_res_d / ss_tot_d

    mse_log = np.mean((y_test_log - pred_log) ** 2)
    mae_dollars = np.mean(np.abs(y_test_dollars - pred_dollars))

    print(f"\n{'='*50}")
    print(f"  Test R² (log-price):  {r2:.4f}")
    print(f"  Test R² (dollars):    {r2_dollars:.4f}")
    print(f"  Test MSE (log-price): {mse_log:.4f}")
    print(f"  Test MAE (dollars):   ${mae_dollars:,.0f}")
    print(f"  Best iteration:       {model.best_iteration}")
    print(f"{'='*50}")

    # Feature importance (top 20)
    importance = model.feature_importance(importance_type="gain")
    n_tab = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    feat_names = [f"emb_{i}" for i in range(X.shape[1] - n_tab)]
    feat_names += NUMERIC_FEATURES + CATEGORICAL_FEATURES
    top_idx = np.argsort(importance)[::-1][:20]
    print("\n  Top 20 features by gain:")
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank:>2}. {feat_names[i]:<20s}  {importance[i]:,.0f}")

    # Save
    out = output_paths(years)
    model.save_model(str(out["model"]))
    print(f"\n  Model saved → {out['model']}")

    results = {
        "model": "LightGBM (embeddings + tabular)",
        "years": years,
        "r2_log": float(r2),
        "r2_dollars": float(r2_dollars),
        "mse_log": float(mse_log),
        "mae_dollars": float(mae_dollars),
        "best_iteration": model.best_iteration,
        "n_features": int(X.shape[1]),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
    }
    with open(out["results"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out['results']}")

    return model


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GBM on concatenated image embeddings + tabular features"
    )
    parser.add_argument("--years", nargs="+", type=int, default=AVAILABLE_YEARS,
                        help=f"Imagery years to train on (default: all {AVAILABLE_YEARS})")
    args = parser.parse_args()
    years = args.years

    all_emb, all_X, all_y_log, all_y_raw = [], [], [], []

    for year in years:
        gdf = load_data(year)
        embeddings, ssl_order = load_embeddings(year)
        X_tab, y_log, y_raw = prepare_tabular(gdf, ssl_order)

        all_emb.append(embeddings)
        all_X.append(X_tab)
        all_y_log.append(y_log)
        all_y_raw.append(y_raw)

    combined_emb = np.concatenate(all_emb, axis=0)
    combined_X   = np.concatenate(all_X, axis=0)
    combined_y_log = np.concatenate(all_y_log, axis=0)
    combined_y_raw = np.concatenate(all_y_raw, axis=0)

    # Concatenate: [2048 embedding dims | tabular features]
    X_full = np.concatenate([combined_emb, combined_X], axis=1)

    print(f"\n  Combined: {len(combined_y_log):,} samples across {len(years)} years")
    print(f"  Feature matrix: {X_full.shape}")

    train_gbm(X_full, combined_y_log, combined_y_raw, years)
