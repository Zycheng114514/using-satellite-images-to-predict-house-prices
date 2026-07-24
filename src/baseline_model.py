"""
Baseline model evaluation.

Combines all years into one dataset (matching how train_model*.py works),
then evaluates simple and tree-based baselines on the combined test split.

Baselines:
  1. Mean predictor       — always predicts the training-set mean price
  2. Median predictor     — always predicts the training-set median price
  3. Linear regression    — OLS on tabular features
  4. Ridge regression     — L2-regularised linear (handles collinearity)
  5. Lasso regression     — L1-regularised linear (built-in feature selection)
  6. Random forest        — ensemble of decision trees
  7. Gradient boosting    — sequential boosted trees

Results are printed and saved to models/results_baseline_all.json.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR     = ROOT / "models"

AVAILABLE_YEARS = [2025, 2023, 2021, 2019, 2017, 2015]

# ── Feature columns (must match train_model*.py) ────────────────────────────
NUMERIC_FEATURES = [
    "BATHRM", "HF_BATHRM", "ROOMS", "BEDRM", "AYB", "YR_RMDL", "EYB",
    "STORIES", "GBA", "LANDAREA", "NUM_UNITS", "KITCHENS", "FIREPLACES",
    "SALE_YEAR",
]
CATEGORICAL_FEATURES = [
    "HEAT", "AC", "STYLE", "STRUCT", "GRADE", "CNDTN", "EXTWALL",
    "ROOF", "INTWALL", "USECODE", "QUALIFIED", "PROP_TYPE",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def normalise_ssl(ssl: str) -> str:
    return re.sub(r"\s+", " ", str(ssl).strip())


def compute_metrics(y_true, y_pred, label=""):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mse = np.mean((y_true - y_pred) ** 2)

    y_dollars = np.expm1(y_true)
    pred_dollars = np.expm1(y_pred)
    ss_res_d = np.sum((y_dollars - pred_dollars) ** 2)
    ss_tot_d = np.sum((y_dollars - y_dollars.mean()) ** 2)
    r2_dollars = 1 - ss_res_d / ss_tot_d if ss_tot_d > 0 else 0.0
    mae_dollars = np.mean(np.abs(y_dollars - pred_dollars))

    return {
        "model": label,
        "r2_log": float(r2),
        "r2_dollars": float(r2_dollars),
        "mse_log": float(mse),
        "mae_dollars": float(mae_dollars),
    }


# ── Load & combine all years ────────────────────────────────────────────────
def load_combined(years: list[int]):
    """Load merged data for all years and concatenate."""
    all_X, all_y_log, all_y_raw = [], [], []

    for year in years:
        path = PROCESSED_DIR / f"merged_{year}.geojson"
        if not path.exists():
            print(f"  [{year}] No merged dataset — skipping")
            continue

        gdf = gpd.read_file(path)
        gdf["SSL"] = gdf["SSL"].apply(normalise_ssl)
        gdf["SALEDATE"] = pd.to_datetime(gdf["SALEDATE"], errors="coerce")
        gdf["PRICE"] = pd.to_numeric(gdf["PRICE"], errors="coerce")
        gdf = gdf[gdf["QUALIFIED"].str.strip() == "Q"]
        gdf = gdf[gdf["PRICE"] > 0]

        df = gdf.copy()
        df["SALE_YEAR"] = pd.to_datetime(df["SALEDATE"], errors="coerce").dt.year
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Fill missing with 0 — missingness is structural (e.g. commercial has no BATHRM)
        df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype(str).astype("category").cat.codes

        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = df[feature_cols].values.astype(np.float32)
        y_raw = df["PRICE"].values.astype(np.float32)
        y_log = np.log1p(y_raw)

        all_X.append(X)
        all_y_log.append(y_log)
        all_y_raw.append(y_raw)
        print(f"  [{year}] {len(y_log):,} rows")

    if not all_X:
        return None, None, None

    return (np.concatenate(all_X),
            np.concatenate(all_y_log),
            np.concatenate(all_y_raw))


def main(years: list[int]):
    print("Loading and combining data …\n")
    X, y_log, y_raw = load_combined(years)
    if X is None:
        print("No data found.")
        return

    X_scaled = StandardScaler().fit_transform(X)

    # Same split as train_model*.py
    idx = np.arange(len(y_log))
    idx_train, idx_test = train_test_split(idx, test_size=0.10, random_state=42)
    idx_val, idx_test   = train_test_split(idx_test, test_size=0.50, random_state=42)

    X_train, X_test = X_scaled[idx_train], X_scaled[idx_test]
    y_train, y_test = y_log[idx_train], y_log[idx_test]

    print(f"\n  Combined: {len(y_log):,} samples  "
          f"(train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)})")

    # ── Naive baselines ──
    mean_pred   = np.full_like(y_test, y_train.mean())
    median_pred = np.full_like(y_test, np.median(y_train))

    # ── Linear models ──
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    lasso = Lasso(alpha=0.01, max_iter=5000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)

    # ── Tree-based models ──
    print("  Training random forest …")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("  Training gradient boosting …")
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, min_samples_leaf=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)

    results = [
        {**compute_metrics(y_test, mean_pred,   "Mean predictor"),    "n_test": len(idx_test)},
        {**compute_metrics(y_test, median_pred, "Median predictor"),  "n_test": len(idx_test)},
        {**compute_metrics(y_test, lr_pred,     "Linear regression"), "n_test": len(idx_test)},
        {**compute_metrics(y_test, ridge_pred,  "Ridge regression"),  "n_test": len(idx_test)},
        {**compute_metrics(y_test, lasso_pred,  "Lasso regression"),  "n_test": len(idx_test)},
        {**compute_metrics(y_test, rf_pred,     "Random forest"),     "n_test": len(idx_test)},
        {**compute_metrics(y_test, gb_pred,     "Gradient boosting"), "n_test": len(idx_test)},
    ]

    # Print table
    print(f"\n{'=' * 80}")
    print(f"  {'Model':<22} {'N_test':>7} {'R²(log)':>10} {'R²($)':>10} {'MSE(log)':>10} {'MAE($)':>12}")
    print(f"  {'-'*22} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for r in results:
        print(f"  {r['model']:<22} {r['n_test']:>7} {r['r2_log']:>10.4f} {r['r2_dollars']:>10.4f} "
              f"{r['mse_log']:>10.4f} {r['mae_dollars']:>11,.0f}")
    print(f"{'=' * 80}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = str(years[0]) if len(years) == 1 else "all"
    results_path = MODEL_DIR / f"results_baseline_{tag}.json"
    save_data = {"years": years, "n_total": len(y_log), "results": results}
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved → {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    parser.add_argument("--years", nargs="+", type=int, default=AVAILABLE_YEARS,
                        help=f"Years to evaluate (default: {AVAILABLE_YEARS})")
    args = parser.parse_args()
    main(args.years)
