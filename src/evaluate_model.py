"""
Evaluate saved F+I models.

Reloads trained model weights and cached embeddings from disk,
concatenates all years into one dataset (matching how train_model*.py trains),
recomputes the same combined train/val/test split, and prints test-set metrics.

Supports all three encoders:
  - Inception v3  (2048-dim)
  - ConvNeXt V2   (1024-dim)
  - DINOv2        (768-dim)

Requires the corresponding train_model*.py to have been run first.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBED_DIR     = DATA_DIR / "embeddings"
MODEL_DIR     = ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVAILABLE_YEARS = [2025, 2023, 2021, 2019, 2017, 2015]

# ── Encoder configs ──────────────────────────────────────────────────────────
ENCODERS = {
    "inception": {"name": "Inception v3", "embedding_dim": 2048, "suffix": ""},
    "convnextv2": {"name": "ConvNeXt V2", "embedding_dim": 1024, "suffix": "_convnextv2"},
    "dinov2": {"name": "DINOv2", "embedding_dim": 768, "suffix": "_dinov2"},
}

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


# ── Model definition (must match train_model*.py) ───────────────────────────
class HousePriceFI(nn.Module):
    def __init__(self, n_tabular_features: int, embedding_dim: int = 2048):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(embedding_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 64),  nn.ReLU(), nn.Dropout(0.2),
        )
        self.feature_branch = nn.Sequential(
            nn.Linear(n_tabular_features, 64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, image_emb, tabular):
        img_out = self.image_branch(image_emb)
        tab_out = self.feature_branch(tabular)
        return self.fusion(torch.cat([img_out, tab_out], dim=1)).squeeze(1)


# ── Helpers ──────────────────────────────────────────────────────────────────
def normalise_ssl(ssl: str) -> str:
    return re.sub(r"\s+", " ", str(ssl).strip())


def load_data(year: int) -> gpd.GeoDataFrame:
    path = PROCESSED_DIR / f"merged_{year}.geojson"
    gdf = gpd.read_file(path)
    gdf["SSL"] = gdf["SSL"].apply(normalise_ssl)
    gdf["SALEDATE"] = pd.to_datetime(gdf["SALEDATE"], errors="coerce")
    gdf["PRICE"] = pd.to_numeric(gdf["PRICE"], errors="coerce")
    gdf = gdf[gdf["QUALIFIED"].str.strip() == "Q"]
    gdf = gdf[gdf["PRICE"] > 0]
    return gdf


def prepare_tabular(gdf, ssl_order):
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
    X = df[feature_cols].values.astype(np.float32)
    y = df["PRICE"].values.astype(np.float32)
    X = StandardScaler().fit_transform(X)
    y_log = np.log1p(y)
    return X, y_log, y


def model_path_for(encoder_key: str, years: list[int]) -> Path:
    """Find the model file: try 'all' first, then single-year."""
    sfx = ENCODERS[encoder_key]["suffix"]
    tag = str(years[0]) if len(years) == 1 else "all"
    p = MODEL_DIR / f"model_fi{sfx}_{tag}.pt"
    if p.exists():
        return p
    # Fall back: try per-year if only one year
    if len(years) == 1:
        p_all = MODEL_DIR / f"model_fi{sfx}_all.pt"
        if p_all.exists():
            return p_all
    return p  # will be caught by exists() check later


def results_path_for(encoder_key: str, years: list[int]) -> Path:
    sfx = ENCODERS[encoder_key]["suffix"]
    tag = str(years[0]) if len(years) == 1 else "all"
    return MODEL_DIR / f"results_fi{sfx}_{tag}.json"


# ── Load & combine all years (matching train_model*.py) ─────────────────────
def load_combined(encoder_key: str, years: list[int]):
    """Load embeddings + tabular for all years and concatenate."""
    sfx = ENCODERS[encoder_key]["suffix"]
    all_emb, all_X, all_y_raw = [], [], []

    for year in years:
        embed_path = EMBED_DIR / f"image_embeddings{sfx}_{year}.npy"
        ssl_path   = EMBED_DIR / f"ssl_order{sfx}_{year}.npy"
        merged_path = PROCESSED_DIR / f"merged_{year}.geojson"

        if not embed_path.exists() or not ssl_path.exists() or not merged_path.exists():
            print(f"    [{year}] missing files — skipping")
            continue

        gdf = load_data(year)
        embeddings = np.load(embed_path)
        ssl_order  = np.load(ssl_path, allow_pickle=True)
        X, _, y_raw = prepare_tabular(gdf, ssl_order)

        all_emb.append(embeddings)
        all_X.append(X)
        all_y_raw.append(y_raw)
        print(f"    [{year}] {len(y_raw):,} samples")

    if not all_emb:
        return None, None, None, None

    combined_y_raw = np.concatenate(all_y_raw)
    combined_y_log = np.log1p(combined_y_raw)
    scaler_y = StandardScaler()
    combined_y_norm = scaler_y.fit_transform(combined_y_log.reshape(-1, 1)).flatten()

    return (np.concatenate(all_emb),
            np.concatenate(all_X),
            combined_y_norm,
            combined_y_raw,
            scaler_y)


# ── Evaluate one encoder on combined data ────────────────────────────────────
def evaluate_encoder(encoder_key: str, years: list[int]):
    enc = ENCODERS[encoder_key]
    mp = model_path_for(encoder_key, years)

    if not mp.exists():
        print(f"  [{enc['name']}] No model file found — skipping")
        return None

    print(f"\n  [{enc['name']}] Loading combined data …")
    result = load_combined(encoder_key, years)
    if result[0] is None:
        print(f"  [{enc['name']}] No embeddings found — skipping")
        return None
    embeddings, X_tabular, y_norm, y_raw, scaler_y = result

    print(f"    Combined: {len(y_norm):,} samples")

    # Same split as train_model*.py
    idx = np.arange(len(y_norm))
    idx_train, idx_test = train_test_split(idx, test_size=0.10, random_state=42)
    idx_val, idx_test   = train_test_split(idx_test, test_size=0.50, random_state=42)

    model = HousePriceFI(X_tabular.shape[1], enc["embedding_dim"]).to(DEVICE)
    model.load_state_dict(
        torch.load(mp, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        emb = torch.tensor(embeddings[idx_test], dtype=torch.float32).to(DEVICE)
        tab = torch.tensor(X_tabular[idx_test], dtype=torch.float32).to(DEVICE)
        pred_norm = model(emb, tab).cpu().numpy()

    # Inverse-transform from normalized space -> log-price -> dollars
    y_test_log = scaler_y.inverse_transform(y_norm[idx_test].reshape(-1, 1)).flatten()
    pred_log   = scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).flatten()

    ss_res = np.sum((y_test_log - pred_log) ** 2)
    ss_tot = np.sum((y_test_log - y_test_log.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    y_dollars    = np.expm1(y_test_log)
    pred_dollars = np.expm1(pred_log)
    ss_res_d = np.sum((y_dollars - pred_dollars) ** 2)
    ss_tot_d = np.sum((y_dollars - y_dollars.mean()) ** 2)
    r2_dollars = 1 - ss_res_d / ss_tot_d

    mse_log = np.mean((y_test_log - pred_log) ** 2)
    mae_dollars = np.mean(np.abs(y_dollars - pred_dollars))

    results = {
        "model": enc["name"],
        "years": years,
        "r2_log": float(r2),
        "r2_dollars": float(r2_dollars),
        "mse_log": float(mse_log),
        "mae_dollars": float(mae_dollars),
        "n_total": int(len(y_norm)),
        "n_test": int(len(idx_test)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rp = results_path_for(encoder_key, years)
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main(years: list[int], encoders: list[str]):
    print(f"Evaluating F+I models on combined data ({years}) …")
    all_results = []

    for enc_key in encoders:
        result = evaluate_encoder(enc_key, years)
        if result:
            all_results.append(result)

    if not all_results:
        print("No models found to evaluate. Run train_model*.py first.")
        return

    print(f"\n{'=' * 75}")
    print(f"  {'Model':<16} {'N_test':>7} {'R²(log)':>10} {'R²($)':>10} {'MSE(log)':>10} {'MAE($)':>12}")
    print(f"  {'-'*16} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for r in all_results:
        print(f"  {r['model']:<16} {r['n_test']:>7} {r['r2_log']:>10.4f} {r['r2_dollars']:>10.4f} "
              f"{r['mse_log']:>10.4f} {r['mae_dollars']:>11,.0f}")
    print(f"{'=' * 75}")

    print(f"\n  Results saved to models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved F+I models")
    parser.add_argument("--years", nargs="+", type=int, default=AVAILABLE_YEARS,
                        help=f"Years to evaluate (default: {AVAILABLE_YEARS})")
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()),
                        choices=list(ENCODERS.keys()),
                        help="Encoders to evaluate (default: all)")
    args = parser.parse_args()
    main(args.years, args.encoders)
