"""
Part 3: Train F+I Model

Assumes data_preparation.py and eda.py have already been run.
Reads (per year):
  - data/processed/merged_<year>.geojson      : merged dataset from data_preparation.py
  - data/images/<year>/<ssl>.png              : orthophoto chips from data_preparation.py

Produces (per year, cached):
  - data/embeddings/image_embeddings_<year>.npy : cached 2048-dim Inception v3 embeddings
  - data/embeddings/ssl_order_<year>.npy        : SSL ordering matching the embeddings

Produces (combined across all requested years):
  - models/model_fi_all.pt                    : trained F+I model weights
  - models/results_fi_all.json                : evaluation metrics

Replicates the F+I architecture from Semnani & Rezaei (2021):
  Image branch:   Inception v3 (frozen) → 2048 → FC512 → FC64
  Feature branch: tabular features → FC64
  Fusion:         concat(64+64) → FC → price prediction
"""

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBED_DIR     = DATA_DIR / "embeddings"
MODEL_DIR     = ROOT / "models"
ZIP_PATH      = DATA_DIR / "dataset.zip"
LOCAL_IMG     = Path("/content/local_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVAILABLE_YEARS = [2025, 2023, 2021, 2019, 2017, 2015]


def paths_for_year(year: int):
    """Return a dict of year-specific paths (embeddings are cached per year)."""
    return {
        "merged":  PROCESSED_DIR / f"merged_{year}.geojson",
        "image":   LOCAL_IMG / "images" / str(year),
        "embed":   EMBED_DIR / f"image_embeddings_{year}.npy",
        "ssl":     EMBED_DIR / f"ssl_order_{year}.npy",
    }


def output_paths(years: list[int]):
    """Return paths for model/results — 'all' when multi-year, else single year."""
    tag = str(years[0]) if len(years) == 1 else "all"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "model":   MODEL_DIR / f"model_fi_{tag}.pt",
        "results": MODEL_DIR / f"results_fi_{tag}.json",
    }

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_ssl(ssl: str) -> str:
    return re.sub(r"\s+", " ", str(ssl).strip())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load merged data & filter to Jan–May 2025
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Extract Inception v3 embeddings from saved images
# ══════════════════════════════════════════════════════════════════════════════

def build_inception_encoder():
    """Load Inception v3 with final classification layer removed."""
    local_weights = DATA_DIR / "pretrained_weights" / "inception_v3.pth"
    if local_weights.exists():
        print(f"  Loading weights from {local_weights}")
        model = models.inception_v3(weights=None)
        model.load_state_dict(torch.load(local_weights, map_location=DEVICE, weights_only=True))
    else:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model


def extract_embeddings(gdf, year: int):
    """Load saved images and extract 2048-dim Inception v3 embeddings."""
    p = paths_for_year(year)
    if p["embed"].exists() and p["ssl"].exists():
        print(f"Loading cached embeddings from {p['embed']}")
        return np.load(p["embed"]), np.load(p["ssl"], allow_pickle=True)

    encoder = build_inception_encoder()
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    ssl_list = []
    skipped = 0
    total = len(gdf)
    image_dir = p["image"]

    print(f"\nExtracting embeddings for {total:,} lots from {image_dir} …")
    with torch.no_grad():
        for i, (_, row) in enumerate(gdf.iterrows(), 1):
            ssl = row["SSL"]
            ssl_slug = ssl.replace(" ", "_")
            img_path = image_dir / f"{ssl_slug}.png"

            if not img_path.exists():
                skipped += 1
                continue

            img = Image.open(img_path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            embedding = encoder(tensor).cpu().numpy().squeeze()
            embeddings.append(embedding)
            ssl_list.append(ssl)

            if i % 100 == 0 or i == total:
                print(f"  [{i:>5}/{total}]  extracted={len(embeddings)}  "
                      f"skipped={skipped}")

    embeddings = np.array(embeddings)
    ssl_arr = np.array(ssl_list)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(p["embed"], embeddings)
    np.save(p["ssl"], ssl_arr)
    print(f"Saved {len(embeddings)} embeddings → {p['embed']}")
    if skipped:
        print(f"  {skipped} lots had no image — skipped")
    return embeddings, ssl_arr


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Prepare tabular features
# ══════════════════════════════════════════════════════════════════════════════

def prepare_tabular(gdf, ssl_order):
    """Prepare and normalise tabular features, aligned to ssl_order."""
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

    scaler_X = StandardScaler()
    X_tabular = scaler_X.fit_transform(X_tabular)

    y_log = np.log1p(y)
    scaler_y = StandardScaler()
    y_norm = scaler_y.fit_transform(y_log.reshape(-1, 1)).flatten()

    print(f"  Tabular features: {X_tabular.shape[1]} columns")
    print(f"  Price range: ${y.min():,.0f} – ${y.max():,.0f}")
    return X_tabular, y_norm, y, scaler_X, scaler_y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: F+I Neural Network
# ══════════════════════════════════════════════════════════════════════════════

class HousePriceFI(nn.Module):
    """
    F+I model from Semnani & Rezaei (2021):
      Image:   2048 → 512 → 64  (with dropout)
      Tabular: n_features → 64  (with dropout)
      Fusion:  128 → 1
    """
    def __init__(self, n_tabular_features: int):
        super().__init__()

        self.image_branch = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.feature_branch = nn.Sequential(
            nn.Linear(n_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image_emb, tabular):
        img_out = self.image_branch(image_emb)
        tab_out = self.feature_branch(tabular)
        combined = torch.cat([img_out, tab_out], dim=1)
        return self.fusion(combined).squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Training loop
# ══════════════════════════════════════════════════════════════════════════════

class HouseDataset(Dataset):
    def __init__(self, embeddings, tabular, prices):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        self.prices = torch.tensor(prices, dtype=torch.float32)

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.tabular[idx], self.prices[idx]


def train_model(embeddings, X_tabular, y_log, y_raw, scaler_y, years: list[int],
                epochs=200, lr=0.0005, batch_size=1024, l2_reg=0.1,
                decay_alpha=0.0001):
    n_features = X_tabular.shape[1]

    # Train / val / test split (90% / 5% / 5%)
    idx = np.arange(len(y_log))
    idx_train, idx_test = train_test_split(idx, test_size=0.10, random_state=42)
    idx_val, idx_test = train_test_split(idx_test, test_size=0.50, random_state=42)

    print(f"\n  Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")

    train_ds = HouseDataset(embeddings[idx_train], X_tabular[idx_train], y_log[idx_train])
    val_ds   = HouseDataset(embeddings[idx_val],   X_tabular[idx_val],   y_log[idx_val])
    test_ds  = HouseDataset(embeddings[idx_test],  X_tabular[idx_test],  y_log[idx_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds))
    test_loader  = DataLoader(test_ds,  batch_size=len(test_ds))

    model = HousePriceFI(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    # Paper's per-batch decay: lr_new = lr × 1/(1 + α × batch_number)
    batch_counter = [0]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0 / (1.0 + decay_alpha * batch_counter[0])
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    print("\nTraining F+I model …")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for emb_batch, tab_batch, y_batch in train_loader:
            emb_batch = emb_batch.to(DEVICE)
            tab_batch = tab_batch.to(DEVICE)
            y_batch   = y_batch.to(DEVICE)

            pred = model(emb_batch, tab_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_counter[0] += 1
            scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for emb_batch, tab_batch, y_batch in val_loader:
                emb_batch = emb_batch.to(DEVICE)
                tab_batch = tab_batch.to(DEVICE)
                y_batch   = y_batch.to(DEVICE)
                val_pred = model(emb_batch, tab_batch)
                val_loss = criterion(val_pred, y_batch).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"train_mse={np.mean(train_losses):.4f}  val_mse={val_loss:.4f}")

    # ── Test (best model) ──
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        for emb_batch, tab_batch, y_batch in test_loader:
            emb_batch = emb_batch.to(DEVICE)
            tab_batch = tab_batch.to(DEVICE)
            y_batch   = y_batch.to(DEVICE)
            test_pred = model(emb_batch, tab_batch)

    # Inverse-transform from normalized space → log-price → dollars
    y_test_norm = y_batch.cpu().numpy()
    pred_norm   = test_pred.cpu().numpy()
    y_test_log = scaler_y.inverse_transform(y_test_norm.reshape(-1, 1)).flatten()
    pred_log   = scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).flatten()

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
    print(f"{'='*50}")

    # Save results to JSON
    out = output_paths(years)
    results = {
        "model": "Inception v3",
        "years": years,
        "r2_log": float(r2),
        "r2_dollars": float(r2_dollars),
        "mse_log": float(mse_log),
        "mae_dollars": float(mae_dollars),
        "best_val_mse": float(best_val_loss),
        "epochs": epochs,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
    }
    with open(out["results"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out['results']}")

    torch.save(best_state, out["model"])
    print(f"  Model saved → {out['model']}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F+I model (Inception v3)")
    parser.add_argument("--years", nargs="+", type=int, default=AVAILABLE_YEARS,
                        help=f"Imagery years to train on (default: all {AVAILABLE_YEARS})")
    args = parser.parse_args()
    years = args.years

    # Unzip images to fast local storage (once — zip contains all years)
    first_missing = next(
        (y for y in years if not paths_for_year(y)["image"].exists()), None
    )
    if first_missing and ZIP_PATH.exists():
        print(f"Unzipping {ZIP_PATH} → {LOCAL_IMG} …")
        LOCAL_IMG.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(LOCAL_IMG)
        print("  Done.")

    # Load data + extract embeddings per year, then concatenate
    all_embeddings = []
    all_ssl = []
    all_gdf = []

    for year in years:
        gdf = load_data(year)
        emb, ssl_order = extract_embeddings(gdf, year)
        all_embeddings.append(emb)
        all_ssl.append(ssl_order)
        all_gdf.append((gdf, ssl_order))

    combined_emb = np.concatenate(all_embeddings, axis=0)
    combined_ssl = np.concatenate(all_ssl, axis=0)

    # Build combined tabular + price arrays (concatenate across years)
    all_X = []
    all_y_norm = []
    all_y_raw = []
    scaler_y = None
    for gdf, ssl_order in all_gdf:
        X, yn, yr, _, sy = prepare_tabular(gdf, ssl_order)
        all_X.append(X)
        all_y_norm.append(yn)
        all_y_raw.append(yr)
        scaler_y = sy  # last scaler — will refit on combined below

    combined_X = np.concatenate(all_X, axis=0)
    combined_y_raw = np.concatenate(all_y_raw, axis=0)

    # Fit a single scaler_y on the combined log-prices (not per-year)
    combined_y_log = np.log1p(combined_y_raw)
    scaler_y = StandardScaler()
    combined_y_norm = scaler_y.fit_transform(combined_y_log.reshape(-1, 1)).flatten()

    print(f"\n  Combined: {len(combined_y_norm):,} samples across {len(years)} years")

    model = train_model(combined_emb, combined_X, combined_y_norm, combined_y_raw,
                        scaler_y, years=years)
