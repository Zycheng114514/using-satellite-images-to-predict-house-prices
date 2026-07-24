# Using Satellite Images to Predict House Prices

Predicting Washington, DC home sale prices by combining **tabular property records** with
**aerial/satellite imagery**. The project applies and extends the **Feature-plus-Image (F+I)**
framework of Semnani & Rezaei (2021) to public DC data, and compares three frozen image
encoders — **Inception v3**, **ConvNeXt V2**, and **DINOv2** — against tabular-only baselines.

> **TL;DR** — Adding aerial-image features on top of CAMA tabular records lifts price prediction
> by ~13 percentage points of dollar-space R². A LightGBM model on concatenated
> *Inception v3 embeddings + tabular features* is the single best model
> (**R²($) = 0.87**), while the ConvNeXt V2 neural fusion model has the lowest MAE.

---

## Key results

Combined test set across six imagery years (2015–2025), 8,896 qualified sales.

| Model | Image encoder | R² (log-price) | R² (dollars) | MAE (dollars) |
|-------|:-------------:|:--------------:|:------------:|:-------------:|
| Gradient boosting (tabular only) | — | 0.904 | 0.737 | \$343,434 |
| F+I neural net | Inception v3 | 0.747 | 0.208 | \$636,368 |
| F+I neural net | ConvNeXt V2 &nbsp;<sup>†</sup> | 0.811 | 0.720 | **\$184,024** |
| F+I neural net | DINOv2 | 0.834 | 0.298 | \$568,234 |
| **LightGBM (embeddings + tabular)** | **Inception v3** | **0.909** | **0.870** | \$297,319 |

<sup>†</sup> ConvNeXt V2 was evaluated on a smaller image subset (282 test samples vs. 445 for the others).

**Takeaways** (see the [full report](docs/report/)):

- Satellite imagery adds pricing-relevant signal — street character, neighboring building
  quality, tree canopy, lot density — that structural CAMA fields do not capture.
- On a small–medium dataset (~8k training rows), a tree-based model on the *raw* concatenated
  embeddings beats the neural MLP fusion head, which under-fits the high-dimensional input.
- Metrics can mislead in isolation: linear/ridge look fine in log-space (R² ≈ 0.66) but blow up
  in dollar space because a few very large sales dominate the inverse transform.

All numbers above are reproduced from the JSON result files in [`models/`](models/).

---

## Repository structure

```
.
├── src/                     # Pipeline source code
│   ├── data_preparation.py  #  1. Merge CAMA + lots; fetch ortho image chips (DC ArcGIS)
│   ├── eda.py               #  2. Exploratory data analysis → outputs/eda/
│   ├── baseline_model.py    #  3. Tabular-only baselines (mean/median/linear/RF/GBM)
│   ├── train_model.py       #  4. F+I model — Inception v3 encoder
│   ├── train_model_convnextv2.py  # F+I model — ConvNeXt V2 encoder
│   ├── train_model_gbm.py   #  5. LightGBM on concatenated embeddings + tabular
│   └── evaluate_model.py    #  6. Reload trained models and report test metrics
├── models/                  # Trained model artifacts + result JSONs (tracked)
├── outputs/eda/             # EDA figures + summary_stats.csv
├── docs/
│   ├── report/              # Full write-up (.docx)
│   └── references/          # Bibliography (PDFs kept local, not committed)
├── data/                    # ⚠ NOT in repo — see data/README.md for download link
├── requirements.txt
└── README.md
```

---

## ⚠ Data

The `data/` folder is **not included** in this repository (it is large and re-downloadable).

**Download the prepared bundle here and unzip it into [`data/`](data/):**

**📥 https://drive.google.com/drive/folders/158nALbBXwPRBzNIgveQrxfC1c4yMI3eQ?usp=drive_link**

See [`data/README.md`](data/README.md) for the exact expected folder layout and for
instructions on regenerating everything from the public [DC Open Data](https://opendata.dc.gov/)
sources.

---

## Setup

```bash
# Python 3.12 recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> `geopandas` pulls in GDAL/GEOS/PROJ. If `pip` gives you trouble, install it via conda
> (`conda install -c conda-forge geopandas`) and then `pip install -r requirements.txt`
> for the rest. A CUDA-capable GPU is recommended for the encoder/embedding steps but not required.

---

## Usage

All scripts resolve paths relative to the repository root, so run them from anywhere.
Each accepts `--years` to restrict the imagery years (default: all six).

```bash
# 1. Build the dataset: merge records + download aerial image chips  (needs data/raw/)
python src/data_preparation.py

# 2. Exploratory data analysis  → outputs/eda/
python src/eda.py

# 3. Tabular-only baselines     → models/results_baseline_all.json
python src/baseline_model.py

# 4. Train the F+I models       → models/model_fi_*.pt
python src/train_model.py                 # Inception v3
python src/train_model_convnextv2.py      # ConvNeXt V2

# 5. LightGBM on concatenated embeddings + tabular features
python src/train_model_gbm.py

# 6. Evaluate any saved model on the held-out test split
python src/evaluate_model.py
```

### Pipeline overview

```
CAMA CSVs + Common Ownership Lots (GeoJSON)
        │  data_preparation.py
        ▼
merged_<year>.geojson  +  aerial image chips (data/images/<year>/<ssl>.png)
        │  train_model*.py  (frozen encoder → cached .npy embeddings)
        ▼
image embeddings  ─┬─►  F+I MLP fusion head           (train_model.py / _convnextv2.py)
                   └─►  concat with tabular → LightGBM (train_model_gbm.py)
                              │
                              ▼   evaluate_model.py
                     R² (log & dollars), MAE
```

---

## Models

Trained artifacts are checked into [`models/`](models/):

| File | What it is |
|------|-----------|
| `model_fi_all.pt` | F+I fusion head — Inception v3 embeddings |
| `model_fi_convnextv2_all.pt` | F+I fusion head — ConvNeXt V2 embeddings |
| `model_fi_dinov2_all.pt`, `..._v2.pt` | F+I fusion head — DINOv2 embeddings |
| `model_gbm_all.txt` | LightGBM (embeddings + tabular) — best overall |
| `results_*.json` | Test-set metrics for each model |

> Note: trained **DINOv2** weights are provided and are supported by `evaluate_model.py`,
> but the DINOv2 *training* script is not part of this repository.

---

## Method (summary)

- **Data.** DC CAMA records (residential, commercial, condominium) joined to Common Ownership
  Lot polygons on the SSL parcel id; only qualified, positive-price sales are kept. For each of
  six imagery vintages, sales are kept within ±3 months of the orthophoto flight date and a
  20 m-buffered aerial chip is fetched per lot from the DC ArcGIS ImageServer.
- **Features.** 14 numeric + 12 categorical (integer-encoded) predictors, standardized;
  target is standardized `log(1 + price)`.
- **Encoders (frozen).** Inception v3 (2048-d), ConvNeXt V2 Base (1024-d), DINOv2 ViT-B/14 (768-d).
- **Models.** F+I two-branch MLP (image → 512 → 64, tabular → 64, fused 128 → 64 → 1), trained
  with Adam + per-batch LR decay; plus a LightGBM regressor on the raw 2,074-d concatenation.

Full details are in the [report](docs/report/).

---

## References

This work builds directly on the **F+I** framework of **Semnani & Rezaei (2021)** and uses the
**ConvNeXt V2** (Woo et al., 2023) and **DINOv2** (Oquab et al., 2023) encoders. The complete
bibliography and links are in [`docs/references/README.md`](docs/references/README.md).

## Data sources & attribution

Property records, lot geometries, and aerial orthophotography are © the
Government of the District of Columbia, published via the
[DC Open Data portal](https://opendata.dc.gov/).
