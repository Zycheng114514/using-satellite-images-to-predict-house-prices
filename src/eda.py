"""
Part 2: Exploratory Data Analysis

Reads the merged datasets from data_preparation.py (one per year) and produces:
  - Summary statistics
  - Distribution plots (price, GBA, lot area)
  - Correlation heatmap
  - Sample image grid
  - Cross-year price comparison
  - Outlier analysis
  - Price per sqft spatial map

All figures saved to outputs/eda/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGE_DIR     = DATA_DIR / "images"
OUT_DIR    = ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AVAILABLE_YEARS = [2025, 2023, 2021, 2019, 2017, 2015]

NUMERIC_COLS = ["PRICE", "GBA", "LANDAREA", "BATHRM", "HF_BATHRM",
                "ROOMS", "BEDRM", "STORIES", "AYB", "YR_RMDL", "EYB",
                "FIREPLACES", "KITCHENS", "NUM_UNITS"]


def load_year(year: int) -> gpd.GeoDataFrame | None:
    """Load a single year's merged dataset."""
    path = PROCESSED_DIR / f"merged_{year}.geojson"
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    gdf["SALEDATE"] = pd.to_datetime(gdf["SALEDATE"], errors="coerce")
    for col in NUMERIC_COLS:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
    gdf["_year"] = year
    return gdf


def load_all_years(years: list[int] = AVAILABLE_YEARS) -> gpd.GeoDataFrame:
    """Load and concatenate all available year datasets."""
    frames = []
    for year in years:
        gdf = load_year(year)
        if gdf is not None:
            print(f"  [{year}] {len(gdf):,} rows")
            frames.append(gdf)
        else:
            print(f"  [{year}] not found — skipping")
    if not frames:
        raise FileNotFoundError("No merged datasets found")
    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)
    print(f"  Total: {len(combined):,} rows across {len(frames)} years")
    return combined


def load_data(year: int | None = None) -> gpd.GeoDataFrame:
    """Load a single year or all years."""
    print("Loading merged dataset(s) …")
    if year is not None:
        gdf = load_year(year)
        if gdf is None:
            raise FileNotFoundError(f"No merged dataset for {year}")
        print(f"  {len(gdf):,} rows loaded ({year})")
        return gdf
    return load_all_years()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Summary statistics
# ══════════════════════════════════════════════════════════════════════════════

def summary_stats(gdf):
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    num_cols = ["PRICE", "GBA", "LANDAREA", "BATHRM", "BEDRM",
                "ROOMS", "STORIES", "FIREPLACES", "AYB"]
    stats = gdf[num_cols].describe().T
    stats["missing"] = gdf[num_cols].isna().sum()
    print(stats.to_string())

    out_path = OUT_DIR / "summary_stats.csv"
    stats.to_csv(out_path)
    print(f"\n  Saved → {out_path}")

    # Categorical value counts
    print("\n── Categorical distributions ──")
    for col in ["STYLE_D", "STRUCT_D", "GRADE_D", "CNDTN_D", "HEAT_D", "AC"]:
        if col in gdf.columns:
            print(f"\n  {col}:")
            vc = gdf[col].value_counts().head(8)
            for val, cnt in vc.items():
                print(f"    {val}: {cnt}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Price distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_price_distribution(gdf):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Raw price (units: $100K, outliers removed via 1.5× IQR) ──
    prices_all = gdf["PRICE"].dropna()
    q1, q3 = prices_all.quantile(0.25), prices_all.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    prices = prices_all[(prices_all >= lo) & (prices_all <= hi)]
    n_dropped = len(prices_all) - len(prices)

    axes[0].hist(prices / 1e5, bins=60, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Sale Price ($100K)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Price Distribution")
    axes[0].axvline(prices.median() / 1e5, color="red", ls="--",
                    label=f"Median: ${prices.median():,.0f}")
    axes[0].plot([], [], ' ',
                 label=f"Excluded {n_dropped:,} outliers (1.5× IQR)")
    axes[0].legend()

    # ── Log price (uses full distribution) ──
    log_prices = np.log1p(prices_all)
    axes[1].hist(log_prices, bins=60, color="darkorange", edgecolor="white")
    axes[1].set_xlabel("log(1 + Price)")
    axes[1].set_title("Log-Price Distribution")

    # ── Price per sqft (units: $100/sqft, outliers removed via 1.5× IQR) ──
    mask = (gdf["GBA"] > 0) & (gdf["PRICE"] > 0)
    ppsf_all = gdf.loc[mask, "PRICE"] / gdf.loc[mask, "GBA"]
    pq1, pq3 = ppsf_all.quantile(0.25), ppsf_all.quantile(0.75)
    piqr = pq3 - pq1
    plo, phi = pq1 - 1.5 * piqr, pq3 + 1.5 * piqr
    ppsf = ppsf_all[(ppsf_all >= plo) & (ppsf_all <= phi)]
    n_dropped_ppsf = len(ppsf_all) - len(ppsf)

    axes[2].hist(ppsf / 100, bins=60, color="seagreen", edgecolor="white")
    axes[2].set_xlabel("Price per Sq Ft ($100/sqft)")
    axes[2].set_title("Price / GBA Distribution")
    axes[2].axvline(ppsf.median() / 100, color="red", ls="--",
                    label=f"Median: ${ppsf.median():,.0f}/sqft")
    axes[2].plot([], [], ' ',
                 label=f"Excluded {n_dropped_ppsf:,} outliers (1.5× IQR)")
    axes[2].legend()

    plt.tight_layout()
    out_path = OUT_DIR / "price_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature distributions (GBA, lot area, bedrooms, year built)
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_distributions(gdf):
    # (col, label, color, filter_mode)
    #   "iqr"  → drop points outside 1.5× IQR
    #   "year" → drop sentinel/invalid years (keep AYB ≥ 1800)
    #   None   → no filtering (used for low-cardinality counts like BEDRM)
    features = [
        ("GBA",      "Gross Building Area (sqft)", "steelblue",    "iqr"),
        ("LANDAREA", "Land Area (sqft)",           "darkorange",   "iqr"),
        ("BEDRM",    "Bedrooms",                   "seagreen",     None),
        ("AYB",      "Actual Year Built",          "mediumpurple", "year"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, label, color, mode) in zip(axes.flat, features):
        data_all = gdf[col].dropna()

        if mode == "iqr":
            q1, q3 = data_all.quantile(0.25), data_all.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            data = data_all[(data_all >= lo) & (data_all <= hi)]
            n_dropped = len(data_all) - len(data)
            extra_label = f"Excluded {n_dropped:,} outliers (1.5× IQR)"
        elif mode == "year":
            data = data_all[data_all >= 1800]
            n_dropped = len(data_all) - len(data)
            extra_label = f"Excluded {n_dropped:,} invalid years (<1800)"
        else:
            data = data_all
            extra_label = None

        ax.hist(data, bins=50, color=color, edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Distribution")
        ax.axvline(data.median(), color="red", ls="--",
                   label=f"Median: {data.median():,.0f}")
        if extra_label:
            ax.plot([], [], ' ', label=extra_label)
        ax.legend()

    plt.tight_layout()
    out_path = OUT_DIR / "feature_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Correlation heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(gdf):
    num_cols = ["PRICE", "GBA", "LANDAREA", "BATHRM", "HF_BATHRM",
                "BEDRM", "ROOMS", "STORIES", "FIREPLACES", "AYB",
                "YR_RMDL", "EYB", "KITCHENS", "NUM_UNITS"]
    corr = gdf[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)

    # Annotate cells
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    out_path = OUT_DIR / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Sample image grid (show a few lots across years if available)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sample_images(gdf, n_samples=4):
    years = [y for y in [2015, 2019, 2023, 2025] if (IMAGE_DIR / str(y)).exists()]
    if not years:
        print("  No image directories found — skipping image grid")
        return

    samples = gdf.sample(n=min(n_samples, len(gdf)), random_state=42)

    fig = plt.figure(figsize=(4 * len(years), 4 * len(samples)))
    gs = gridspec.GridSpec(len(samples), len(years), wspace=0.05, hspace=0.15)

    for row_i, (_, row) in enumerate(samples.iterrows()):
        ssl_slug = row["SSL"].replace(" ", "_")
        for col_i, year in enumerate(years):
            ax = fig.add_subplot(gs[row_i, col_i])
            img_path = IMAGE_DIR / str(year) / f"{ssl_slug}.png"
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(str(year), fontsize=12, fontweight="bold")
            if col_i == 0:
                price_str = f"${row['PRICE']:,.0f}" if pd.notna(row['PRICE']) else "?"
                ax.set_ylabel(f"{row['SSL']}\n{price_str}",
                              fontsize=8, rotation=0, labelpad=80, va="center")

    fig.suptitle("Sample Lots Across Years", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = OUT_DIR / "sample_images.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Cross-year price comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_year_comparison(gdf):
    if "_year" not in gdf.columns:
        print("  Single year loaded — skipping cross-year comparison")
        return

    years_present = sorted(gdf["_year"].unique())
    if len(years_present) < 2:
        print("  Only one year available — skipping cross-year comparison")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Median price by year
    yearly = gdf.groupby("_year")["PRICE"].agg(["median", "mean", "count"])
    axes[0].bar(yearly.index.astype(str), yearly["median"] / 1e3,
                color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Median Price ($K)")
    axes[0].set_title("Median Sale Price by Year")
    for i, (yr, row) in enumerate(yearly.iterrows()):
        axes[0].text(i, row["median"] / 1e3 + 5, f"n={int(row['count'])}",
                     ha="center", fontsize=8)

    # Price distribution by year (boxplot)
    data_by_year = [gdf.loc[gdf["_year"] == y, "PRICE"].dropna() / 1e3
                    for y in years_present]
    bp = axes[1].boxplot(data_by_year, labels=[str(y) for y in years_present],
                         patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Price ($K)")
    axes[1].set_title("Price Distribution by Year")

    # Sale count by year
    axes[2].bar(yearly.index.astype(str), yearly["count"],
                color="darkorange", edgecolor="white")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Number of Sales")
    axes[2].set_title("Sale Count by Year")

    plt.tight_layout()
    out_path = OUT_DIR / "cross_year_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Outlier analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_outlier_analysis(gdf):
    prices = gdf["PRICE"].dropna()
    q1, q3 = prices.quantile(0.25), prices.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = prices[(prices < lower) | (prices > upper)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Violin plot of price by year
    if "_year" in gdf.columns:
        years_present = sorted(gdf["_year"].unique())
        data = [gdf.loc[gdf["_year"] == y, "PRICE"].dropna() / 1e6 for y in years_present]
        vp = axes[0].violinplot(data, positions=range(len(years_present)), showmedians=True)
        axes[0].set_xticks(range(len(years_present)))
        axes[0].set_xticklabels([str(y) for y in years_present])
        axes[0].set_xlabel("Year")
    else:
        axes[0].violinplot([prices / 1e6], showmedians=True)
    axes[0].set_ylabel("Price ($M)")
    axes[0].set_title("Price Distribution (Violin)")

    # Log-price with outlier thresholds
    log_prices = np.log1p(prices)
    lq1, lq3 = log_prices.quantile(0.25), log_prices.quantile(0.75)
    liqr = lq3 - lq1
    axes[1].hist(log_prices, bins=60, color="steelblue", edgecolor="white")
    axes[1].axvline(lq1 - 1.5 * liqr, color="red", ls="--", label="1.5x IQR bounds")
    axes[1].axvline(lq3 + 1.5 * liqr, color="red", ls="--")
    axes[1].set_xlabel("log(1 + Price)")
    axes[1].set_title("Log-Price with Outlier Bounds")
    axes[1].legend()

    # Top outliers table as text
    top = prices.nlargest(10)
    text = f"Outliers: {len(outliers)} / {len(prices)} ({100*len(outliers)/len(prices):.1f}%)\n"
    text += f"IQR bounds: ${lower:,.0f} – ${upper:,.0f}\n\n"
    text += "Top 10 prices:\n"
    for i, (idx, val) in enumerate(top.items(), 1):
        text += f"  {i}. ${val:,.0f}\n"
    axes[2].text(0.05, 0.95, text, transform=axes[2].transAxes, fontsize=9,
                 verticalalignment="top", fontfamily="monospace")
    axes[2].axis("off")
    axes[2].set_title("Outlier Summary")

    plt.tight_layout()
    out_path = OUT_DIR / "outlier_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Price per sqft spatial map
# ══════════════════════════════════════════════════════════════════════════════

def plot_ppsf_spatial(gdf):
    gdf_plot = gdf[(gdf["PRICE"] > 0) & (gdf["GBA"] > 0)].copy()
    gdf_plot["PPSF"] = gdf_plot["PRICE"] / gdf_plot["GBA"]

    # Clip extreme price/sqft for better color scale
    p5, p95 = gdf_plot["PPSF"].quantile(0.05), gdf_plot["PPSF"].quantile(0.95)
    gdf_plot["PPSF_clipped"] = gdf_plot["PPSF"].clip(p5, p95)

    gdf_plot["centroid"] = gdf_plot.geometry.centroid
    gdf_plot = gdf_plot.set_geometry("centroid")

    fig, ax = plt.subplots(figsize=(12, 12))
    gdf_plot.plot(
        ax=ax,
        column="PPSF_clipped",
        cmap="RdYlGn_r",
        markersize=3,
        alpha=0.6,
        legend=True,
        legend_kwds={"label": "Price per Sq Ft ($)", "shrink": 0.5},
    )
    ax.set_title("Price per Sq Ft — Washington DC (5th–95th percentile)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    plt.tight_layout()
    out_path = OUT_DIR / "ppsf_spatial_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EDA for DC house price data")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year to analyze (default: all available)")
    args = parser.parse_args()

    gdf = load_data(year=args.year)

    print("\n── 1. Summary Statistics ──")
    summary_stats(gdf)

    print("\n── 2. Price Distribution ──")
    plot_price_distribution(gdf)

    print("\n── 3. Feature Distributions ──")
    plot_feature_distributions(gdf)

    print("\n── 4. Correlation Heatmap ──")
    plot_correlation_heatmap(gdf)

    print("\n── 5. Sample Image Grid ──")
    plot_sample_images(gdf)

    print("\n── 6. Cross-Year Comparison ──")
    plot_cross_year_comparison(gdf)

    print("\n── 7. Outlier Analysis ──")
    plot_outlier_analysis(gdf)

    print("\n── 8. Price/SqFt Spatial Map ──")
    plot_ppsf_spatial(gdf)

    print(f"\nAll EDA outputs saved to {OUT_DIR}/")
