"""
Part 1: Data Preparation
  - Load GeoJSON (lot polygons) and CAMA Residential CSV
  - For each imagery year, filter to qualified sales ±3 months of the
    ortho collection date
  - Merge on SSL
  - Fetch orthophoto image chips from DC ArcGIS ImageServer

Output:
  - data/processed/merged_<year>.geojson  : merged GeoDataFrame per year
  - data/images/<year>/<ssl>.png          : orthophoto chip per lot per year
"""

import time
import re
import zipfile
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import geopandas as gpd
from dateutil.relativedelta import relativedelta
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_DIR     = ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GEOJSON_PATH   = RAW_DIR / "Common_Ownership_Lots.geojson"
CSV_RESIDENTIAL = RAW_DIR / "Computer_Assisted_Mass_Appraisal_-_Residential.csv"
CSV_COMMERCIAL  = RAW_DIR / "COMMERCIAL_(CAMA).csv"
CSV_CONDOMINIUM = RAW_DIR / "CONDOMINIUM_(CAMA).csv"

# ── ArcGIS ImageServer ─────────────────────────────────────────────────────────
ARCGIS_BASE     = "https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho"
AVAILABLE_YEARS = [2025, 2023, 2021, 2019, 2017, 2015]

# Orthophoto collection (flight) dates — used to compute ±3-month sale windows
COLLECTION_DATES = {
    2025: pd.Timestamp("2025-03-15"),   # inferred from original Jan–May filter
    2023: pd.Timestamp("2023-05-08"),   # flown May 6–10, 2023
    2021: pd.Timestamp("2021-03-11"),   # flown March 11, 2021
    2019: pd.Timestamp("2019-04-23"),   # flown April 23, 2019
    2017: pd.Timestamp("2017-03-08"),   # flown early March, completed March 8
    2015: pd.Timestamp("2015-04-24"),   # flown mid-late April, completed April 24
}

NATIVE_RES_M  = 0.08   # 3-inch / 0.08m per pixel (source resolution)
MAX_IMAGE_PX  = 1024   # cap to avoid huge requests for very large lots
LOT_BUFFER_M  = 20     # metres of padding around lot boundary
MAX_RETRIES   = 5      # retry attempts per image on network failure
RETRY_BACKOFF = 2.0    # seconds — doubles on each retry
MAX_WORKERS   = 6      # concurrent download threads


def sale_window(year: int) -> tuple[str, str]:
    """Return (start, end) date strings for ±3 months around the collection date."""
    centre = COLLECTION_DATES[year]
    start = centre - relativedelta(months=3)
    end   = centre + relativedelta(months=3)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_ssl(ssl: str) -> str:
    """Collapse internal whitespace so SSLs match across datasets."""
    return re.sub(r"\s+", " ", str(ssl).strip())


def image_server_url(year: int) -> str:
    return f"{ARCGIS_BASE}/Ortho_{year}/ImageServer/exportImage"


def native_pixel_size(bounds_m) -> tuple[int, int]:
    """Compute native pixel dimensions for a bounding box in metres at 0.08m/px."""
    minx, miny, maxx, maxy = bounds_m
    w = min(int((maxx - minx) / NATIVE_RES_M), MAX_IMAGE_PX)
    h = min(int((maxy - miny) / NATIVE_RES_M), MAX_IMAGE_PX)
    return max(w, 1), max(h, 1)


def fetch_image(bounds_wgs84, bounds_m, year: int) -> Image.Image | None:
    """Fetch a PNG chip from the ArcGIS ImageServer with exponential backoff retry."""
    minx, miny, maxx, maxy = bounds_wgs84
    w, h = native_pixel_size(bounds_m)
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}", "bboxSR": 4326,
        "size": f"{w},{h}", "imageSR": 4326,
        "format": "png", "pixelType": "U8", "f": "image",
    }
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(image_server_url(year), params=params, timeout=30)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"    [fail] {year} gave up after {MAX_RETRIES} attempts: {e}")
                return None
            print(f"    [retry {attempt}/{MAX_RETRIES}] {year} — {e} — waiting {delay:.0f}s")
            time.sleep(delay)
            delay *= 2


# ── Step 1: Load & merge ──────────────────────────────────────────────────────

def _load_raw():
    """Load lot polygons and CAMA data (residential + commercial + condominium)."""
    print("Loading GeoJSON …")
    lots = gpd.read_file(GEOJSON_PATH)
    lots["SSL"] = lots["SSL"].apply(normalise_ssl)

    # Load all three property type CSVs
    print("Loading residential CSV …")
    df_res = pd.read_csv(CSV_RESIDENTIAL)
    df_res["PROP_TYPE"] = "residential"

    print("Loading commercial CSV …")
    df_com = pd.read_csv(CSV_COMMERCIAL)
    df_com["PROP_TYPE"] = "commercial"
    df_com.rename(columns={"LIVING_GBA": "GBA"}, inplace=True)

    print("Loading condominium CSV …")
    df_con = pd.read_csv(CSV_CONDOMINIUM)
    df_con["PROP_TYPE"] = "condominium"
    df_con.rename(columns={"LIVING_GBA": "GBA"}, inplace=True)

    # Concatenate — missing columns become NaN automatically
    df = pd.concat([df_res, df_com, df_con], ignore_index=True)
    df["SSL"] = df["SSL"].apply(normalise_ssl)
    df["SALEDATE"] = pd.to_datetime(df["SALEDATE"], errors="coerce")
    df["PRICE"] = pd.to_numeric(df["PRICE"], errors="coerce")

    # Keep only qualified sales with a valid price (year filter applied later)
    df = df[df["QUALIFIED"].str.strip() == "Q"]
    df = df[df["PRICE"] > 0]

    print(f"  Lots:        {len(lots):,} rows")
    print(f"  Residential: {len(df[df['PROP_TYPE']=='residential']):,}")
    print(f"  Commercial:  {len(df[df['PROP_TYPE']=='commercial']):,}")
    print(f"  Condominium: {len(df[df['PROP_TYPE']=='condominium']):,}")
    print(f"  Total:       {len(df):,} rows (qualified, price > 0)")
    return lots, df


def load_and_merge(year: int, lots=None, df=None) -> gpd.GeoDataFrame:
    """Merge lots with sales filtered to ±3 months of the collection date."""
    if lots is None or df is None:
        lots, df = _load_raw()

    start, end = sale_window(year)
    df_year = df[(df["SALEDATE"] >= start) & (df["SALEDATE"] <= end)]
    print(f"\n  [{year}] Sale window: {start} → {end}")
    print(f"  [{year}] Qualified sales in window: {len(df_year):,}")

    merged = lots[["SSL", "geometry"]].merge(df_year, on="SSL", how="inner")
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=lots.crs)
    print(f"  [{year}] Merged: {len(merged):,} rows")
    return merged


def merged_path_for_year(year: int) -> Path:
    return PROCESSED_DIR / f"merged_{year}.geojson"


def save_merged(gdf: gpd.GeoDataFrame, year: int) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = merged_path_for_year(year)
    gdf.to_file(out, driver="GeoJSON")
    print(f"  Saved → {out}")


# ── Step 2: Fetch images ─────────────────────────────────────────────────────

def _download_one(task: tuple) -> str:
    """Worker: fetch and save a single image chip. Returns 'saved', 'failed', or 'skipped'."""
    out_path, bounds_wgs84, bounds_m, year = task
    if out_path.exists():
        return "skipped"
    img = fetch_image(bounds_wgs84, bounds_m, year)
    if img is None:
        return "failed"
    img.save(out_path)
    return "saved"


def fetch_images_for_year(gdf_m: gpd.GeoDataFrame, year: int) -> None:
    image_dir = DATA_DIR / "images" / str(year)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute geometry on the main thread (geopandas is not thread-safe)
    tasks = []
    for _, row in gdf_m.iterrows():
        ssl_slug = row["SSL"].replace(" ", "_")
        out_path = image_dir / f"{ssl_slug}.png"
        buffered_m     = row["geometry"].buffer(LOT_BUFFER_M)
        bounds_m       = buffered_m.bounds
        buffered_wgs84 = gpd.GeoSeries([buffered_m], crs=32618).to_crs(4326).iloc[0]
        bounds_wgs84   = buffered_wgs84.bounds
        tasks.append((out_path, bounds_wgs84, bounds_m, year))

    total   = len(tasks)
    saved   = 0
    skipped = 0
    failed  = 0

    print(f"\n  [{year}] Fetching {total:,} image chips → {image_dir}  ({MAX_WORKERS} threads)")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_download_one, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result == "saved":
                saved += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1

            if i % 200 == 0 or i == total:
                print(f"    [{i:>5}/{total}]  saved={saved}  skipped={skipped}  failed={failed}")

    print(f"  [{year}] Done.")


def fetch_all_images(gdf: gpd.GeoDataFrame, years: list[int] = AVAILABLE_YEARS) -> None:
    gdf_m = gdf.to_crs(epsg=32618)
    for year in years:
        fetch_images_for_year(gdf_m, year)
    print(f"\nAll years complete. Images saved under {DATA_DIR / 'images'}/")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DC data preparation pipeline")
    parser.add_argument(
        "--years", nargs="+", type=int, default=AVAILABLE_YEARS,
        help=f"Years to fetch (default: all {AVAILABLE_YEARS})"
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip merging step for years whose merged file already exists"
    )
    args = parser.parse_args()

    # Load raw data once (shared across all years)
    lots, df = _load_raw()

    for year in args.years:
        mp = merged_path_for_year(year)
        if args.skip_merge and mp.exists():
            print(f"\n[{year}] Loading existing {mp.name} …")
            merged = gpd.read_file(mp)
        else:
            merged = load_and_merge(year, lots, df)
            save_merged(merged, year)

        fetch_all_images(merged, years=[year])

    # Zip all images for fast unzip during training
    image_src = DATA_DIR / "images"
    zip_path  = DATA_DIR / "dataset.zip"
    print(f"\nZipping {image_src} → {zip_path} …")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_file in sorted(image_src.rglob("*.png")):
            zf.write(img_file, img_file.relative_to(DATA_DIR))
    print(f"Done. {zip_path.stat().st_size / 1e6:.0f} MB")
