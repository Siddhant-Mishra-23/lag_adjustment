import os
import glob
from datetime import datetime, timedelta
import numpy as np
import h5py
import re
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# --- Folder paths containing raw files ---
SMAP_DIR = r"E:\AHRC_SMAP_L4_20_25\AHRC-Sata_Download"
RAIN_DIR = r"E:\AHRC_SMAP_L4_20_25\Rainfall"
TEMP_DIR = r"E:\AHRC_SMAP_L4_20_25\temperature"

# --- Output ---
OUTPUT_DIR = "./final_output"
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "odisha_merged_tabular.parquet")

# --- Odisha bounding box 
LAT_MIN, LAT_MAX = 17.5, 23.0
LON_MIN, LON_MAX = 81.0, 87.5

def subset_odisha(lats, lons):
    mask = (lats >= LAT_MIN) & (lats <= LAT_MAX) & (lons >= LON_MIN) & (lons <= LON_MAX)
    return mask

def extract_date_from_filename(filepath):
    basename = os.path.basename(filepath)
    # Pattern : YYYYMMDD 
    m = re.search(r'(?<!\d)(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)', basename)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()

    return None

SMAP_CONFIG = {
    "sm_dataset":  "Geophysical_Data/sm_surface",           # AM pass
    "lat_dataset": "cell_lat",
    "lon_dataset": "cell_lon",
    "fill_value":  -9999.0,
}

# --- Rain NetCDF variable names (IMD / GPM / CHIRPS typical names) ---
RAIN_CONFIG = {
    "rain_var": "RAINFALL",          # could be 'precip', 'precipitation', 'rain', 'tp', etc.
    "lat_var":  "LATITUDE",         # could be 'latitude', 'LAT', etc.
    "lon_var":  "LONGITUDE",         # could be 'longitude', 'LON', etc.
    "time_var": "TIME",        # could be 'TIME', 'date', etc.
}

# --- Temperature .grd variable names (IMD / CRU / Berkeley typical names) ---
TEMP_CONFIG = {
    "temp_var": "temp",        # could be 'tmax', 'tmin', 'tmp', 'temperature', etc.
    "lat_var":  "lat",         # could be 'latitude'
    "lon_var":  "lon",         # could be 'longitude'
    "time_var": "time",
}

def read_smap_h5(filepath):  
    file_date = extract_date_from_filename(filepath)
    if file_date is None:
        print(f"  ⚠ Could not parse date from: {os.path.basename(filepath)}, skipping.")
        return pd.DataFrame()

    with h5py.File(filepath, "r") as f:
        sm  = f[SMAP_CONFIG["sm_dataset"]][:]
        lat = f[SMAP_CONFIG["lat_dataset"]][:]
        lon = f[SMAP_CONFIG["lon_dataset"]][:]

    sm = sm.astype(np.float64)
    sm[sm == SMAP_CONFIG["fill_value"]] = np.nan

    # Flatten (SMAP L4 grids are 2D arrays)
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()
    sm_flat  = sm.ravel()

    # Subset to Odisha bounding box
    mask = subset_odisha(lat_flat, lon_flat)
    lat_sub = lat_flat[mask]
    lon_sub = lon_flat[mask]
    sm_sub  = sm_flat[mask]

    # Drop NaN SM values (no-data pixels)
    valid = ~np.isnan(sm_sub)

    df = pd.DataFrame({
        "date":          file_date,
        "lat":           np.round(lat_sub[valid], 4),
        "lon":           np.round(lon_sub[valid], 4),
        "soil_moisture": np.round(sm_sub[valid], 6),
    })

    print(f"  ✓ SMAP  {file_date}  →  {len(df):>6,} pixels")
    return df

def read_all_smap(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.h5"), recursive=True))
    if not files:
        print(f"  ℹ No .h5 files found in {folder}")
        return pd.DataFrame()

    print(f"\n{'─'*60}")
    print(f"  SMAP (.h5)  — {len(files)} file(s) in {folder}")
    print(f"{'─'*60}")
    dfs = [read_smap_h5(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def read_rain_nc(filepath):
    ds = xr.open_dataset(filepath)

    # --- Get variable names from config
    rain_var = RAIN_CONFIG["rain_var"]
    lat_var  = RAIN_CONFIG["lat_var"]
    lon_var  = RAIN_CONFIG["lon_var"]
    time_var = RAIN_CONFIG["time_var"]

    # Fallback auto-detection
    all_vars = list(ds.data_vars) + list(ds.coords)
    if rain_var not in ds.data_vars:
        candidates = [v for v in ds.data_vars if v not in ("lat", "lon", "latitude", "longitude", "time")]
        if candidates:
            rain_var = candidates[0]
            print(f"    ↳ Auto-detected rain variable: '{rain_var}'")

    if lat_var not in ds.coords and lat_var not in ds.dims:
        for candidate in ["latitude", "LAT", "Latitude", "lat"]:
            if candidate in all_vars:
                lat_var = candidate
                break

    if lon_var not in ds.coords and lon_var not in ds.dims:
        for candidate in ["longitude", "LON", "Longitude", "lon"]:
            if candidate in all_vars:
                lon_var = candidate
                break

    if time_var not in ds.coords and time_var not in ds.dims:
        for candidate in ["TIME", "time", "date", "Date"]:
            if candidate in all_vars:
                time_var = candidate
                break

    # --- Subset to Odisha ---
    lats = ds[lat_var].values
    lons = ds[lon_var].values

    # Handle ascending / descending lat
    lat_sel = sorted([LAT_MIN, LAT_MAX])
    lon_sel = sorted([LON_MIN, LON_MAX])

    if lats[0] > lats[-1]:  # descending
        ds_sub = ds.sel(**{lat_var: slice(lat_sel[1], lat_sel[0]),
                           lon_var: slice(lon_sel[0], lon_sel[1])})
    else:
        ds_sub = ds.sel(**{lat_var: slice(lat_sel[0], lat_sel[1]),
                           lon_var: slice(lon_sel[0], lon_sel[1])})
        
    records = []
    for t in ds_sub[time_var].values:
        ts = pd.Timestamp(t).date()
        slice_2d = ds_sub[rain_var].sel(**{time_var: t})
        lats_2d, lons_2d = np.meshgrid(ds_sub[lat_var].values,ds_sub[lon_var].values, indexing="ij")
        rain_flat = slice_2d.values.ravel()
        lat_flat  = lats_2d.ravel()
        lon_flat  = lons_2d.ravel()
        
        valid = np.isfinite(rain_flat)
        records.append(pd.DataFrame({
            "date":     ts,
            "lat":      np.round(lat_flat[valid], 4),
            "lon":      np.round(lon_flat[valid], 4),
            "rainfall": np.round(rain_flat[valid], 4),
            }))
    print(f"  ✓ Rain  {ts}  →  {valid.sum():>6,} pixels")
    ds.close()
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        
def read_all_rain(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.nc"), recursive=True))
    if not files:
        print(f" No .nc files found in {folder}")
        return pd.DataFrame()

    print(f"\n{'─'*60}")
    print(f"  RAIN (.nc)  — {len(files)} file(s) in {folder}")
    print(f"{'─'*60}")
    dfs = [read_rain_nc(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def read_temp_grd(filepath):
    nx, ny = 31, 31
    lon_start, lat_start = 67.5, 7.5
    resolution = 1.0

    # --- Extract year from filename (e.g. "Maxtemp_MaxT_2020.GRD" → 2020) ---
    basename = os.path.basename(filepath)
    year_match = re.search(r'(\d{4})', basename)
    year = int(year_match.group(1))
    start_date = datetime(year, 1, 1)

    filesize = os.path.getsize(filepath)
    n_expected = nx * ny
    n_timesteps = filesize // (n_expected * 4)  # float32 = 4 bytes

    data = np.fromfile(filepath, dtype=np.float32)

    lons = np.arange(lon_start, lon_start + nx * resolution, resolution)[:nx]
    lats = np.arange(lat_start, lat_start + ny * resolution, resolution)[:ny]
    lons_2d, lats_2d = np.meshgrid(lons, lats)

    records = []
    for t in range(n_timesteps):
        offset = t * n_expected
        data_2d = data[offset:offset + n_expected].reshape(ny, nx)

        data_2d[(data_2d > 60) | (data_2d < -60)] = np.nan

        temp_flat = data_2d.ravel()
        lat_flat  = lats_2d.ravel()
        lon_flat  = lons_2d.ravel()

        mask = subset_odisha(lat_flat, lon_flat) & np.isfinite(temp_flat)

        current_date = start_date + timedelta(days=t)

        records.append(pd.DataFrame({
            "date":        current_date,
            "lat":         np.round(lat_flat[mask], 4),
            "lon":         np.round(lon_flat[mask], 4),
            "temperature": np.round(temp_flat[mask], 4),
        }))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    
def read_all_temp(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.grd"), recursive=True) +
                   glob.glob(os.path.join(folder, "**", "*.GRD"), recursive=True))
    if not files:
        print(f"  ℹ No .grd files found in {folder}")
        return pd.DataFrame()

    print(f"\n{'─'*60}")
    print(f"  TEMP (.grd)  — {len(files)} file(s) in {folder}")
    print(f"{'─'*60}")
    dfs = [read_temp_grd(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def nn_fill(df_base, df_other, value_col, tolerance_deg):
    if df_other.empty:
        df_base[value_col] = np.nan
        return df_base

    other_dates = set(df_other["date"].unique())
    results = []

    for date, grp_base in df_base.groupby("date"):
        grp_base = grp_base.copy()

        if date not in other_dates:
            grp_base[value_col] = np.nan
            results.append(grp_base)
            continue

        grp_other = df_other[df_other["date"] == date]

        tree = cKDTree(grp_other[["lat", "lon"]].values)
        dists, idxs = tree.query(grp_base[["lat", "lon"]].values, k=1)

        vals = grp_other[value_col].values[idxs]
        vals[dists > tolerance_deg] = np.nan  # too far → skip
        grp_base[value_col] = vals

        matched = np.sum(dists <= tolerance_deg)
        print(f"    {value_col:<14} {date}  →  {matched:>6,}/{len(grp_base):,} matched")

        results.append(grp_base)

    return pd.concat(results, ignore_index=True)

def merge_and_save(df_sm, df_rain, df_temp, output_path):
    """
    Merge the three datasets on (date, lat, lon) and save to CSV.

    Strategy:
      • Each dataset lives on its own grid (different resolutions).
      • We do an OUTER merge so NO pixels are lost.
      • Where grids don't overlap exactly, the unmatched variable is NaN.
      • For a nearest-neighbor spatial merge, see the optional section below.
    """
    for df in [df_sm, df_rain, df_temp]:
        df["date"] = pd.to_datetime(df["date"])
        df["lat"]  = df["lat"].astype(float)
        df["lon"]  = df["lon"].astype(float)
  
    # Rainfall: IMD 0.25° grid → tolerance ~0.18° (half-diagonal of 0.25° cell)
    merged = nn_fill(df_sm, df_rain, "rainfall", tolerance_deg=0.18)
    print(f"  After rainfall join: {merged['rainfall'].notna().sum():,} matched")

    # Temperature: IMD 1.0° grid → tolerance ~0.72° (half-diagonal of 1.0° cell)
    merged = nn_fill(merged, df_temp, "temperature", tolerance_deg=0.72)
    print(f"  After temperature join: {merged['temperature'].notna().sum():,} matched")

    # Sort
    merged = merged.sort_values(["date", "lat", "lon"],
                                ascending=[True, False, True]).reset_index(drop=True)

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_parquet(output_path, index=False)

    print(f"\n  ✅ Saved: {output_path}")
    print(f"     Rows       : {len(merged):,}")
    print(f"     Columns    : {list(merged.columns)}")
    print(f"     Date range : {merged['date'].min()} → {merged['date'].max()}")
    print(f"     Lat range  : {merged['lat'].min():.4f} → {merged['lat'].max():.4f}")
    print(f"     Lon range  : {merged['lon'].min():.4f} → {merged['lon'].max():.4f}")
    print(f"     Rainfall fill  : {merged['rainfall'].notna().mean()*100:.1f}%")
    print(f"     Temp fill      : {merged['temperature'].notna().mean()*100:.1f}%")

    # Per-variable CSVs
    if not df_sm.empty:
        path = output_path.replace(".parquet", "_soil_moisture.parquet")
        df_sm.sort_values(["date", "lat", "lon"]).to_parquet(path, index=False)
        print(f"     + {path}")
    if not df_rain.empty:
        path = output_path.replace(".parquet", "_rainfall.parquet")
        df_rain.sort_values(["date", "lat", "lon"]).to_parquet(path, index=False)
        print(f"     + {path}")
    if not df_temp.empty:
        path = output_path.replace(".parquet", "_temperature.parquet")
        df_temp.sort_values(["date", "lat", "lon"]).to_parquet(path, index=False)
        print(f"     + {path}")

    return merged


if __name__ == "__main__": 
    df_sm   = read_all_smap(SMAP_DIR)
    df_rain = read_all_rain(RAIN_DIR)
    df_temp = read_all_temp(TEMP_DIR)
    # --- Merge & save ---
    merged = merge_and_save(df_sm, df_rain, df_temp, OUTPUT_PARQUET)


