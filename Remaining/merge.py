def merge_and_save(df_sm, df_rain, df_temp, output_path):
    """
    Merge strategy:
      1. Use SMAP (soil moisture) as the base grid (finest resolution ~9km).
      2. For each date, spatially match rainfall & temperature grid points
         to the nearest SMAP pixel using cKDTree.
      3. This avoids NaN bloat from mismatched grids.
    """
    from scipy.spatial import cKDTree

    print(f"\n{'═'*60}")
    print("  MERGING DATASETS (Nearest-Neighbor Spatial Match)")
    print(f"{'═'*60}")

    # Standardize column types
    for df in [df_sm, df_rain, df_temp]:
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["lat"]  = df["lat"].astype(float)
            df["lon"]  = df["lon"].astype(float)

    if df_sm.empty:
        print("  ⚠ SMAP (base grid) is empty. Cannot merge.")
        return pd.DataFrame()

    # --- Nearest-neighbor helper ---
    def nn_fill(df_base, df_other, value_col, tolerance_deg):
        """
        For each date in df_base, find the nearest point in df_other
        and attach value_col. Points beyond tolerance_deg → NaN.
        """
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

    # --- Build merged dataset on SMAP base grid ---
    print(f"\n  Base grid (SMAP): {len(df_sm):,} rows")

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
    merged.to_csv(output_path, index=False)

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
        path = output_path.replace(".csv", "_soil_moisture.csv")
        df_sm.sort_values(["date", "lat", "lon"]).to_csv(path, index=False)
        print(f"     + {path}")
    if not df_rain.empty:
        path = output_path.replace(".csv", "_rainfall.csv")
        df_rain.sort_values(["date", "lat", "lon"]).to_csv(path, index=False)
        print(f"     + {path}")
    if not df_temp.empty:
        path = output_path.replace(".csv", "_temperature.csv")
        df_temp.sort_values(["date", "lat", "lon"]).to_csv(path, index=False)
        print(f"     + {path}")

    return merged