import pandas as pd
from pathlib import Path

# --- Inputs/paths ---
in_csv = "/global/cfs/cdirs/m4731/edric/clusteringWork/summary.csv"
out_dir = Path("/global/cfs/cdirs/m4731/edric/clustersMaxComp/clusterCSVs")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Data prep ---
reso_cols = ["reso1", "reso2", "reso3", "reso4"]

df = pd.read_csv(in_csv)

# Min/max across the resolution columns
df["min_res"] = df[reso_cols].min(axis=1)
df["max_res"] = df[reso_cols].max(axis=1)
df["spread"]  = df["max_res"] - df["min_res"]
df["which_min"] = df[reso_cols].idxmin(axis=1)
df["which_max"] = df[reso_cols].idxmax(axis=1)

# Optional mask handling (uses 'icemask' if present, else 'mask', else no mask)
mask_col = "icemask" if "icemask" in df.columns else ("mask" if "mask" in df.columns else None)
if mask_col:
    # accept True or 1 as "masked in"
    mask_filter = (df[mask_col] == True) | (df[mask_col] == 1)
else:
    mask_filter = pd.Series(True, index=df.index)

# Columns to keep in outputs
base_cols = [c for c in ["file", "fname", "shot", "shot_num", mask_col] if c and c in df.columns]
cols_to_keep = base_cols + reso_cols + ["min_res", "max_res", "spread", "which_min", "which_max"]

# --- Cluster definitions (from your screenshot) ---
clusters = {
    "A": {"min": (1.02, 1.16), "max": (1.06, 1.22)},
    "B": {"min": (1.12, 1.24), "max": (1.26, 1.38)},
    "C": {"min": (1.25, 1.45), "max": (1.35, 1.65)},
    "D": {"min": (1.50, 1.80), "max": (1.78, 1.94)},
    "E": {"min": (2.00, 2.60), "max": (2.94, 3.15)},
    "F": {"min": (2.94, 3.02), "max": (3.20, 4.00)},
    "G": {"min": (1.30, 1.80), "max": (2.75, 4.75)},
}

# --- Produce one CSV per cluster, sorted by spread desc ---
for name, bounds in clusters.items():
    min_lo, min_hi = bounds["min"]
    max_lo, max_hi = bounds["max"]

    sel = (
        df["min_res"].between(min_lo, min_hi, inclusive="both") &
        df["max_res"].between(max_lo, max_hi, inclusive="both") &
        mask_filter
    )

    hits = df.loc[sel, cols_to_keep].copy()
    hits = hits.sort_values("spread", ascending=False)

    out_path = out_dir / f"cluster{name}.csv"
    hits.to_csv(out_path, index=False)
    print(f"Cluster {name}: {len(hits)} rows -> {out_path}")