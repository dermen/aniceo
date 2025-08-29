import pandas as pd

df = pd.read_csv("/global/cfs/cdirs/m4731/edric/clusteringWork/summary.csv")

reso_cols = ["reso1", "reso2", "reso3", "reso4"]
df["min_res"] = df[reso_cols].min(axis=1)
df["max_res"] = df[reso_cols].max(axis=1)

min_lo, min_hi = 3.5, 5.00
max_lo, max_hi = 10.00, 12.00 

# mask_col = "icemask" if "icemask" in df.columns else "mask"
# mask_filter = (df[mask_col] == True)
# Otherwise:
mask_filter = True

sel = (
    df["min_res"].between(min_lo, min_hi, inclusive="both") &
    df["max_res"].between(max_lo, max_hi, inclusive="both") &
    mask_filter
)
hits = df.loc[sel].copy()

hits["which_min"] = df[reso_cols].idxmin(axis=1)
hits["which_max"] = df[reso_cols].idxmax(axis=1)

mask_col = "icemask" if "icemask" in hits.columns else ("mask" if "mask" in hits.columns else None)
base_cols = ["file", "fname", "shot", "shot_num"]
if mask_col: base_cols.append(mask_col)
cols_to_keep = [c for c in base_cols if c in hits.columns] + reso_cols + ["min_res","max_res","which_min","which_max"]

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(f"Matches: {len(hits)} rows (min in [{min_lo}, {min_hi}], max in [{max_lo}, {max_hi}])")
print(hits[cols_to_keep].head(50).to_string(index=False))  # preview first 50 rows

out_path = "rows_between_min_and_max_ranges.csv"
hits[cols_to_keep].to_csv(out_path, index=False)
print(f"Saved all matches to: {out_path}")