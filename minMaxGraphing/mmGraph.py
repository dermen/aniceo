import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/global/cfs/cdirs/m4731/edric/clusteringWork/summary.csv")

mask_col = "icemask" if "icemask" in df.columns else "mask"

reso_cols = ["reso1", "reso2", "reso3", "reso4"]
df["min_res"] = df[reso_cols].min(axis=1)
df["max_res"] = df[reso_cols].max(axis=1)

plt.figure(figsize=(8, 6))
for val, part in df.groupby(mask_col):
    plt.scatter(
        part["min_res"], part["max_res"],
        label=f"{mask_col}={val}",
        s=3,    
        linewidths=0,   
        alpha=0.4,
    )

lo = min(df["min_res"].min(), df["max_res"].min())
hi = max(df["min_res"].max(), df["max_res"].max())
plt.plot([lo, hi], [lo, hi])

plt.xlabel("Min of reso1–reso4 (Å)")
plt.ylabel("Max of reso1–reso4 (Å)")
plt.title("Per-row Min vs Max Resolution by icemask")
plt.legend(title=mask_col)
plt.tight_layout()
plt.savefig("min_vs_max_by_icemask.png", dpi=200, bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(8, 6))
ax = plt.gca()

extent = [
    df["min_res"].min(), df["min_res"].max(),
    df["max_res"].min(), df["max_res"].max()
]

hb = ax.hexbin(
    df["min_res"], df["max_res"],
    gridsize=1000,             
    cmap="gnuplot",           
    mincnt=1,                 
    bins="log",             
    extent=extent
)
fig.colorbar(hb, ax=ax, label="log10(count)")

ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.plot([lo, hi], [lo, hi])
ax.set_xlabel("Min of reso1–reso4 (Å)")
ax.set_ylabel("Max of reso1–reso4 (Å)")
ax.set_title("Hexbin: Min vs Max Resolution (all points)")
plt.tight_layout()
plt.savefig("hexbin_min_vs_max_all.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

groups = list(df.groupby(mask_col))
n = len(groups)
if n >= 1:
    fig, axes = plt.subplots(1, n, figsize=(12, 5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (val, part) in zip(axes, groups):
        hb = ax.hexbin(
            part["min_res"], part["max_res"],
            gridsize=1000,
            cmap="gnuplot",
            mincnt=1,
            bins="log",
            extent=extent
        )
        ax.plot([lo, hi], [lo, hi])
        ax.set_title(f"{mask_col}={val}")
        ax.set_xlabel("Min (Å)")
        ax.set_ylabel("Max (Å)")
    plt.tight_layout()
    plt.savefig("hexbin_min_vs_max_by_icemask.png", dpi=200, bbox_inches="tight")
    plt.close()

print("Saved:")
print(" - min_vs_max_by_icemask.png")
print(" - hexbin_min_vs_max_all.png")
print(" - hexbin_min_vs_max_by_icemask.png")