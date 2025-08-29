from pathlib import Path
import re
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths ---
BASE = Path("/global/cfs/cdirs/m4731/edric")
CLUSTER_CSV_DIR = BASE / "clustersMaxComp" / "clusterCSVs"
OUT_DIR = BASE / "clustersMaxComp" / "clusterIMGs"

# NOTE: federated_training lives under /global/cfs/cdirs/m4731/, not under .../edric/
FED_TRAIN_DIR = Path("/global/cfs/cdirs/m4731/federated_training")  # contains chunk0, chunk1, ...

LETTERS = list("ABCDEFG")  # A..G

def top_row(csv_path: Path) -> pd.Series | None:
    if not csv_path.exists():
        print(f"[WARN] missing CSV: {csv_path}")
        return None
    df = pd.read_csv(csv_path, nrows=1)
    if df.empty:
        print(f"[INFO] empty CSV: {csv_path}")
        return None
    return df.iloc[0]

def chunk_id_from_file(res_csv_path: str) -> str | None:
    m = re.search(r"resolution_chunk(\d+)\.csv$", str(res_csv_path))
    return m.group(1) if m else None

def find_h5_in_federated_training(chunk_id: str) -> Path | None:
    """
    Look under federated_training/chunk{chunk_id}/ for any .h5 file.
    If multiple exist, pick the largest.
    """
    chunk_dir = FED_TRAIN_DIR / f"chunk{chunk_id}"
    if not chunk_dir.exists():
        print(f"[WARN] {chunk_dir} not found")
        return None
    hits = glob.glob(str(chunk_dir / "**/*.h5"), recursive=True)
    if not hits:
        print(f"[WARN] no .h5 under {chunk_dir}")
        return None
    hits = [Path(h) for h in hits]
    hits.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return hits[0]

def load_one_shot_image(shot_group: h5py.Group) -> np.ndarray:
    """
    Replicates: np.hstack(shot['ice_mask_False/pixels8'][:,0])
    with a minimal fallback to _True if needed.
    """
    for path in ("ice_mask_False/pixels8", "ice_mask_True/pixels8"):
        if path in shot_group:
            ds = shot_group[path]
            return np.hstack(ds[:, 0])  # will raise if shape unexpected
    raise RuntimeError("neither 'ice_mask_False/pixels8' nor 'ice_mask_True/pixels8' present")

def make_max_comp(h5_path: Path, vmax: float = 30.0) -> np.ndarray:
    print(f"[INFO] opening {h5_path}")
    imgs = []
    with h5py.File(h5_path, "r") as h:
        for s in h.keys():
            try:
                img = load_one_shot_image(h[s])
                imgs.append(img)
            except Exception as e:
                print(f"  [WARN] skip {s}: {e}")
    if not imgs:
        raise RuntimeError("no images extracted from H5")
    return np.max(np.stack(imgs, axis=0), axis=0)

def save_png(img: np.ndarray, out_path: Path, vmax: float = 30.0, dpi: int = 150):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(img, vmin=0, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[OK] wrote {out_path}")

def process_cluster(letter: str):
    csv_path = CLUSTER_CSV_DIR / f"cluster{letter}.csv"
    row = top_row(csv_path)
    if row is None:
        return
    res_csv_path = row["file"] if "file" in row.index else row.iloc[0]
    cid = chunk_id_from_file(str(res_csv_path))
    if not cid:
        print(f"[WARN] cannot parse chunk id from: {res_csv_path}")
        return
    h5 = find_h5_in_federated_training(cid)
    if not h5:
        return
    img = make_max_comp(h5, vmax=30.0)
    out_png = OUT_DIR / f"cluster_{letter}_top.png"
    save_png(img, out_png, vmax=30.0, dpi=150)

if __name__ == "__main__":
    print(f"[INFO] reading from {CLUSTER_CSV_DIR}")
    print(f"[INFO] writing PNGs to {OUT_DIR}")
    for L in LETTERS:
        print(f"\n=== Cluster {L} ===")
        process_cluster(L)
    print("\n[DONE]")