import os, glob, re, h5py, pandas as pd

inputDirectory = "/global/cfs/cdirs/m4731/federated_training"
outputDirectory = "/global/cfs/cdirs/m4731/edric/chunkOutputs"

records = []

fnames = glob.glob(os.path.join(inputDirectory, "chunk*/rank*h5"))
for f in fnames:
    chunk_match = re.search(r"chunk(\d+)", f)
    rank_match = re.search(r"rank(\d+)", f)
    chunk_num = int(chunk_match.group(1)) if chunk_match else None
    rank_num = int(rank_match.group(1)) if rank_match else None

    with h5py.File(f, 'r') as h:
        for shot in h.keys():
            shot_str = str(shot)
            m = re.search(r"(\d+)$", shot_str)
            shot_num = int(m.group(1)) if m else None

            g = h[shot]

            fname_attr = g.attrs.get("fname", None)
            if isinstance(fname_attr, (bytes, bytearray)):
                fname_attr = fname_attr.decode("utf-8", "ignore")

            ds_false = g.get("ice_mask_False/reso")
            ds_true  = g.get("ice_mask_True/reso")

            if ds_false is None or ds_true is None:
                continue

            for mask_val, dataset_name in [(False, "ice_mask_False/reso"),
                                           (True,  "ice_mask_True/reso")]:
                res = g[dataset_name][()]
                try:
                    res = res.item()
                except Exception:
                    pass

                base = dataset_name.rsplit("/", 1)[0]
                geom_path = f"{base}/geom"
                detdist = pixelsize = wavelength = None
                geom_ds = g.get(geom_path)
                if geom_ds is not None:
                    geom_vals = geom_ds[()]
                    flat = getattr(geom_vals, "ravel", lambda: geom_vals)()
                    if hasattr(flat, "size"):
                        detdist   = float(flat[0]) if flat.size >= 1 else None
                        pixelsize = float(flat[1]) if flat.size >= 2 else None
                        wavelength= float(flat[2]) if flat.size >= 3 else None

                records.append({
                    "file": f,
                    "chunk": chunk_num,
                    "rank": rank_num,
                    "shot": shot_str,
                    "shot_num": shot_num,
                    "mask": mask_val,
                    "reso": res,
                    "detector_distance": detdist,
                    "pixel_size": pixelsize,
                    "wavelength": wavelength,
                    "fname": fname_attr
                })

df = pd.DataFrame(records)
df.sort_values(["chunk", "rank", "shot_num", "mask"], inplace=True)
print(df.head(10))

for chunk_val, grp in df.groupby("chunk", sort=True):
    fn = os.path.join(outputDirectory, f"resolution_chunk{chunk_val}.csv")
    grp.to_csv(fn, index=False)
    print("Wrote:", fn)
