import os, glob, re, h5py

inputDirectory = "/global/cfs/cdirs/m4731/federated_training"

fnames = glob.glob(os.path.join(inputDirectory, "chunk*/rank*h5"))
for f in fnames:
    chunk_match = re.search(r"chunk(\d+)", f)
    rank_match = re.search(r"rank(\d+)", f)
    chunk_num = int(chunk_match.group(1)) if chunk_match else None
    rank_num = int(rank_match.group(1)) if rank_match else None

    with h5py.File(f, 'r') as h:
        for shot in h.keys():
            shot_str = str(shot)

            g = h[shot]
            ds_false = g.get("ice_mask_False/reso")
            ds_true  = g.get("ice_mask_True/reso")

            if ds_false is None:
                print(f"Missing ice_mask_False/reso in chunk {chunk_num}, rank {rank_num}, shot {shot_str}")
            if ds_true is None:
                print(f"Missing ice_mask_True/reso in chunk {chunk_num}, rank {rank_num}, shot {shot_str}")