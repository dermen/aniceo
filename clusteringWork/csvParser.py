import csv
import re
from pathlib import Path

INPUT_DIR = Path("/global/cfs/cdirs/m4731/edric/chunkOutputs")
OUTPUT_DIR = Path("/global/cfs/cdirs/m4731/edric/clusteringWork")
OUTPUT_FILE = OUTPUT_DIR / "summary.csv"

# match floats/ints with optional exponent
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

def find_csv_files(root_dir: Path):
    return list(root_dir.rglob("*.csv"))

def parse_reso_field(val) -> list:
    """Return first 4 numeric values from the reso field (padded with None)."""
    if val is None:
        nums = []
    else:
        nums = [float(x) for x in NUMBER.findall(str(val))]
    return (nums + [None, None, None, None])[:4]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = find_csv_files(INPUT_DIR)
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    rows_written = 0
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "file", "chunk", "rank", "shot", "shot_num", "mask", "fname",
            "reso1", "reso2", "reso3", "reso4"
        ])

        for csv_path in sorted(csv_files):
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r1, r2, r3, r4 = parse_reso_field(row.get("reso"))
                    writer.writerow([
                        str(csv_path),
                        row.get("chunk"),
                        row.get("rank"),
                        row.get("shot"),
                        row.get("shot_num"),
                        row.get("mask"),
                        row.get("fname"),
                        r1, r2, r3, r4
                    ])
                    rows_written += 1

    print(f"[summary] Wrote {rows_written} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()