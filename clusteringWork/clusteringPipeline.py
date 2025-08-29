import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_SUMMARY = Path("/global/cfs/cdirs/m4731/edric/clusteringWork/summary.csv")
DEFAULT_OUTDIR  = Path("/global/cfs/cdirs/m4731/edric/clusteringWork")

def load_X(summary_file: Path, target_dim: int = 4) -> np.ndarray:
    import csv
    import numpy as np

    data = []
    with summary_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        fields = reader.fieldnames or []
        if any(fn.startswith("reso") for fn in fields):
            cols = [f"reso{i+1}" for i in range(target_dim) if f"reso{i+1}" in fields]
        else:
            cols = [f"val{i+1}" for i in range(target_dim) if f"val{i+1}" in fields]

        if len(cols) < target_dim:
            raise RuntimeError(
                f"Requested {target_dim} dims but only found columns: {cols}"
            )

        for r in reader:
            try:
                vals = []
                ok = True
                for c in cols:
                    s = r.get(c, "")
                    if s is None or s.strip() == "":
                        ok = False
                        break
                    vals.append(float(s))
                if ok and len(vals) == 4:
                    v1, v2, v3, v4 = vals
                    #vertical, horizontal, axis preference, resolution range and standard deviation
                    v5 = abs(v1 - v3)         
                    v6 = abs(v2 - v4)        
                    v7 = (v1 + v3) - (v2 + v4)  
                    v8 = max(vals) - min(vals) 
                    v9 = np.std(vals)
                    more_vals = vals + [v5, v6, v7, v8, v9]
                    data.append(more_vals)
                elif ok:
                    data.append(vals)
            except (ValueError, TypeError):
                continue
    X = np.asarray(data, dtype=float)
    if X.size == 0:
        raise RuntimeError(f"No valid rows loaded from {summary_file}")
    print(f"[load] X shape: {X.shape}")
    return X

def make_elbow_plot(X_scaled: np.ndarray, out_dir: Path, kmin: int, kmax: int):
    ks = list(range(kmin, kmax + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"[elbow] k={k} inertia={km.inertia_:.4f}")

    # Save CSV of inertias
    csv_path = out_dir / "elbow_inertia.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "inertia"])
        for k, inn in zip(ks, inertias):
            w.writerow([k, inn])
    print(f"[elbow] Inertia CSV -> {csv_path}")

    # Plot elbow
    fig = plt.figure(figsize=(8, 6))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow Method")
    plt.grid(True)
    elbow_path = out_dir / "elbow_plot.png"
    fig.tight_layout()
    fig.savefig(elbow_path, dpi=150)
    plt.close(fig)
    print(f"[elbow] Plot -> {elbow_path}")

def run_kmeans_pca(X: np.ndarray, out_dir: Path, k: int, summary_file: Path = Path("/global/cfs/cdirs/m4731/edric/clusteringWork/summary.csv")):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    #from IPython import embed; embed()
    #X_pca = np.vstack((X.mean(1), X.std(1))).T
    #X_pca = np.vstack((X.min(1), X.max(1))).T
    #from IPython import embed; embed()

    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(X_pca[:, 0], X_pca[:, 1], gridsize=1000, cmap='gnuplot', mincnt=1, bins='log')
    plt.savefig(out_dir / "clusters_pca_hexbin.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.7, s = 1)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"KMeans (k={k}) on summary.csv — PCA projection")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [f"Cluster {i}" for i in range(k)], title="Clusters")
    plot_path = out_dir / "clusters_pca.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"[plot] PCA cluster plot -> {plot_path}")

    if summary_file is not None:
        original_data = []
        row_indices = []
        with summary_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_idx = 0
            
            fields = reader.fieldnames or []
            if any(fn.startswith("reso") for fn in fields):
                cols = [f"reso{i+1}" for i in range(4) if f"reso{i+1}" in fields]
            else:
                cols = [f"val{i+1}" for i in range(4) if f"val{i+1}" in fields]
            
            for r in reader:
                try:
                    vals = []
                    ok = True
                    for c in cols:
                        s = r.get(c, "")
                        if s is None or s.strip() == "":
                            ok = False
                            break
                        vals.append(float(s))
                    
                    if ok and len(vals) == 4:
                        original_data.append(r)
                        row_indices.append(row_idx)
                except (ValueError, TypeError):
                    pass
                row_idx += 1

        # Create comprehensive mapping CSV
        mapping_csv = out_dir / "cluster_mapping.csv"
        with mapping_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["original_row", "pc1", "pc2", "cluster", "reso1", "reso2", "reso3", "reso4", 
                       "vertical_aniso", "horizontal_aniso", "axis_preference", "resolution_range", 
                       "resolution_std"])
            
            for i, (original_row, pc_coords, cluster_label, features) in enumerate(zip(row_indices, X_pca, labels, X)):
                w.writerow([original_row, pc_coords[0], pc_coords[1], cluster_label] + list(features))
        
        print(f"[mapping] Cluster mapping -> {mapping_csv}")

    pca_csv = out_dir / "pca_coords.csv"
    with pca_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc1", "pc2", "cluster"])
        for (pc1, pc2), lab in zip(X_pca, labels):
            w.writerow([pc1, pc2, lab])
    print(f"[data] PCA coords -> {pca_csv}")

    centers_scaled = kmeans.cluster_centers_
    centers_unscaled = scaler.inverse_transform(centers_scaled)
    centers_csv = out_dir / "cluster_centers.csv"
    with centers_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "center_val1", "center_val2", "center_val3", "center_val4"])
        for idx, row in enumerate(centers_unscaled):
            w.writerow([idx] + list(row))
    print(f"[data] Cluster centers -> {centers_csv}")

    ev_csv = out_dir / "pca_explained_variance.csv"
    with ev_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, r in enumerate(pca.explained_variance_ratio_, start=1):
            w.writerow([f"PC{i}", r])
    print(f"[data] PCA explained variance -> {ev_csv}")

def run_dbscan_pca(
    X: np.ndarray,
    out_dir: Path,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    summary_file: Path = DEFAULT_SUMMARY,
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[dbscan] clusters={n_clusters} (noise label = -1 present={-1 in labels})")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # hexbin background
    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(X_pca[:, 0], X_pca[:, 1], gridsize=1000, cmap='gnuplot', mincnt=1, bins='log')
    plt.savefig(out_dir / "dbscan_clusters_pca_hexbin.png", dpi=150)
    plt.close(fig)

    # colored scatter by labels (noise=-1)
    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.7, s=1)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"DBSCAN (eps={eps}, min_samples={min_samples}) — PCA projection")
    # Build legend with noise handled
    unique_labs = sorted(set(labels))
    handles, _ = scatter.legend_elements()
    legend_labels = []
    # matplotlib builds handles in order of appearance, but we want clear names:
    # We'll regenerate legend entries explicitly:
    plt.legend([], [], frameon=False)  # clear auto legend
    # make custom legend
    legend_entries = []
    for lab in unique_labs:
        mask = labels == lab
        pt = plt.scatter([], [], c=[lab], s=10)
        legend_entries.append(pt)
        legend_labels.append("Noise" if lab == -1 else f"Cluster {lab}")
    plt.legend(legend_entries, legend_labels, title="Labels", scatterpoints=1)
    plot_path = out_dir / "dbscan_clusters_pca.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"[plot] DBSCAN PCA cluster plot -> {plot_path}")

    # Mapping CSV (same columns as kmeans for consistency)
    if summary_file is not None:
        original_data = []
        row_indices = []
        with summary_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_idx = 0
            fields = reader.fieldnames or []
            if any(fn.startswith("reso") for fn in fields):
                cols = [f"reso{i+1}" for i in range(4) if f"reso{i+1}" in fields]
            else:
                cols = [f"val{i+1}" for i in range(4) if f"val{i+1}" in fields]

            for r in reader:
                try:
                    vals = []
                    ok = True
                    for c in cols:
                        s = r.get(c, "")
                        if s is None or s.strip() == "":
                            ok = False
                            break
                        vals.append(float(s))
                    if ok and len(vals) == 4:
                        original_data.append(r)
                        row_indices.append(row_idx)
                except (ValueError, TypeError):
                    pass
                row_idx += 1

        mapping_csv = out_dir / "dbscan_cluster_mapping.csv"
        with mapping_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "original_row", "pc1", "pc2", "cluster",
                "reso1", "reso2", "reso3", "reso4",
                "vertical_aniso", "horizontal_aniso", "axis_preference", "resolution_range",
                "resolution_std"
            ])
            for i, (original_row, pc_coords, cluster_label, features) in enumerate(zip(row_indices, X_pca, labels, X)):
                w.writerow([original_row, pc_coords[0], pc_coords[1], cluster_label] + list(features))
        print(f"[mapping] DBSCAN cluster mapping -> {mapping_csv}")

    # PCA coords
    pca_csv = out_dir / "dbscan_pca_coords.csv"
    with pca_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc1", "pc2", "cluster"])
        for (pc1, pc2), lab in zip(X_pca, labels):
            w.writerow([pc1, pc2, lab])
    print(f"[data] DBSCAN PCA coords -> {pca_csv}")

    # Centroids (means) per cluster in original feature space (exclude noise)
    centers_csv = out_dir / "dbscan_cluster_centers.csv"
    with centers_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["cluster"] + [f"center_feature_{i+1}" for i in range(X.shape[1])]
        w.writerow(header)
        for lab in sorted(set(labels)):
            if lab == -1:
                continue
            mask = labels == lab
            center = X[mask].mean(axis=0)
            w.writerow([lab] + list(center))
    print(f"[data] DBSCAN cluster centers (means) -> {centers_csv}")

def run_agglomerative_pca(
    X: np.ndarray,
    out_dir: Path,
    k: int,
    linkage: str = "ward",
    metric: str = "euclidean",
    summary_file: Path = DEFAULT_SUMMARY,
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ward requires Euclidean; sklearn >=1.2 uses 'metric', older 'affinity'
    if linkage == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)

    labels = model.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # hexbin background
    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(X_pca[:, 0], X_pca[:, 1], gridsize=1000, cmap='gnuplot', mincnt=1, bins='log')
    plt.savefig(out_dir / "agglomerative_clusters_pca_hexbin.png", dpi=150)
    plt.close(fig)

    # colored scatter
    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.7, s=1)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Agglomerative (k={k}, linkage={linkage}, metric={metric if linkage!='ward' else 'euclidean'}) — PCA projection")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [f"Cluster {i}" for i in range(k)], title="Clusters")
    plot_path = out_dir / "agglomerative_clusters_pca.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"[plot] Agglomerative PCA cluster plot -> {plot_path}")

    # Mapping CSV (same columns as kmeans for consistency)
    if summary_file is not None:
        original_data = []
        row_indices = []
        with summary_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_idx = 0
            fields = reader.fieldnames or []
            if any(fn.startswith("reso") for fn in fields):
                cols = [f"reso{i+1}" for i in range(4) if f"reso{i+1}" in fields]
            else:
                cols = [f"val{i+1}" for i in range(4) if f"val{i+1}" in fields]

            for r in reader:
                try:
                    vals = []
                    ok = True
                    for c in cols:
                        s = r.get(c, "")
                        if s is None or s.strip() == "":
                            ok = False
                            break
                        vals.append(float(s))
                    if ok and len(vals) == 4:
                        original_data.append(r)
                        row_indices.append(row_idx)
                except (ValueError, TypeError):
                    pass
                row_idx += 1

        mapping_csv = out_dir / "agglomerative_cluster_mapping.csv"
        with mapping_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "original_row", "pc1", "pc2", "cluster",
                "reso1", "reso2", "reso3", "reso4",
                "vertical_aniso", "horizontal_aniso", "axis_preference", "resolution_range",
                "resolution_std"
            ])
            for i, (original_row, pc_coords, cluster_label, features) in enumerate(zip(row_indices, X_pca, labels, X)):
                w.writerow([original_row, pc_coords[0], pc_coords[1], cluster_label] + list(features))
        print(f"[mapping] Agglomerative cluster mapping -> {mapping_csv}")

    # PCA coords
    pca_csv = out_dir / "agglomerative_pca_coords.csv"
    with pca_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc1", "pc2", "cluster"])
        for (pc1, pc2), lab in zip(X_pca, labels):
            w.writerow([pc1, pc2, lab])
    print(f"[data] Agglomerative PCA coords -> {pca_csv}")

    # Centroids (means) per cluster in original feature space
    centers_csv = out_dir / "agglomerative_cluster_centers.csv"
    with centers_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["cluster"] + [f"center_feature_{i+1}" for i in range(X.shape[1])]
        w.writerow(header)
        for lab in sorted(set(labels)):
            mask = labels == lab
            center = X[mask].mean(axis=0)
            w.writerow([lab] + list(center))
    print(f"[data] Agglomerative cluster centers (means) -> {centers_csv}")

def main():
    ap = argparse.ArgumentParser(description="Elbow + optional clustering/PCA on summary.csv")

    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Path to summary.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTDIR, help="Output directory")

    # method selection
    ap.add_argument(
        "--method",
        choices=["kmeans", "dbscan", "agglomerative"],
        default="kmeans",
        help="Clustering method to run",
    )

    # shared / kmeans & agglomerative
    ap.add_argument("--k", type=int, default=None, help="Number of clusters for KMeans/Agglomerative")

    # elbow for kmeans only
    ap.add_argument("--kmin", type=int, default=2, help="Min k for elbow (KMeans only)")
    ap.add_argument("--kmax", type=int, default=10, help="Max k for elbow (KMeans only)")

    # DBSCAN params
    ap.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps")
    ap.add_argument("--min-samples", type=int, default=5, dest="min_samples", help="DBSCAN min_samples")
    ap.add_argument("--metric", type=str, default="euclidean", help="Distance metric (DBSCAN/Agglomerative; ward forces euclidean)")

    # Agglomerative params
    ap.add_argument(
        "--linkage",
        choices=["ward", "complete", "average", "single"],
        default="ward",
        help="Agglomerative linkage",
    )

    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X = load_X(args.summary, target_dim=4)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow is only meaningful for KMeans
    if args.method == "kmeans":
        make_elbow_plot(X_scaled, out_dir, args.kmin, args.kmax)

    # Run the chosen clustering method
    if args.method == "kmeans":
        if args.k is not None:
            run_kmeans_pca(X, out_dir, args.k, args.summary)
        else:
            print("[info] No --k provided for KMeans. Elbow plot generated; choose a K and re-run with --k.")
    elif args.method == "dbscan":
        run_dbscan_pca(
            X,
            out_dir,
            eps=args.eps,
            min_samples=args.min_samples,
            metric=args.metric,
            summary_file=args.summary,
        )
    elif args.method == "agglomerative":
        if args.k is None:
            print("[info] Agglomerative requires --k. Please provide it and re-run.")
        else:
            run_agglomerative_pca(
                X,
                out_dir,
                k=args.k,
                linkage=args.linkage,
                metric=args.metric,
                summary_file=args.summary,
            )

if __name__ == "__main__":
    main()