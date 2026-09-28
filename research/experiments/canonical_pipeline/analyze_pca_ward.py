"""Check whether a 128-dimensional Euclidean semantic view stabilizes Ward."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, write_json)


def _labels(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    pca = PCA(n_components=128, svd_solver="randomized", random_state=seed)
    coordinates = pca.fit_transform(matrix)
    tree = linkage(pdist(coordinates, metric="euclidean"), method="ward")
    return (fcluster(tree, 12, criterion="maxclust"),
            fcluster(tree, 24, criterion="maxclust"),
            float(pca.explained_variance_ratio_.sum()))


def run(manifest_path: Path, clip_path: Path, kec_path: Path,
        temperature_path: Path, output: Path, *, seed: int = 20260928,
        resamples: int = 10) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    resolution = np.asarray([row["width"] == row["height"] == 150
                             for row in representatives(manifest)])
    vectors = {"C0": load_matrix(clip_path, ids),
               "C1_N300": load_matrix(kec_path, ids),
               "C1_N50_t01": load_matrix(temperature_path, ids)}
    rng = np.random.default_rng(seed)
    samples = [np.sort(np.concatenate([
        rng.choice(np.flatnonzero(resolution == value),
                   size=round(np.sum(resolution == value) * 0.8), replace=False)
        for value in (False, True)])) for _ in range(resamples)]
    summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
               "seed": seed, "resamples": resamples, "components": 128,
               "methods": {}}
    for name, matrix in vectors.items():
        full_broad, full_fine, explained = _labels(matrix, seed)
        broad_scores, fine_scores = [], []
        for sample in samples:
            broad, fine, _ = _labels(matrix[sample], seed)
            broad_scores.append(float(adjusted_rand_score(full_broad[sample], broad)))
            fine_scores.append(float(adjusted_rand_score(full_fine[sample], fine)))
        summary["methods"][name] = {
            "full_explained_variance": explained,
            "broad_ari_mean": float(np.mean(broad_scores)),
            "fine_ari_mean": float(np.mean(fine_scores)),
            "broad_ari_p10_p90": np.percentile(broad_scores, [10, 90]).tolist(),
            "fine_ari_p10_p90": np.percentile(fine_scores, [10, 90]).tolist(),
            "largest_broad": int(np.bincount(full_broad)[1:].max()),
            "largest_fine": int(np.bincount(full_fine)[1:].max()),
        }
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--kec", type=Path, required=True)
    parser.add_argument("--temperature", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10)
    args = parser.parse_args()
    result = run(args.manifest, args.clip, args.kec, args.temperature,
                 args.output, resamples=args.resamples)
    for name, row in result["methods"].items():
        print(name, "ARI@12", round(row["broad_ari_mean"], 3),
              "ARI@24", round(row["fine_ari_mean"], 3),
              "variance", round(row["full_explained_variance"], 3))


if __name__ == "__main__":
    main()
