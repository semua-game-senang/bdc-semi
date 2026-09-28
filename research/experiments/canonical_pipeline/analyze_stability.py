"""Stratified resampling stability of matched semantic hierarchies."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, write_json)


def _cuts(matrix: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    metric = "euclidean" if method == "ward" else "cosine"
    tree = linkage(pdist(matrix, metric=metric), method=method)
    return (fcluster(tree, 12, criterion="maxclust"),
            fcluster(tree, 24, criterion="maxclust"))


def run(manifest_path: Path, clip_path: Path, kec_path: Path,
        temperature_path: Path, output: Path, *, seed: int = 20260928,
        resamples: int = 20, methods: tuple[str, ...] = ("average", "complete")) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    rows = representatives(manifest)
    resolution = np.asarray([row["width"] == row["height"] == 150 for row in rows])
    vectors = {"C0": load_matrix(clip_path, ids),
               "C1_N300": load_matrix(kec_path, ids),
               "C1_N50_t01": load_matrix(temperature_path, ids)}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        sample = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(resolution == value),
                       size=round(np.sum(resolution == value) * 0.8), replace=False)
            for value in (False, True)]))
        samples.append(sample)
    summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
               "seed": seed, "resamples": resamples, "fraction": 0.8,
               "methods": {}}
    for method in methods:
        summary["methods"][method] = {}
        for name, matrix in vectors.items():
            full = _cuts(matrix, method)
            scores = {12: [], 24: []}
            largest = {12: [], 24: []}
            for sample in samples:
                fitted = _cuts(matrix[sample], method)
                for index, cut in enumerate((12, 24)):
                    scores[cut].append(float(adjusted_rand_score(
                        full[index][sample], fitted[index])))
                    largest[cut].append(int(np.bincount(fitted[index])[1:].max()))
            summary["methods"][method][name] = {
                str(cut): {"ari_mean": float(np.mean(scores[cut])),
                           "ari_p10_p90": np.percentile(scores[cut], [10, 90]).tolist(),
                           "largest_mean": float(np.mean(largest[cut]))}
                for cut in (12, 24)}
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--kec", type=Path, required=True)
    parser.add_argument("--temperature", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linkage", choices=("average", "complete", "ward"),
                        help="run one linkage rule; default compares average and complete")
    args = parser.parse_args()
    result = run(args.manifest, args.clip, args.kec, args.temperature, args.output,
                 methods=(args.linkage,) if args.linkage else ("average", "complete"))
    for method, methods in result["methods"].items():
        for name, cuts in methods.items():
            print(method, name, "ARI@12", round(cuts["12"]["ari_mean"], 3),
                  "ARI@24", round(cuts["24"]["ari_mean"], 3))


if __name__ == "__main__":
    main()
