"""Find reproducible hierarchy branches without assigning semantic labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, write_json)


def _labels(matrix: np.ndarray, count: int, method: str) -> np.ndarray:
    tree = linkage(pdist(matrix, metric="cosine"), method=method)
    return fcluster(tree, count, criterion="maxclust")


def analyze(manifest_path: Path, clip_path: Path, kec_path: Path,
            output: Path, *, count: int = 24, resamples: int = 20,
            seed: int = 20260928) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    rows = representatives(manifest)
    resolution = np.asarray([row["width"] == row["height"] == 150 for row in rows])
    vectors = {"C0": load_matrix(clip_path, ids),
               "C1_N300": load_matrix(kec_path, ids)}
    rng = np.random.default_rng(seed)
    samples = [np.sort(np.concatenate([
        rng.choice(np.flatnonzero(resolution == value),
                   size=round(np.sum(resolution == value) * 0.8), replace=False)
        for value in (False, True)])) for _ in range(resamples)]
    summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
               "count": count, "resamples": resamples, "fraction": 0.8,
               "seed": seed, "methods": {}}
    for method in ("average", "complete"):
        summary["methods"][method] = {}
        for name, matrix in vectors.items():
            full = _labels(matrix, count, method)
            groups = sorted(set(map(int, full)))
            values = {group: [] for group in groups}
            for sample in samples:
                fitted = _labels(matrix[sample], count, method)
                restricted = full[sample]
                sample_groups = sorted(set(map(int, fitted)))
                contingency = np.zeros((max(groups) + 1, max(sample_groups) + 1), dtype=int)
                np.add.at(contingency, (restricted, fitted), 1)
                full_sizes = contingency.sum(axis=1)
                sample_sizes = contingency.sum(axis=0)
                for group in groups:
                    if full_sizes[group] == 0:
                        continue
                    shared = contingency[group, sample_groups]
                    jaccard = shared / np.maximum(
                        full_sizes[group] + sample_sizes[sample_groups] - shared, 1)
                    values[group].append(float(jaccard.max()))
            branches = [{"group": group, "size": int(np.sum(full == group)),
                         "mean_best_jaccard": float(np.mean(values[group]))}
                        for group in groups]
            branches.sort(key=lambda row: row["size"], reverse=True)
            eligible = [row for row in branches if row["size"] >= 20]
            stable = [row for row in eligible if row["mean_best_jaccard"] >= 0.75]
            summary["methods"][method][name] = {
                "eligible_branches": len(eligible),
                "stable_branches": len(stable),
                "stable_image_coverage": sum(row["size"] for row in stable) / len(ids),
                "weighted_survival": float(np.average(
                    [row["mean_best_jaccard"] for row in eligible],
                    weights=[row["size"] for row in eligible])) if eligible else None,
                "branches": branches,
            }
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--kec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.manifest, args.clip, args.kec, args.output)
    for method, methods in result["methods"].items():
        for name, row in methods.items():
            print(method, name, "stable branches", row["stable_branches"],
                  "coverage", round(row["stable_image_coverage"], 3),
                  "weighted survival", round(row["weighted_survival"], 3))


if __name__ == "__main__":
    main()
