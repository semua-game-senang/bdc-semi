"""Unlabelled matched-tree sensitivity for canonical semantic representations."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, unit_rows, write_json)
from .similarity import cosine_similarity, to_dissimilarity
from .tree import hierarchy_tree


def run(root: Path, manifest_path: Path, n50: Path, temp01: Path,
        n50_raw: Path, output: Path) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    resolution = np.asarray([int(row["width"] == row["height"] == 150)
                             for row in representatives(manifest)])
    x = load_matrix(root / "clip.npy", ids)
    vectors = {
        "C0": x,
        "C1_N300": load_matrix(root / "kec.npy", ids),
        "C1_N50": load_matrix(n50, ids),
        "C1_N50_t01": load_matrix(temp01, ids),
    }
    raw = load_matrix(n50_raw, ids)
    if not np.allclose(raw[:, :x.shape[1]], x, atol=1e-6):
        raise ValueError("raw KEC representation differs from CLIP")
    residual = raw[:, x.shape[1]:]
    residual -= residual.mean(axis=0, keepdims=True)
    vectors["C1_N50_centered_025"] = unit_rows(
        np.concatenate((x, 0.25 * unit_rows(residual)), axis=1))
    distances = {name: to_dissimilarity(cosine_similarity(matrix))
                 for name, matrix in vectors.items()}
    summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
               "broad_cut": 12, "fine_cut": 24, "linkage": {}}
    for method in ("average", "complete"):
        trees = {name: hierarchy_tree(distance, broad=12, fine=24, method=method)
                 for name, distance in distances.items()}
        baseline = trees["C0"]
        summary["linkage"][method] = {}
        for name, tree in trees.items():
            summary["linkage"][method][name] = {
                "broad_ari_vs_clip": float(adjusted_rand_score(baseline["broad"], tree["broad"])),
                "fine_ari_vs_clip": float(adjusted_rand_score(baseline["fine"], tree["fine"])),
                "broad_resolution_ami": float(adjusted_mutual_info_score(
                    resolution, tree["broad"])),
                "fine_resolution_ami": float(adjusted_mutual_info_score(
                    resolution, tree["fine"])),
                "broad_sizes": sorted(np.bincount(tree["broad"])[1:].tolist(), reverse=True),
                "fine_sizes": sorted(np.bincount(tree["fine"])[1:].tolist(), reverse=True),
            }
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n50", type=Path, required=True)
    parser.add_argument("--temp01", type=Path, required=True)
    parser.add_argument("--n50-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.manifest, args.n50, args.temp01,
                 args.n50_raw, args.output)
    for method, methods in result["linkage"].items():
        for name, row in methods.items():
            print(method, name, "broad ARI", round(row["broad_ari_vs_clip"], 4),
                  "fine ARI", round(row["fine_ari_vs_clip"], 4),
                  "largest broad", row["broad_sizes"][0],
                  "resolution AMI", round(row["broad_resolution_ami"], 4))


if __name__ == "__main__":
    main()
