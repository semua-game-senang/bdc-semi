"""EXP038: test hubness correction before selective C2 surface refinement.

This is a full-corpus development experiment. Local scaling subtracts each
image's mean similarity to its semantic neighbours symmetrically, so average
linkage still sees one symmetric whole-image relation. Review scores are
model-assisted diagnostics and are not an untouched validation set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (load_manifest, load_matrix, read_json, representative_ids,
                     representatives, sha256_file, write_json)
from .evaluate_luna_review import board_precision
from .evaluation import load_exp009_reviews
from .selective_c2_ablation import studio_proxy
from .similarity import cosine_similarity, to_dissimilarity
from .tree import average_linkage_tree


def local_scale(similarity: np.ndarray, k: int, alpha: float) -> np.ndarray:
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    neighbours = np.partition(work, -k, axis=1)[:, -k:]
    density = neighbours.mean(axis=1)
    centred = similarity - (alpha / 2) * (density[:, None] + density[None, :])
    centred += alpha * density.mean()
    np.fill_diagonal(centred, 1.0)
    return centred.astype(np.float32)


def run(root: Path, secret: Path, review: Path, output: Path) -> dict:
    manifest = load_manifest(root / "manifest.json")
    ids = representative_ids(manifest)
    rows = representatives(manifest)
    if len(ids) != 3931:
        raise ValueError("expected full canonical cohort")
    baseline = cosine_similarity(load_matrix(root / "kec.npy", ids))
    edges = read_json(root / "C2_changed_edges.json")
    left = np.fromiter((e["i"] for e in edges), dtype=np.int32)
    right = np.fromiter((e["j"] for e in edges), dtype=np.int32)
    delta = np.fromiter((e["delta"] for e in edges), dtype=np.float32)
    reliability = np.fromiter((e["reliability"] for e in edges), dtype=np.float32)
    reviews = load_exp009_reviews(secret, review, manifest)
    representative = {row["id"]: row["duplicate_group"] for row in manifest["rows"]}
    for row in reviews:
        row["id"] = representative[row["id"]]
    if len(reviews) != 150 or len({row["id"] for row in reviews}) != 150:
        raise ValueError("review cohort changed")

    summary = {"scope": "full 3931; exploratory development; model-assisted labels",
               "manifest_sha256": manifest["manifest_sha256"],
               "review_sha256": sha256_file(review),
               "relation": "S_ij - alpha*(r_i+r_j)/2 + alpha*mean(r); r=mean top-k C1 similarity",
               "candidate_edges": len(edges), "cases": {}}
    for k in (20, 50):
        for alpha in (0.25, 0.5, 1.0):
            scaled = local_scale(baseline, k, alpha)
            for gate, weight in (("none", 0.0), ("rel060_full", 1.0),
                                 ("rel060_half", 0.5)):
                matrix = scaled.copy()
                enabled = reliability >= 0.60 if weight else np.zeros(len(edges), bool)
                correction = np.where(enabled, weight * delta, 0).astype(np.float32)
                matrix[left, right] += correction
                matrix[right, left] += correction
                np.fill_diagonal(matrix, 1.0)
                tree = average_linkage_tree(to_dissimilarity(matrix), broad=12, fine=24)
                counts = np.bincount(tree["broad"])[1:]
                name = f"k{k}_a{alpha}_{gate}"
                summary["cases"][name] = {
                    "local_k": k, "alpha": alpha, "surface_gate": gate,
                    "changed_edges": int(np.count_nonzero(correction)),
                    "board_p_at_5": board_precision(matrix, ids, reviews)["p_at_5"],
                    "board_cross_resolution_p_at_5": board_precision(
                        matrix, ids, reviews, cross_resolution=True)["p_at_5"],
                    "studio_proxy_p_at_5": studio_proxy(matrix, rows),
                    "largest_average_12": int(counts.max()),
                    "clusters_below_five": int(np.sum(counts < 5)),
                }
                print(name, summary["cases"][name], flush=True)
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "secret", "review", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.secret, args.review, args.output)


if __name__ == "__main__":
    main()
