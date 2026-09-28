"""EXP039: compare existing frozen semantic representations before C2 rerun.

This screening run uses no surface edges. It checks whether an existing KEC
variant gives a better cross-resolution semantic basis. All review metrics
are exploratory and model-assisted, not independent validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, sha256_file, write_json)
from .evaluate_luna_review import board_precision
from .evaluation import load_exp009_reviews
from .local_scale_c2 import local_scale
from .selective_c2_ablation import studio_proxy
from .similarity import cosine_similarity, to_dissimilarity
from .tree import average_linkage_tree


SOURCES = ("clip", "kec", "kec-n50", "kec-complete", "kec-raw-complete",
           "kec-raw-n50", "t0.1-kec", "t0.25-kec", "t0.5-kec")


def run(root: Path, secret: Path, review: Path, output: Path) -> dict:
    manifest = load_manifest(root / "manifest.json")
    ids, rows = representative_ids(manifest), representatives(manifest)
    if len(ids) != 3931:
        raise ValueError("full canonical cohort required")
    reviews = load_exp009_reviews(secret, review, manifest)
    representative = {row["id"]: row["duplicate_group"] for row in manifest["rows"]}
    for row in reviews:
        row["id"] = representative[row["id"]]
    summary = {"scope": "full 3931; model-assisted development review",
               "review_sha256": sha256_file(review),
               "manifest_sha256": manifest["manifest_sha256"], "cases": {}}
    for source in SOURCES:
        matrix = cosine_similarity(load_matrix(root / f"{source}.npy", ids))
        for name, similarity in (("raw", matrix),
                                 ("ls20a05", local_scale(matrix, 20, 0.5)),
                                 ("ls50a05", local_scale(matrix, 50, 0.5))):
            tree = average_linkage_tree(to_dissimilarity(similarity), broad=12, fine=24)
            counts = np.bincount(tree["broad"])[1:]
            key = f"{source}_{name}"
            summary["cases"][key] = {
                "board_p_at_5": board_precision(similarity, ids, reviews)["p_at_5"],
                "board_cross_resolution_p_at_5": board_precision(
                    similarity, ids, reviews, cross_resolution=True)["p_at_5"],
                "studio_proxy_p_at_5": studio_proxy(similarity, rows),
                "largest_average_12": int(counts.max()),
            }
            print(key, summary["cases"][key], flush=True)
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
