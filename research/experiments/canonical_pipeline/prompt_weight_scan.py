"""EXP043: bounded development sweep of the fixed proposal concept bank.

The reviewed board annotations are reused during selection, so any chosen
setting requires fresh independent validation before an accuracy claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, sha256_file, write_json)
from .evaluate_luna_review import board_precision
from .evaluation import load_exp009_reviews
from .prompt_relation_audit import PROMPTS
from .selective_c2_ablation import studio_proxy
from .similarity import cosine_similarity, to_dissimilarity
from .tree import average_linkage_tree


TEMPERATURES = (0.035, 0.040, 0.045, 0.050, 0.055, 0.060)
WEIGHTS = (0.15, 0.20, 0.25, 0.30, 0.35)


def run(root: Path, text_vectors: Path, secret: Path,
        review: Path, output: Path) -> dict:
    manifest = load_manifest(root / "manifest.json")
    ids, rows = representative_ids(manifest), representatives(manifest)
    clip = load_matrix(root / "clip.npy", ids)
    kec = load_matrix(root / "kec.npy", ids)
    texts = load_matrix(text_vectors, list(PROMPTS))
    if len(ids) != 3931:
        raise ValueError("full dataset required")
    reviews = load_exp009_reviews(secret, review, manifest)
    representative = {row["id"]: row["duplicate_group"] for row in manifest["rows"]}
    for row in reviews:
        row["id"] = representative[row["id"]]
    base = cosine_similarity(kec)
    logits = clip @ texts.T
    summary = {"scope": "full 3931; bounded development sweep, not independent test",
               "manifest_sha256": manifest["manifest_sha256"],
               "review_sha256": sha256_file(review),
               "temperatures": TEMPERATURES, "weights": WEIGHTS,
               "review_used_for_selection": True, "cases": {}}
    for temperature in TEMPERATURES:
        posterior = softmax(logits / temperature, axis=1).astype(np.float32)
        concept = cosine_similarity(posterior)
        for weight in WEIGHTS:
            relation = (1 - weight) * base + weight * concept
            tree = average_linkage_tree(to_dissimilarity(relation), broad=12, fine=24)
            counts = np.bincount(tree["broad"])[1:]
            key = f"t{temperature:.3f}_w{weight:.2f}"
            summary["cases"][key] = {
                "board_p_at_5": board_precision(relation, ids, reviews)["p_at_5"],
                "board_cross_resolution_p_at_5": board_precision(
                    relation, ids, reviews, cross_resolution=True)["p_at_5"],
                "studio_proxy_p_at_5": studio_proxy(relation, rows),
                "largest_average_12": int(counts.max()),
            }
    write_json(output, summary)
    qualified = [(key, row) for key, row in summary["cases"].items()
                 if row["board_p_at_5"] >= 0.70
                 and row["board_cross_resolution_p_at_5"] >= 0.3875
                 and row["studio_proxy_p_at_5"] >= 0.8947
                 and row["largest_average_12"] <= 2200]
    print("qualified semantic controls:", qualified, flush=True)
    print("best board score:", max(row["board_p_at_5"] for row in summary["cases"].values()), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "text_vectors", "secret", "review", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.text_vectors, args.secret, args.review, args.output)


if __name__ == "__main__":
    main()
