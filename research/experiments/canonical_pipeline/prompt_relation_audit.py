"""EXP040: a balanced visible-object text bank for CLIP/KEC neighbour search.

Prompts describe broad object families already present in the proposal and
preliminary images. They do not use EXP009 review labels or filenames as model
inputs. This is an exploratory development audit, not independent validation.
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
from .local_scale_c2 import local_scale
from .selective_c2_ablation import studio_proxy
from .similarity import cosine_similarity, to_dissimilarity
from .tree import average_linkage_tree


PROMPTS = (
    "a photo of an exposed printed circuit board with copper traces and components",
    "a photo of an electronic circuit board",
    "a photo of a discarded battery",
    "a photo of a computer keyboard",
    "a photo of a computer mouse",
    "a photo of a discarded mobile phone",
    "a photo of a television or monitor",
    "a photo of a broken printer",
    "a photo of a washing machine",
    "a photo of a microwave oven",
    "a photo of a discarded audio player or speaker",
    "a photo of mixed electronic waste in a pile",
)


def run(root: Path, text_vectors: Path, secret: Path, review: Path, output: Path) -> dict:
    manifest = load_manifest(root / "manifest.json")
    ids, rows = representative_ids(manifest), representatives(manifest)
    clip = load_matrix(root / "clip.npy", ids)
    kec = load_matrix(root / "kec.npy", ids)
    texts = load_matrix(text_vectors, list(PROMPTS))
    if len(ids) != 3931:
        raise ValueError("full canonical cohort required")
    reviews = load_exp009_reviews(secret, review, manifest)
    representative = {row["id"]: row["duplicate_group"] for row in manifest["rows"]}
    for row in reviews:
        row["id"] = representative[row["id"]]
    base = cosine_similarity(kec)
    logits = clip @ texts.T
    summary = {"scope": "full 3931; proposal-defined prompt bank; model-assisted development review",
               "prompts": list(PROMPTS), "manifest_sha256": manifest["manifest_sha256"],
               "review_sha256": sha256_file(review),
               "text_vectors_sha256": sha256_file(text_vectors), "cases": {}}
    for temperature in (0.03, 0.05, 0.1):
        posterior = softmax(logits / temperature, axis=1).astype(np.float32)
        concept = cosine_similarity(posterior)
        for weight in (0.05, 0.10, 0.20):
            relation = (1 - weight) * base + weight * concept
            for name, matrix in (("raw", relation),
                                 ("ls20a05", local_scale(relation, 20, 0.5))):
                tree = average_linkage_tree(to_dissimilarity(matrix), broad=12, fine=24)
                counts = np.bincount(tree["broad"])[1:]
                key = f"t{temperature}_w{weight}_{name}"
                summary["cases"][key] = {
                    "board_p_at_5": board_precision(matrix, ids, reviews)["p_at_5"],
                    "board_cross_resolution_p_at_5": board_precision(
                        matrix, ids, reviews, cross_resolution=True)["p_at_5"],
                    "studio_proxy_p_at_5": studio_proxy(matrix, rows),
                    "largest_average_12": int(counts.max()),
                }
                print(key, summary["cases"][key], flush=True)
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "text_vectors", "secret", "review", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.text_vectors, args.secret, args.review, args.output)


if __name__ == "__main__":
    main()
