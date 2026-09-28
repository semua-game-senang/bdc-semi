"""Score the frozen C0/C1/C2 comparison on approved model-assisted blind reviews.

This is a development diagnostic. The labels were produced by GPT-6 Luna,
not by an independent human annotator. Do not use it as competition accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .common import (load_manifest, load_matrix, representative_ids,
                     sha256_file, write_json)
from .evaluation import (cmrg, family_coherence, load_exp009_reviews,
                         within_family_surface_p_at_k)
from .similarity import cosine_similarity, to_dissimilarity
from .tree import hierarchy_tree


def board_precision(similarity: np.ndarray, ids: list[str], reviews: list[dict],
                    *, cross_resolution: bool = False, k: int = 5) -> dict:
    index = {image_id: i for i, image_id in enumerate(ids)}
    rows = [row for row in reviews if row["id"] in index]
    board = [row for row in rows if row["family"].strip().lower() == "board"]
    scores, chance = [], []
    for query in board:
        options = [row for row in rows if row["id"] != query["id"] and
                   (not cross_resolution or
                    row["source_stratum"] != query["source_stratum"])]
        if len(options) < k:
            continue
        ranked = sorted(options, key=lambda row: -float(
            similarity[index[query["id"]], index[row["id"]]]))[:k]
        scores.append(sum(row["family"].strip().lower() == "board"
                          for row in ranked) / k)
        chance.append(sum(row["family"].strip().lower() == "board"
                          for row in options) / len(options))
    return {"queries": len(scores), "board_images": len(board),
            "p_at_5": float(np.mean(scores)) if scores else None,
            "random_baseline": float(np.mean(chance)) if chance else None,
            "candidate_scope": "opposite resolution" if cross_resolution else "all reviewed images"}


def run(manifest_path: Path, secret_path: Path, review_path: Path,
        clip_path: Path, kec_path: Path, c2_path: Path, output: Path) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    reviews = load_exp009_reviews(secret_path, review_path, manifest)
    representative = {row["id"]: row["duplicate_group"] for row in manifest["rows"]}
    for row in reviews:
        row["id"] = representative[row["id"]]
    if len({row["id"] for row in reviews}) != len(reviews):
        raise ValueError("confirmation sample contains duplicate representatives")
    summary = {"scope": "approved GPT-6 Luna model-assisted review; not human ground truth",
               "selection": "C0/C1/C2 and 12/24 cuts frozen before joining blind mapping",
               "manifest_sha256": manifest["manifest_sha256"],
               "review_sha256": sha256_file(review_path),
               "blind_manifest_sha256": sha256_file(secret_path),
               "reviewed_images": len(reviews),
               "family_counts": dict(Counter(row["family"] for row in reviews)),
               "methods": {}}
    clip = load_matrix(clip_path, ids)
    kec = load_matrix(kec_path, ids)
    matrices = {"C0": cosine_similarity(clip),
                "C1": cosine_similarity(kec),
                "C2": load_matrix(c2_path, ids)}
    for method, similarity in matrices.items():
        distance = to_dissimilarity(similarity)
        result = {"board_p_at_5": board_precision(similarity, ids, reviews),
                  "board_cross_resolution_p_at_5": board_precision(
                      similarity, ids, reviews, cross_resolution=True),
                  "surface_word_overlap_proxy": within_family_surface_p_at_k(
                      similarity, ids, reviews), "linkage": {}}
        for linkage in ("average", "complete"):
            tree = hierarchy_tree(distance, broad=12, fine=24, method=linkage)
            cuts = {}
            for name in ("broad", "fine"):
                labels = dict(zip(ids, map(int, tree[name]), strict=True))
                cuts[name] = {"family": family_coherence(reviews, labels),
                              "cmrg_word_overlap_proxy": cmrg(reviews, labels)}
            result["linkage"][linkage] = cuts
        summary["methods"][method] = result
        del distance, similarity
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "secret", "review", "clip", "kec", "c2", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.secret, args.review, args.clip,
                 args.kec, args.c2, args.output)
    for name, method in result["methods"].items():
        print(name, "board P@5", method["board_p_at_5"]["p_at_5"],
              "cross-resolution", method["board_cross_resolution_p_at_5"]["p_at_5"],
              "average AMI@12", method["linkage"]["average"]["broad"]["family"]["ami"])


if __name__ == "__main__":
    main()
