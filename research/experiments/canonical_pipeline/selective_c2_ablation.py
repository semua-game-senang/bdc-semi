"""Replay measured C2 edges through fixed reliability gates on all 3,931 images.

EXP037 is an exploratory development ablation. The 150-image GPT-6 Luna review
is model-assisted and reused here; its scores are not held-out validation.
No image labels or review judgments enter the pairwise similarity formula.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .analyze_c2_gates import _top
from .common import (load_manifest, load_matrix, read_json, representative_ids,
                     representatives, sha256_file, write_json)
from .evaluate_luna_review import board_precision
from .evaluation import load_exp009_reviews
from .similarity import cosine_similarity, to_dissimilarity
from .tree import average_linkage_tree


def studio_proxy(matrix: np.ndarray, rows: list[dict]) -> float:
    indices, labels = [], []
    for i, row in enumerate(rows):
        if row["width"] == row["height"] == 150:
            match = re.fullmatch(r"(.+)_\d+\.[^.]+", row["id"])
            if match:
                indices.append(i)
                labels.append(match.group(1))
    labels = np.asarray(labels)
    if len(indices) != 2641 or len(set(labels)) != 10:
        raise ValueError("studio proxy cohort changed")
    neighbours = _top(matrix[np.ix_(indices, indices)], 5)
    return float(np.mean(labels[neighbours] == labels[:, None]))


def run(root: Path, secret: Path, review: Path, output: Path) -> dict:
    manifest = load_manifest(root / "manifest.json")
    ids = representative_ids(manifest)
    if len(ids) != 3931:
        raise ValueError("expected full 3,931-image representative dataset")
    rows = representatives(manifest)
    kec = load_matrix(root / "kec.npy", ids)
    baseline = cosine_similarity(kec)
    actual = load_matrix(root / "C2_similarity.npy", ids)
    edges = read_json(root / "C2_changed_edges.json")
    left = np.fromiter((edge["i"] for edge in edges), dtype=np.int32)
    right = np.fromiter((edge["j"] for edge in edges), dtype=np.int32)
    delta = np.fromiter((edge["delta"] for edge in edges), dtype=np.float32)
    reliability = np.fromiter((edge["reliability"] for edge in edges), dtype=np.float32)
    excess = np.fromiter((edge["excess"] for edge in edges), dtype=np.float32)
    if len(edges) != 145036 or np.max(np.abs(actual[left, right] - baseline[left, right] - delta)) > 1e-5:
        raise ValueError("stored C2 changes do not reproduce full canonical C2")

    reviews = load_exp009_reviews(secret, review, manifest)
    representative = {row["id"]: row["duplicate_group"] for row in manifest["rows"]}
    for row in reviews:
        row["id"] = representative[row["id"]]
    if len({row["id"] for row in reviews}) != len(reviews):
        raise ValueError("review sample repeats an exact-duplicate representative")

    cases = [("C1", 1.0, "none", 0.0), ("C2_permissive", 1.0, "signed", 0.0)]
    for threshold in (0.55, 0.60, 0.65, 0.70):
        for sign, weight in (("signed", 1.0), ("positive", 1.0),
                             ("signed", 0.5), ("positive", 0.5)):
            cases.append((f"rel_{threshold:.2f}_{sign}_w{weight:.1f}",
                          weight, sign, threshold))

    result = {
        "scope": "full 3931 representatives; exploratory model-assisted development review",
        "manifest_sha256": manifest["manifest_sha256"],
        "review_sha256": sha256_file(review),
        "input_c2_sha256": sha256_file(root / "C2_similarity.npy"),
        "candidate_edges": len(edges),
        "selection_note": "fixed reliability thresholds; no labels in relation computation; same review reused for diagnostics",
        "cases": {},
    }
    for name, weight, sign, threshold in cases:
        enabled = (reliability >= threshold) if sign != "none" else np.zeros(len(delta), bool)
        if sign == "positive":
            enabled &= excess > 0
        correction = np.where(enabled, weight * delta, 0).astype(np.float32)
        matrix = baseline.copy()
        matrix[left, right] += correction
        matrix[right, left] += correction
        np.fill_diagonal(matrix, 1.0)
        board = board_precision(matrix, ids, reviews)
        cross = board_precision(matrix, ids, reviews, cross_resolution=True)
        tree = average_linkage_tree(to_dissimilarity(matrix), broad=12, fine=24)
        counts = np.bincount(tree["broad"])[1:]
        result["cases"][name] = {
            "reliability_min": threshold,
            "sign": sign,
            "weight": weight,
            "changed_edges": int(np.count_nonzero(correction)),
            "changed_fraction": float(np.count_nonzero(correction) / len(edges)),
            "board_p_at_5": board["p_at_5"],
            "board_cross_resolution_p_at_5": cross["p_at_5"],
            "board_queries": board["queries"],
            "studio_proxy_p_at_5": studio_proxy(matrix, rows),
            "largest_average_12": int(counts.max()),
            "clusters_below_five": int(np.sum(counts < 5)),
        }
        print(name, result["cases"][name], flush=True)
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--secret", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.secret, args.review, args.output)


if __name__ == "__main__":
    main()
