"""Exploratory C2 sign and texture-informativeness ablations.

The filename-family retrieval proxy is development metadata, not a human
material label or locked competition result. These ablations are not a
selection rule until independent surface judgments exist.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .common import (load_manifest, load_matrix, read_json,
                     representative_ids, representatives, write_json)
from .surface import load_surfaces


def _entropy(histogram: np.ndarray) -> np.ndarray:
    first, second = histogram[:, :10], histogram[:, 10:]
    return ((-np.sum(first * np.log(np.maximum(first, 1e-9)), axis=1) / np.log(10)) +
            (-np.sum(second * np.log(np.maximum(second, 1e-9)), axis=1) / np.log(18))) / 2


def _top(matrix: np.ndarray, k: int) -> np.ndarray:
    work = np.asarray(matrix, dtype=np.float32).copy()
    np.fill_diagonal(work, -np.inf)
    candidates = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(work, candidates, axis=1), axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def run(manifest_path: Path, kec_path: Path, c2_path: Path,
        surfaces_path: Path, changes_path: Path, output: Path) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    rows = representatives(manifest)
    kec = load_matrix(kec_path, ids)
    baseline = kec @ kec.T
    actual = load_matrix(c2_path, ids)
    store = load_surfaces(surfaces_path, ids)
    changes = read_json(changes_path)
    left = np.fromiter((edge["i"] for edge in changes), dtype=np.int32)
    right = np.fromiter((edge["j"] for edge in changes), dtype=np.int32)
    delta = np.fromiter((edge["delta"] for edge in changes), dtype=np.float32)
    if np.max(np.abs(actual[left, right] - baseline[left, right] - delta)) > 1e-5:
        raise ValueError("edge changes do not reconstruct the finished C2 matrix")
    entropy = _entropy(store.lbp)
    matched_entropy = np.fromiter(
        (min(entropy[edge["regions"][0][0]], entropy[edge["regions"][0][1]])
         for edge in changes), dtype=np.float32)
    studio, labels = [], []
    for i, row in enumerate(rows):
        if row["width"] != row["height"] or row["width"] != 150:
            continue
        match = re.fullmatch(r"(.+)_\d+\.[^.]+", row["id"])
        if match:
            studio.append(i)
            labels.append(match.group(1))
    studio = np.asarray(studio, dtype=int)
    labels = np.asarray(labels)
    if len(set(labels)) != 10:
        raise ValueError("studio filename proxy changed")
    resolution = np.asarray([row["width"] == row["height"] == 150 for row in rows])
    variants = {
        "C1": np.zeros(len(delta), dtype=np.float32),
        "C2_signed": delta,
        "positive_only": np.maximum(delta, 0),
        "signed_entropy_08": np.where(matched_entropy >= 0.8, delta, 0),
        "positive_entropy_08": np.where(matched_entropy >= 0.8, np.maximum(delta, 0), 0),
        "positive_entropy_085": np.where(matched_entropy >= 0.85, np.maximum(delta, 0), 0),
        "positive_half": np.maximum(delta, 0) * 0.5,
    }
    summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
               "candidate_edges": len(delta),
               "entropy_gate": "minimum of first mutual matched-region LBP entropy",
               "filename_proxy_images": len(studio), "variants": {}}
    for name, modification in variants.items():
        matrix = baseline.copy()
        matrix[left, right] += modification
        matrix[right, left] += modification
        ranked = _top(matrix, 50)
        subset = matrix[np.ix_(studio, studio)]
        neighbours = _top(subset, 5)
        matches = labels[neighbours] == labels[:, None]
        summary["variants"][name] = {
            "changed_edges": int(np.sum(modification != 0)),
            "mean_abs_delta": float(np.abs(modification).mean()),
            "filename_p_at_5": float(matches.mean()),
            "same_resolution_top50": float(np.mean(
                resolution[:, None] == resolution[ranked])),
            "per_family_p_at_5": {family: float(matches[labels == family].mean())
                                  for family in sorted(set(labels))},
        }
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kec", type=Path, required=True)
    parser.add_argument("--c2", type=Path, required=True)
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--changes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.kec, args.c2,
                 args.surfaces, args.changes, args.output)
    for name, row in result["variants"].items():
        print(name, "edges", row["changed_edges"],
              "P@5", round(row["filename_p_at_5"], 4),
              "resolution", round(row["same_resolution_top50"], 4))


if __name__ == "__main__":
    main()
