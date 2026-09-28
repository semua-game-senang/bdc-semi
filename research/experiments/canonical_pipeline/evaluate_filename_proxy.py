"""Development-only device-family retrieval proxy from 150x150 filenames.

Filename prefixes are not independent human labels and must never be reported
as confirmation accuracy. Restricting neighbours to 150x150 controls the
strong resolution shortcut seen in the full corpus.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, unit_rows, write_json)


def run(manifest_path: Path, clip_path: Path, kec_path: Path,
        n50_path: Path, temp01_path: Path, raw_n50_path: Path,
        output: Path) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    rows = representatives(manifest)
    selected, labels = [], []
    for index, row in enumerate(rows):
        if row["width"] != 150 or row["height"] != 150:
            continue
        match = re.fullmatch(r"(.+)_\d+\.[^.]+", row["id"])
        if match:
            selected.append(index)
            labels.append(match.group(1))
    if len(set(labels)) != 10 or len(selected) < 2500:
        raise ValueError("filename proxy no longer matches the audited studio subset")
    selected = np.asarray(selected, dtype=int)
    labels = np.asarray(labels)
    n = len(selected)
    x = load_matrix(clip_path, ids)
    raw = load_matrix(raw_n50_path, ids)
    if not np.allclose(raw[:, :x.shape[1]], x, atol=1e-6):
        raise ValueError("raw KEC vectors do not match CLIP")
    residual = raw[:, x.shape[1]:]
    residual -= residual.mean(axis=0, keepdims=True)
    vectors = {"C0": x,
               "C1_N300": load_matrix(kec_path, ids),
               "C1_N50": load_matrix(n50_path, ids),
               "C1_N50_t01": load_matrix(temp01_path, ids),
               "C1_N50_centered_025": unit_rows(np.concatenate(
                   (x, 0.25 * unit_rows(residual)), axis=1))}
    unique, counts = np.unique(labels, return_counts=True)
    random_baseline = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
    summary = {"manifest_sha256": manifest["manifest_sha256"],
               "scope": "150x150 training images with filename family proxy only",
               "images": n, "families": dict(zip(unique.tolist(), counts.tolist())),
               "random_pair_baseline": random_baseline, "methods": {}}
    for name, matrix in vectors.items():
        sample = matrix[selected]
        similarity = sample @ sample.T
        np.fill_diagonal(similarity, -np.inf)
        candidates = np.argpartition(-similarity, kth=4, axis=1)[:, :5]
        order = np.argsort(-np.take_along_axis(similarity, candidates, axis=1), axis=1)
        neighbours = np.take_along_axis(candidates, order, axis=1)
        matched = labels[neighbours] == labels[:, None]
        per_query = matched.mean(axis=1)
        summary["methods"][name] = {
            "p_at_5": float(per_query.mean()),
            "balanced_family_p_at_5": float(np.mean([
                per_query[labels == family].mean() for family in unique])),
            "top1_match": float(matched[:, 0].mean()),
            "per_family_p_at_5": {family: float(per_query[labels == family].mean())
                                  for family in unique},
        }
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--kec", type=Path, required=True)
    parser.add_argument("--n50", type=Path, required=True)
    parser.add_argument("--temp01", type=Path, required=True)
    parser.add_argument("--raw-n50", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.clip, args.kec, args.n50,
                 args.temp01, args.raw_n50, args.output)
    print("random baseline", round(result["random_pair_baseline"], 4))
    for name, row in result["methods"].items():
        print(name, "P@5", round(row["p_at_5"], 4),
              "balanced P@5", round(row["balanced_family_p_at_5"], 4))


if __name__ == "__main__":
    main()
