"""Label-free and filename-proxy checks of the finished canonical C2 matrix."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, write_json)


def _top(matrix: np.ndarray, k: int) -> np.ndarray:
    work = np.asarray(matrix, dtype=np.float32).copy()
    np.fill_diagonal(work, -np.inf)
    candidates = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(work, candidates, axis=1), axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def run(manifest_path: Path, clip_path: Path, kec_path: Path,
        c2_path: Path, output: Path) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    rows = representatives(manifest)
    clip = load_matrix(clip_path, ids)
    kec = load_matrix(kec_path, ids)
    c2 = load_matrix(c2_path, ids)
    if c2.shape != (len(ids), len(ids)) or not np.allclose(c2, c2.T, atol=1e-5):
        raise ValueError("C2 similarity is not aligned or symmetric")
    similarities = {"C0": clip @ clip.T, "C1": kec @ kec.T, "C2": c2}
    small = np.asarray([row["width"] == row["height"] == 150 for row in rows])
    filename_indices, labels = [], []
    for index, row in enumerate(rows):
        if not small[index]:
            continue
        match = re.fullmatch(r"(.+)_\d+\.[^.]+", row["id"])
        if match:
            filename_indices.append(index)
            labels.append(match.group(1))
    filename_indices = np.asarray(filename_indices, dtype=int)
    labels = np.asarray(labels)
    if len(set(labels)) != 10:
        raise ValueError("studio filename proxy changed")
    summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
               "filename_proxy_images": len(labels), "methods": {}}
    top50 = {}
    for name, similarity in similarities.items():
        ranked = _top(similarity, 50)
        top50[name] = ranked
        subset = similarity[np.ix_(filename_indices, filename_indices)]
        studio_neighbours = _top(subset, 5)
        same_family = labels[studio_neighbours] == labels[:, None]
        same_resolution = small[:, None] == small[ranked]
        summary["methods"][name] = {
            "filename_p_at_5": float(same_family.mean()),
            "filename_p_at_5_by_family": {family: float(same_family[labels == family].mean())
                                          for family in sorted(set(labels))},
            "same_resolution_top10": float(same_resolution[:, :10].mean()),
            "same_resolution_top50": float(same_resolution.mean()),
        }
    for name in ("C0", "C1"):
        summary["methods"]["C2"][f"top10_retention_vs_{name}"] = float(np.mean([
            len(set(left[:10]) & set(right[:10])) / 10
            for left, right in zip(top50[name], top50["C2"], strict=True)]))
        summary["methods"]["C2"][f"top50_retention_vs_{name}"] = float(np.mean([
            len(set(left) & set(right)) / 50
            for left, right in zip(top50[name], top50["C2"], strict=True)]))
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--kec", type=Path, required=True)
    parser.add_argument("--c2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.clip, args.kec, args.c2, args.output)
    for name, row in result["methods"].items():
        print(name, "filename P@5", round(row["filename_p_at_5"], 4),
              "same-resolution top50", round(row["same_resolution_top50"], 4))
    print("C2 top50 retention vs C1",
          round(result["methods"]["C2"]["top50_retention_vs_C1"], 4))


if __name__ == "__main__":
    main()
