"""Label-free audit of how KEC changes frozen CLIP neighbourhoods."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, unit_rows, write_json)


def top_neighbours(vectors: np.ndarray, k: int = 50) -> np.ndarray:
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    candidates = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(similarity, candidates, axis=1), axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def report(root: Path, n50_path: Path, manifest_path: Path,
           n50_raw_path: Path | None = None,
           complete_path: Path | None = None,
           temperature_root: Path | None = None) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    rows = representatives(manifest)
    small = np.asarray([row["width"] == row["height"] == 150 for row in rows])
    vectors = {
        "C0": load_matrix(root / "clip.npy", ids),
        "C1_N300": load_matrix(root / "kec.npy", ids),
        "C1_N50": load_matrix(n50_path, ids),
    }
    if complete_path is not None:
        vectors["C1_N300_complete"] = load_matrix(complete_path, ids)
    temperature_diagnostic = {}
    if temperature_root is not None:
        for temperature in (0.5, 0.25, 0.1):
            key = f"C1_N50_grounding_t{temperature:g}"
            vectors[key] = load_matrix(temperature_root / f"t{temperature:g}-kec.npy", ids)
            weights = np.load(temperature_root / f"t{temperature:g}-kec_weights.npy",
                              allow_pickle=False)
            if weights.shape != (len(ids), 3):
                raise ValueError("concept weights have unexpected shape")
            temperature_diagnostic[str(temperature)] = {
                "median_max_weight": float(np.median(weights.max(axis=1))),
                "argmax_counts": np.bincount(weights.argmax(axis=1), minlength=3).tolist(),
            }
    residual_diagnostic = None
    if n50_raw_path is not None:
        raw = load_matrix(n50_raw_path, ids)
        x = vectors["C0"]
        if raw.shape != (len(ids), 2 * x.shape[1]) or not np.allclose(raw[:, :x.shape[1]], x, atol=1e-6):
            raise ValueError("N/50 raw KEC block is not aligned with CLIP")
        knowledge = raw[:, x.shape[1]:]
        residual = knowledge - knowledge.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(residual, axis=1)
        residual_diagnostic = {"knowledge_residual_norm_p10_p50_p90":
                               np.percentile(norms, [10, 50, 90]).tolist()}
        for strength in (0.25, 0.5, 1.0):
            vectors[f"C1_N50_centered_{strength:g}"] = unit_rows(
                np.concatenate((x, strength * unit_rows(residual)), axis=1))
    neighbours = {name: top_neighbours(value) for name, value in vectors.items()}
    base = neighbours["C0"]
    rng = np.random.default_rng(20260928)
    left = rng.integers(len(ids), size=200000)
    right = rng.integers(len(ids), size=200000)
    unequal = left != right
    left, right = left[unequal], right[unequal]

    def pair_scores(matrix: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.einsum("ij,ij->i", matrix[left[start:start + 10000]],
                      matrix[right[start:start + 10000]])
            for start in range(0, len(left), 10000)])

    clip_scores = pair_scores(vectors["C0"])
    summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
               "resolution_150x150_fraction": float(small.mean()),
               "methods": {}}
    primary_pairs = {(min(i, int(j)), max(i, int(j)))
                     for i, row in enumerate(neighbours["C1_N300"])
                     for j in row if i != j}
    cross_pairs = sum(small[i] != small[j] for i, j in primary_pairs)
    summary["primary_candidate_graph"] = {
        "top_k": 50, "edges": len(primary_pairs),
        "cross_resolution_edges": cross_pairs,
        "cross_resolution_fraction": cross_pairs / len(primary_pairs),
    }
    if residual_diagnostic is not None:
        summary["residual_diagnostic"] = residual_diagnostic
    if temperature_diagnostic:
        summary["temperature_diagnostic"] = temperature_diagnostic
    for name, ranked in neighbours.items():
        scores = pair_scores(vectors[name])
        slope, intercept = np.polyfit(clip_scores, scores, 1)
        residual = scores - (intercept + slope * clip_scores)
        retained_10 = np.fromiter(
            (len(set(left[:10]) & set(right[:10])) / 10
             for left, right in zip(base, ranked, strict=True)),
            dtype=np.float32, count=len(ids))
        retained_50 = np.fromiter(
            (len(set(left) & set(right)) / 50
             for left, right in zip(base, ranked, strict=True)),
            dtype=np.float32, count=len(ids))
        summary["methods"][name] = {
            "clip_top10_retention_mean": float(retained_10.mean()),
            "clip_top10_retention_p10": float(np.percentile(retained_10, 10)),
            "clip_top50_retention_mean": float(retained_50.mean()),
            "same_top1_as_clip_fraction": float(np.mean(ranked[:, 0] == base[:, 0])),
            "same_resolution_top10_fraction": float(np.mean(
                small[:, None] == small[ranked[:, :10]])),
            "same_resolution_top50_fraction": float(np.mean(
                small[:, None] == small[ranked[:, :50]])),
            "same_resolution_top50_by_query": {
                "150x150": float(np.mean(small[ranked[small, :50]])),
                "other": float(np.mean(~small[ranked[~small, :50]])),
            },
            "pairwise_pearson_vs_clip": float(np.corrcoef(clip_scores, scores)[0, 1]),
            "pairwise_affine_slope": float(slope),
            "pairwise_affine_intercept": float(intercept),
            "pairwise_affine_residual_std": float(residual.std()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--n50", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n50-raw", type=Path)
    parser.add_argument("--complete", type=Path)
    parser.add_argument("--temperature-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = report(args.root, args.n50, args.manifest, args.n50_raw,
                     args.complete, args.temperature_root)
    write_json(args.output, summary)
    print(summary)


if __name__ == "__main__":
    main()
