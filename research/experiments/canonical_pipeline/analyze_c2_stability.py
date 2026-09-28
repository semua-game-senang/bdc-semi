"""Conditional tree stability of a frozen, measured C2 relation matrix.

The C2 candidate graph and reference scores stay fixed when subsampling. This
tests hierarchy sensitivity, not end-to-end refit stability, and can overstate
the stability of a fully refitted pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .common import (load_manifest, load_matrix, representative_ids,
                     representatives, write_json)
from .similarity import to_dissimilarity
from .tree import hierarchy_tree


def run(manifest_path: Path, c2_path: Path, output: Path, *,
        seed: int = 20260928, resamples: int = 20) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    rows = representatives(manifest)
    small = np.asarray([row["width"] == row["height"] == 150 for row in rows])
    similarity = load_matrix(c2_path, ids)
    distance = to_dissimilarity(similarity)
    rng = np.random.default_rng(seed)
    samples = [np.sort(np.concatenate([
        rng.choice(np.flatnonzero(small == value),
                   size=round(np.sum(small == value) * 0.8), replace=False)
        for value in (False, True)])) for _ in range(resamples)]
    summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
               "seed": seed, "resamples": resamples, "fraction": 0.8,
               "scope": "fixed C2 similarity; candidate graph and reference not refitted",
               "linkage": {}}
    for method in ("average", "complete"):
        full = hierarchy_tree(distance, broad=12, fine=24, method=method)
        scores = {"broad": [], "fine": []}
        for sample in samples:
            refit = hierarchy_tree(distance[np.ix_(sample, sample)],
                                   broad=12, fine=24, method=method)
            for cut in scores:
                scores[cut].append(float(adjusted_rand_score(
                    full[cut][sample], refit[cut])))
        summary["linkage"][method] = {
            cut: {"ari_mean": float(np.mean(values)),
                  "ari_p10_p90": np.percentile(values, [10, 90]).tolist()}
            for cut, values in scores.items()}
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--c2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.c2, args.output)
    for name, cuts in result["linkage"].items():
        print(name, "ARI@12", round(cuts["broad"]["ari_mean"], 3),
              "ARI@24", round(cuts["fine"]["ari_mean"], 3))


if __name__ == "__main__":
    main()
