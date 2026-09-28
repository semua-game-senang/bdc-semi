"""Audit C0/C1/C2 average-linkage cuts in one frozen CLIP geometry.

Silhouette is descriptive here: the common CLIP space is fair across methods,
but it is not a test of whether a cluster represents useful e-waste meaning.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "data/eda-runs/canonical-run-v1"


def cut(distance: np.ndarray, k: int) -> np.ndarray:
    return cut_tree(linkage(distance, method="average"), n_clusters=k).ravel()


def main() -> None:
    clip = np.load(RUN / "clip.npy")
    kec = np.load(RUN / "kec.npy")
    c2 = np.load(RUN / "C2_similarity.npy", mmap_mode="r")
    saved = np.load(RUN / "C2_labels.npy").astype(int)
    ids = json.loads((RUN / "run.json").read_text(encoding="utf-8"))["base"]["ids"]
    families = set(json.loads((RUN / "filename-proxy-audit.json").read_text(
        encoding="utf-8"))["families"])
    studio = np.asarray([
        (match := re.fullmatch(r"(.+)_\d+\.[^.]+", image_id)) is not None
        and match.group(1) in families for image_id in ids
    ])
    assert clip.shape[0] == kec.shape[0] == c2.shape[0] == saved.shape[0] == 3931
    common = squareform(pdist(clip, metric="cosine")).astype("float32")
    np.fill_diagonal(common, 0)
    trees = {
        "C0": pdist(clip, metric="cosine"),
        "C1": pdist(kec, metric="cosine"),
        "C2": squareform(np.maximum(0, float(np.max(c2)) - c2), checks=False),
    }
    result = {
        "scope": "full 3931 representatives; common frozen CLIP cosine distance",
        "metric_note": "secondary geometric diagnostic; not material validity",
        "methods": {},
    }
    for method, distance in trees.items():
        result["methods"][method] = {}
        for column, k in enumerate((12, 24)):
            labels = cut(distance, k)
            sizes = np.bincount(labels)
            if method == "C2":
                from sklearn.metrics import adjusted_rand_score
                assert adjusted_rand_score(labels, saved[:, column]) > .999999
            score = silhouette_score(common, labels, metric="precomputed")
            result["methods"][method][str(k)] = {
                "silhouette_common_clip": float(score),
                "largest": int(sizes.max()),
                "clusters_below_five": int((sizes < 5).sum()),
                "sizes": sorted(map(int, sizes), reverse=True),
            }
            if method == "C2" and k == 12:
                giant = int(np.argmax(sizes))
                result["methods"][method][str(k)]["studio_prefix_in_largest"] = int(
                    np.count_nonzero(studio & (labels == giant)))
                result["methods"][method][str(k)]["studio_prefix_total"] = int(
                    np.count_nonzero(studio))
            print(method, k, round(score, 5), int(sizes.max()))
    destination = RUN / "common-silhouette-audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
