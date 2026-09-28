"""EXP042: common-geometry and source-bias audit of the new full C2 tree.

Full-data silhouette is computed in the same frozen CLIP geometry for the
original C1, revised semantic control, and selective C2 labels. Review-family
scores use reused model-assisted development annotations only.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             silhouette_score)

from .common import (load_manifest, load_matrix, read_json, representative_ids,
                     representatives, sha256_file, write_json)
from .evaluation import family_coherence, load_exp009_reviews
from .prompt_relation_audit import PROMPTS
from .similarity import cosine_similarity, to_dissimilarity
from .tree import average_linkage_tree


def run(root: Path, run_dir: Path, text_vectors: Path, secret: Path,
        review: Path, output: Path) -> dict:
    manifest = load_manifest(root / "manifest.json")
    ids, rows = representative_ids(manifest), representatives(manifest)
    clip = load_matrix(root / "clip.npy", ids)
    kec = load_matrix(root / "kec.npy", ids)
    texts = load_matrix(text_vectors, list(PROMPTS))
    c2 = load_matrix(run_dir / "C2_similarity.npy", ids)
    saved = load_matrix(run_dir / "C2_labels.npy", ids).astype(int)
    run = read_json(run_dir / "run.json")
    if run["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("C2 run belongs to a different manifest")
    if len(ids) != 3931 or saved.shape != (3931, 2):
        raise ValueError("full-run IDs or labels changed")
    temperature = run["semantic_prompt_temperature"]
    weight = run["semantic_prompt_weight"]
    posterior = softmax((clip @ texts.T) / temperature, axis=1).astype(np.float32)
    revised = ((1 - weight) * cosine_similarity(kec)
               + weight * cosine_similarity(posterior))
    np.fill_diagonal(revised, 1.0)
    relations = {"original_C1": cosine_similarity(kec),
                 "prompt_control": revised, "selective_C2": c2}
    resolution = np.asarray([int(row["width"] == row["height"] == 150)
                             for row in rows])
    studio_indices, studio_labels = [], []
    for i, row in enumerate(rows):
        match = re.fullmatch(r"(.+)_\d+\.[^.]+", row["id"])
        if row["width"] == row["height"] == 150 and match:
            studio_indices.append(i)
            studio_labels.append(match.group(1))
    if len(studio_indices) != 2641 or len(set(studio_labels)) != 10:
        raise ValueError("studio proxy cohort changed")
    reviews = load_exp009_reviews(secret, review, manifest)
    representative = {row["id"]: row["duplicate_group"] for row in manifest["rows"]}
    for row in reviews:
        row["id"] = representative[row["id"]]
    common = np.maximum(0, 1 - cosine_similarity(clip)).astype(np.float32)
    np.fill_diagonal(common, 0)
    summary = {"scope": "full 3931 representatives; common frozen CLIP cosine geometry",
               "metric_note": "descriptive development diagnostics, not material validity",
               "prompt_temperature": temperature, "prompt_weight": weight,
               "manifest_sha256": manifest["manifest_sha256"],
               "review_sha256": sha256_file(review), "methods": {}}
    for name, relation in relations.items():
        tree = average_linkage_tree(to_dissimilarity(relation), broad=12, fine=24)
        summary["methods"][name] = {}
        for column, (cut_name, k) in enumerate((("broad", 12), ("fine", 24))):
            labels = tree[cut_name]
            if name == "selective_C2" and adjusted_rand_score(labels, saved[:, column]) < 0.999999:
                raise ValueError("saved C2 labels do not match recomputed tree")
            counts = np.bincount(labels)[1:]
            label_map = dict(zip(ids, map(int, labels), strict=True))
            review_score = family_coherence(reviews, label_map)
            summary["methods"][name][str(k)] = {
                "silhouette_common_clip": float(silhouette_score(
                    common, labels, metric="precomputed")),
                "largest": int(counts.max()),
                "sizes": sorted(counts.tolist(), reverse=True),
                "resolution_ami": float(adjusted_mutual_info_score(resolution, labels)),
                "studio_family_ami": float(adjusted_mutual_info_score(
                    studio_labels, labels[studio_indices])),
                "review_family_ami": review_score["ami"],
                "review_scope": "150-image reused GPT-6 Luna model-assisted review",
            }
            print(name, k, summary["methods"][name][str(k)], flush=True)
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "run_dir", "text_vectors", "secret", "review", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.run_dir, args.text_vectors, args.secret, args.review, args.output)


if __name__ == "__main__":
    main()
