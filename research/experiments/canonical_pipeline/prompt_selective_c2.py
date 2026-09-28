"""EXP041: full-corpus selective C2 over the fixed proposal-concept relation.

Recomputes the symmetric top-50 candidate union and all Gabor/LBP matches
from cached native-pixel regions. The prompt relation is the matched semantic
control. Model-assisted review scores remain exploratory development metrics.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .common import (load_manifest, load_matrix, read_json, representative_ids,
                     representatives, save_matrix, sha256_file, write_json)
from .evaluate_luna_review import board_precision
from .evaluation import load_exp009_reviews
from .prompt_relation_audit import PROMPTS
from .selective_c2_ablation import studio_proxy
from .similarity import (candidate_pairs, cosine_similarity, fit_excess_reference,
                         refine_similarity, to_dissimilarity)
from .surface import fit_distance_reference, load_surfaces
from .tree import average_linkage_tree


def run(root: Path, text_vectors: Path, secret: Path | None, review: Path | None,
        output_dir: Path, *, prompt_temperature: float = 0.05,
        prompt_weight: float = 0.25) -> dict:
    started = time.perf_counter()
    manifest = load_manifest(root / "manifest.json")
    ids, rows = representative_ids(manifest), representatives(manifest)
    if len(ids) != 3931:
        raise ValueError("full 3,931-image cohort required")
    clip = load_matrix(root / "clip.npy", ids)
    kec = load_matrix(root / "kec.npy", ids)
    texts = load_matrix(text_vectors, list(PROMPTS))
    surfaces = load_surfaces(root / "surfaces.npz", ids)
    reference_path = root / "surface_distance_reference.json"
    distance_reference = (read_json(reference_path) if reference_path.exists()
                          else fit_distance_reference(
                              surfaces, gate=0.45, pairs=20000, seed=20260927))
    if (secret is None) != (review is None):
        raise ValueError("supply both review paths or neither")
    reviews = load_exp009_reviews(secret, review, manifest) if secret and review else None
    if reviews is not None:
        representative = {row["id"]: row["duplicate_group"] for row in manifest["rows"]}
        for row in reviews:
            row["id"] = representative[row["id"]]
        if len(reviews) != 150 or len({row["id"] for row in reviews}) != 150:
            raise ValueError("review cohort changed")

    if not 0 < prompt_temperature <= 1 or not 0 <= prompt_weight <= 1:
        raise ValueError("invalid prompt temperature or weight")
    prompt_posterior = softmax((clip @ texts.T) / prompt_temperature, axis=1).astype(np.float32)
    semantic = ((1 - prompt_weight) * cosine_similarity(kec)
                + prompt_weight * cosine_similarity(prompt_posterior))
    np.fill_diagonal(semantic, 1.0)
    candidate_count = len(candidate_pairs(semantic, 50))
    excess_reference = fit_excess_reference(
        surfaces, semantic, distance_reference, candidate_k=50, gate=0.45,
        samples_per_stratum=500, seed=20260927)
    full, edges = refine_similarity(
        semantic, surfaces, distance_reference, excess_reference,
        candidate_k=50, strength=0.1, gate=0.45)
    if len(edges) != candidate_count or len(edges) < 100000:
        raise ValueError("unexpected candidate edge coverage")
    left = np.fromiter((e["i"] for e in edges), dtype=np.int32)
    right = np.fromiter((e["j"] for e in edges), dtype=np.int32)
    delta = np.fromiter((e["delta"] for e in edges), dtype=np.float32)
    reliable = np.fromiter((e["reliability"] >= 0.60 for e in edges), dtype=bool)
    if np.max(np.abs(full[left, right] - semantic[left, right] - delta)) > 1e-5:
        raise ValueError("C2 correction reconstruction failed")

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "distance_reference.json", distance_reference)
    write_json(output_dir / "excess_reference.json", excess_reference)
    write_json(output_dir / "candidate_edges.json", edges)
    summary = {
        "scope": "full 3931 representatives; same revised semantic relation for control and C2",
        "status": "exploratory development; model-assisted labels reused for diagnostics",
        "manifest_sha256": manifest["manifest_sha256"],
        "review_sha256": sha256_file(review) if review else None,
        "text_vectors_sha256": sha256_file(text_vectors),
        "surface_sha256": sha256_file(root / "surfaces.npz"),
        "candidate_k": 50, "candidate_edges": candidate_count,
        "region_gate": 0.45, "pair_reliability_min": 0.60,
        "semantic_prompt_temperature": prompt_temperature,
        "semantic_prompt_weight": prompt_weight,
        "cases": {},
    }
    for name, correction_weight in (("prompt_control", 0.0),
                                    ("C2_quarter", 0.25),
                                    ("C2_half", 0.5),
                                    ("C2_full", 1.0)):
        modification = np.where(reliable, correction_weight * delta, 0).astype(np.float32)
        matrix = semantic.copy()
        matrix[left, right] += modification
        matrix[right, left] += modification
        np.fill_diagonal(matrix, 1.0)
        tree = average_linkage_tree(to_dissimilarity(matrix), broad=12, fine=24)
        sizes = np.bincount(tree["broad"])[1:]
        summary["cases"][name] = {
            "changed_edges": int(np.count_nonzero(modification)),
            "changed_fraction": float(np.count_nonzero(modification) / candidate_count),
            "board_p_at_5": board_precision(matrix, ids, reviews)["p_at_5"]
                            if reviews is not None else None,
            "board_cross_resolution_p_at_5": board_precision(
                matrix, ids, reviews, cross_resolution=True)["p_at_5"]
                if reviews is not None else None,
            "studio_proxy_p_at_5": studio_proxy(matrix, rows),
            "largest_average_12": int(sizes.max()),
            "clusters_below_five": int(np.sum(sizes < 5)),
            "broad_sizes": sorted(sizes.tolist(), reverse=True),
        }
        if name == "C2_half":
            save_matrix(output_dir / "C2_similarity.npy", matrix, ids, {
                "method": "prompt semantic + reliability-gated native-region C2",
                "prompt_temperature": prompt_temperature,
                "prompt_weight": prompt_weight,
                "candidate_k": 50, "region_gate": 0.45,
                "pair_reliability_min": 0.60, "surface_strength": 0.05,
                "manifest_sha256": manifest["manifest_sha256"]})
            save_matrix(output_dir / "C2_labels.npy",
                        np.column_stack([tree["broad"], tree["fine"]]), ids,
                        {"linkage": "average", "columns": ["broad_12", "fine_24"]})
        print(name, summary["cases"][name], flush=True)
    summary["elapsed_s"] = time.perf_counter() - started
    write_json(output_dir / "run.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "text_vectors", "output_dir"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--secret", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--prompt-temperature", type=float, default=0.05)
    parser.add_argument("--prompt-weight", type=float, default=0.25)
    args = parser.parse_args()
    run(args.root, args.text_vectors, args.secret, args.review, args.output_dir,
        prompt_temperature=args.prompt_temperature, prompt_weight=args.prompt_weight)


if __name__ == "__main__":
    main()

