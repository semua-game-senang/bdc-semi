"""Canonical pairwise refinement and acquisition-matched reference scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import save_matrix, unit_rows, write_json
from .surface import SurfaceStore, surface_pair


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    x = unit_rows(matrix)
    similarity = x @ x.T
    np.clip(similarity, -1, 1, out=similarity)
    return similarity.astype(np.float32)


def candidate_pairs(similarity: np.ndarray, k: int) -> list[tuple[int, int]]:
    n = len(similarity)
    if similarity.shape != (n, n) or not np.allclose(similarity, similarity.T, atol=1e-5):
        raise ValueError("candidate similarity must be square and symmetric")
    k = min(k, n - 1)
    if k < 1:
        return []
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    partial = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    return sorted({(min(i, int(j)), max(i, int(j))) for i, row in enumerate(partial)
                   for j in row if i != j})


def fit_excess_reference(store: SurfaceStore, similarity: np.ndarray, distance_reference: dict,
                         *, candidate_k: int = 50, gate: float = 0.45,
                         samples_per_stratum: int = 500, seed: int = 20260927) -> dict:
    """Non-neighbour expected surface score by resolution-stratum pair."""
    rng = np.random.default_rng(seed)
    excluded = set(candidate_pairs(similarity, candidate_k))
    by_stratum: dict[str, list[int]] = {}
    for index, stratum in enumerate(store.strata):
        if len(store.indices(index, gate)):
            by_stratum.setdefault(stratum, []).append(index)
    means = {}
    counts = {}
    all_scores = []
    strata = sorted(by_stratum)
    for left_index, a in enumerate(strata):
        for b in strata[left_index:]:
            values = []
            attempts = 0
            while len(values) < samples_per_stratum and attempts < samples_per_stratum * 30:
                attempts += 1
                i = int(rng.choice(by_stratum[a]))
                j = int(rng.choice(by_stratum[b]))
                pair = (min(i, j), max(i, j))
                if i == j or pair in excluded:
                    continue
                match = surface_pair(store, i, j, distance_reference, gate=gate)
                if match is not None:
                    values.append(match[0])
            key = f"{a}|{b}"
            counts[key] = len(values)
            if values:
                means[key] = float(np.mean(values))
                all_scores.extend(values)
    if not all_scores:
        raise ValueError("no acquisition-matched reference pairs with usable regions")
    return {"means": means, "counts": counts, "global_mean": float(np.mean(all_scores)),
            "seed": seed, "candidate_k": candidate_k, "gate": gate}


def refine_similarity(semantic: np.ndarray, store: SurfaceStore, distance_reference: dict,
                      excess_reference: dict, *, candidate_k: int = 50,
                      strength: float = 0.1, gate: float = 0.45) -> tuple[np.ndarray, list[dict]]:
    if len(semantic) != len(store.ids):
        raise ValueError("surface and semantic IDs differ")
    refined = np.asarray(semantic, dtype=np.float32).copy()
    changed = []
    for i, j in candidate_pairs(semantic, candidate_k):
        match = surface_pair(store, i, j, distance_reference, gate=gate)
        if match is None:
            continue
        score, reliability, regions = match
        key = "|".join(sorted((store.strata[i], store.strata[j])))
        baseline = excess_reference["means"].get(key, excess_reference["global_mean"])
        excess = float(np.clip(score - baseline, -1, 1))
        delta = strength * reliability * excess
        refined[i, j] = refined[j, i] = semantic[i, j] + delta
        changed.append({"i": i, "j": j, "score": score, "reliability": reliability,
                        "expected": baseline, "excess": excess, "delta": delta, "regions": regions})
    return refined, changed


def to_dissimilarity(similarity: np.ndarray) -> np.ndarray:
    similarity = np.asarray(similarity, dtype=np.float32)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    if not np.allclose(similarity, similarity.T, atol=1e-5) or not np.isfinite(similarity).all():
        raise ValueError("similarity must be symmetric and finite")
    maximum = float(np.max(similarity))
    distance = np.maximum(0, maximum - similarity)
    np.fill_diagonal(distance, 0)
    return distance.astype(np.float32)


def build_base(clip: np.ndarray, kec: np.ndarray, ids: list[str], store: SurfaceStore,
               distance_reference: dict, excess_reference: dict, output_dir: Path,
               *, candidate_k: int = 50, strength: float = 0.1,
               gate: float = 0.45) -> dict:
    if ids != store.ids or len(ids) != len(clip) or len(ids) != len(kec):
        raise ValueError("base method ID alignment failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    c0, c1 = cosine_similarity(clip), cosine_similarity(kec)
    c2, changes = refine_similarity(c1, store, distance_reference, excess_reference,
                                    candidate_k=candidate_k, strength=strength, gate=gate)
    for name, matrix in (("C0", c0), ("C1", c1), ("C2", c2)):
        save_matrix(output_dir / f"{name}_similarity.npy", matrix, ids,
                    {"method": name, "candidate_k": candidate_k, "strength": strength, "gate": gate})
        save_matrix(output_dir / f"{name}_distance.npy", to_dissimilarity(matrix), ids,
                    {"method": name, "distance": "S_max - S"})
    write_json(output_dir / "C2_changed_edges.json", changes)
    return {"ids": ids, "changed_edges": len(changes), "candidate_edges": len(candidate_pairs(c1, candidate_k)),
            "surface_edge_coverage": len(changes) / max(len(candidate_pairs(c1, candidate_k)), 1)}
