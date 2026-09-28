"""Locked human confirmation, retrieval, hierarchy, and nuisance evaluation."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from .common import digest_json, read_json, sha256_file, write_json
from .similarity import candidate_pairs


def load_reviews(path: Path, allowed_ids: set[str]) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    required = {"id", "family", "visible_surfaces", "surface_observable", "source_stratum", "mixed"}
    if not reviews or not required.issubset(reviews[0]):
        raise ValueError(f"review CSV requires columns {sorted(required)}")
    ids = [row["id"] for row in reviews]
    if len(ids) != len(set(ids)) or not set(ids).issubset(allowed_ids):
        raise ValueError("review IDs are duplicate or outside the manifest")
    for row in reviews:
        if row["surface_observable"].lower() not in {"yes", "no"}:
            raise ValueError("surface_observable must be yes or no")
        if row["mixed"].lower() not in {"yes", "no", "unknown"}:
            raise ValueError("mixed must be yes, no, or unknown")
    return reviews


def load_exp009_reviews(secret_manifest: Path, blinded_review: Path,
                        manifest: dict) -> list[dict]:
    """Join the untouched EXP009 blind IDs only during locked evaluation."""
    def read_csv(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    secret = read_csv(secret_manifest)
    reviewed = read_csv(blinded_review)
    if not secret or not reviewed:
        raise ValueError("EXP009 secret manifest and completed review are required")
    by_blind = {row["blind_id"]: row for row in secret}
    if len(by_blind) != len(secret):
        raise ValueError("duplicate blind IDs in EXP009 secret manifest")
    if len(reviewed) != len(secret) or {row["blind_id"] for row in reviewed} != set(by_blind):
        raise ValueError("EXP009 review must cover exactly the frozen blind IDs")
    by_image = {row["id"]: row for row in manifest["rows"]}
    result = []
    for reviewed_row in reviewed:
        original = by_blind[reviewed_row["blind_id"]]
        relative = original["relative_path"].replace("\\", "/")
        image_id = relative.removeprefix("1_Electronic/")
        if image_id not in by_image or by_image[image_id]["sha256"].lower() != original["raw_sha256"].lower():
            raise ValueError(f"EXP009 manifest image/hash mismatch for blind ID {reviewed_row['blind_id']}")
        materials = [reviewed_row.get(key, "").strip() for key in ("material_1", "material_2")]
        materials = [item for item in materials if item and item.lower() not in
                     {"unknown", "none", "not visible", "unobservable"}]
        visibility = reviewed_row.get("material_visibility", "").strip().lower()
        observable = bool(materials) and visibility not in {
            "no", "none", "not visible", "unobservable", "unknown"}
        count = reviewed_row.get("object_count", "").strip().lower()
        scene = reviewed_row.get("scene_unit", "").strip().lower()
        mixed = ("yes" if any(word in scene for word in ("pile", "mixed")) or
                 (count.isdigit() and int(count) > 1) else
                 "no" if count in {"1", "one"} else "unknown")
        result.append({"id": image_id, "family": reviewed_row.get("object_family", "").strip(),
                       "visible_surfaces": ";".join(materials),
                       "surface_observable": "yes" if observable else "no",
                       "source_stratum": original["resolution_group"],
                       "mixed": mixed, "blind_id": reviewed_row["blind_id"],
                       "confidence": reviewed_row.get("annotation_confidence", "").strip()})
    return result


def visible_tags(row: dict) -> set[str] | None:
    if row["surface_observable"].lower() != "yes":
        return None
    return {term.strip().lower() for term in row["visible_surfaces"].split(";") if term.strip()}


def surface_jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def cmrg(reviews: list[dict], fine_labels: dict[str, int], minimum_family: int = 6) -> dict:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in reviews:
        if row["family"] and visible_tags(row) is not None and row["id"] in fine_labels:
            by_family[row["family"]].append(row)
    per_family = {}
    for family, rows in by_family.items():
        if len(rows) < minimum_family:
            continue
        all_scores, child_scores = [], []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                score = surface_jaccard(visible_tags(rows[i]), visible_tags(rows[j]))
                all_scores.append(score)
                if fine_labels[rows[i]["id"]] == fine_labels[rows[j]["id"]]:
                    child_scores.append(score)
        baseline = float(np.mean(all_scores))
        coverage = len(child_scores) / len(all_scores)
        child = float(np.mean(child_scores)) if child_scores else None
        gain = coverage * (child - baseline) if child is not None else 0.0
        per_family[family] = {"images": len(rows), "all_pairs": len(all_scores),
                              "child_pairs": len(child_scores), "baseline": baseline,
                              "child_mean": child, "pair_coverage": coverage, "gain": gain}
    return {"value": float(np.mean([v["gain"] for v in per_family.values()])) if per_family else None,
            "eligible_families": len(per_family), "per_family": per_family}


def within_family_surface_p_at_k(similarity: np.ndarray, ids: list[str], reviews: list[dict],
                                 k: int = 5) -> dict:
    index = {id_: i for i, id_ in enumerate(ids)}
    by_family = defaultdict(list)
    for row in reviews:
        if row["family"] and visible_tags(row) is not None and row["id"] in index:
            by_family[row["family"]].append(row)
    matches = []
    random_baselines = []
    for family, rows in by_family.items():
        if len(rows) < 2:
            continue
        for row in rows:
            options = [other for other in rows if other["id"] != row["id"]]
            ranked = sorted(options, key=lambda other: -float(similarity[index[row["id"]], index[other["id"]]]))
            chosen = ranked[:k]
            query_tags = visible_tags(row)
            matches.append(float(np.mean([bool(query_tags & visible_tags(other)) for other in chosen])))
            random_baselines.append(float(np.mean([bool(query_tags & visible_tags(other)) for other in options])))
    return {"precision_at_k": float(np.mean(matches)) if matches else None,
            "family_conditioned_random": float(np.mean(random_baselines)) if random_baselines else None,
            "queries": len(matches), "k": k}


def reviewed_candidate_recall(similarity: np.ndarray, ids: list[str], reviews: list[dict],
                              k: int = 50) -> dict:
    index = {id_: i for i, id_ in enumerate(ids)}
    candidates = set(candidate_pairs(similarity, k))
    relevant = 0
    available = 0
    for i, left in enumerate(reviews):
        if left["id"] not in index or visible_tags(left) is None:
            continue
        for right in reviews[i + 1:]:
            if right["id"] not in index or right["family"] != left["family"] or visible_tags(right) is None:
                continue
            if not visible_tags(left) & visible_tags(right):
                continue
            relevant += 1
            pair = tuple(sorted((index[left["id"]], index[right["id"]])))
            available += pair in candidates
    return {"relevant_pairs": relevant, "available_pairs": available,
            "recall": available / relevant if relevant else None, "k": k}


def conditional_leakage(reviews: list[dict], labels: dict[str, int]) -> dict:
    by_family = defaultdict(list)
    for row in reviews:
        if row["id"] in labels and row["family"] and row["source_stratum"]:
            by_family[row["family"]].append(row)
    per_family = {}
    for family, rows in by_family.items():
        if len(rows) < 6 or len({row["source_stratum"] for row in rows}) < 2:
            continue
        per_family[family] = float(adjusted_mutual_info_score(
            [labels[row["id"]] for row in rows], [row["source_stratum"] for row in rows]))
    return {"within_family_ami": float(np.mean(list(per_family.values()))) if per_family else None,
            "per_family": per_family}


def family_coherence(reviews: list[dict], labels: dict[str, int]) -> dict:
    rows = [row for row in reviews if row["id"] in labels and row["family"]]
    if len(rows) < 2:
        return {"images": len(rows), "ami": None, "ari": None}
    truth = [row["family"] for row in rows]
    predicted = [labels[row["id"]] for row in rows]
    return {"images": len(rows), "ami": float(adjusted_mutual_info_score(truth, predicted)),
            "ari": float(adjusted_rand_score(truth, predicted))}


def subsample_stability(distance: np.ndarray, full_labels: np.ndarray, *,
                        clusters: int, repeats: int = 20, fraction: float = 0.8,
                        seed: int = 20260927) -> dict:
    rng = np.random.default_rng(seed)
    n = len(distance)
    scores = []
    for _ in range(repeats):
        selected = np.sort(rng.choice(n, size=max(clusters + 1, round(n * fraction)), replace=False))
        tree = linkage(squareform(distance[np.ix_(selected, selected)], checks=True), method="average")
        partial = fcluster(tree, clusters, criterion="maxclust")
        scores.append(float(adjusted_rand_score(full_labels[selected], partial)))
    return {"mean_ari": float(np.mean(scores)), "fifth_percentile": float(np.percentile(scores, 5)),
            "repeats": repeats, "fraction": fraction}


def freeze_selection(path: Path, *, config_path: Path, artifact_paths: list[Path],
                     method: str, broad: int, fine: int, review_id_path: Path) -> dict:
    if path.exists():
        raise FileExistsError("selection is already frozen; create a new versioned run")
    if not review_id_path.exists():
        raise FileNotFoundError("locked EXP009 ID manifest is required")
    frozen = {"method": method, "broad": broad, "fine": fine,
              "config_sha256": sha256_file(config_path),
              "artifacts": {str(p.resolve()): sha256_file(p) for p in artifact_paths},
              "review_ids_sha256": sha256_file(review_id_path)}
    frozen["selection_sha256"] = digest_json(frozen)
    write_json(path, frozen)
    return frozen


def verify_selection(path: Path, *, config_path: Path, review_id_path: Path) -> dict:
    frozen = read_json(path)
    checksum = frozen.pop("selection_sha256")
    if digest_json(frozen) != checksum:
        raise ValueError("selection file was changed after freezing")
    frozen["selection_sha256"] = checksum
    if sha256_file(config_path) != frozen["config_sha256"] or sha256_file(review_id_path) != frozen["review_ids_sha256"]:
        raise ValueError("config or locked review IDs changed after freeze")
    for name, digest in frozen["artifacts"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"frozen artifact changed: {name}")
    return frozen


def evaluate_method(similarity: np.ndarray, ids: list[str], reviews: list[dict],
                    broad_labels: np.ndarray, fine_labels: np.ndarray, *,
                    candidate_k: int = 50, distance: np.ndarray | None = None) -> dict:
    if len(ids) != len(broad_labels) or len(ids) != len(fine_labels):
        raise ValueError("method label alignment failed")
    broad_map = dict(zip(ids, map(int, broad_labels), strict=True))
    fine_map = dict(zip(ids, map(int, fine_labels), strict=True))
    result = {"broad_family": family_coherence(reviews, broad_map),
              "fine_family": family_coherence(reviews, fine_map),
              "cmrg": cmrg(reviews, fine_map),
              "surface_retrieval": within_family_surface_p_at_k(similarity, ids, reviews),
              "candidate_recall": reviewed_candidate_recall(similarity, ids, reviews, candidate_k),
              "conditional_leakage_broad": conditional_leakage(reviews, broad_map),
              "conditional_leakage_fine": conditional_leakage(reviews, fine_map)}
    if distance is not None:
        result["broad_stability"] = subsample_stability(distance, broad_labels,
                                                        clusters=len(set(broad_labels)))
        result["fine_stability"] = subsample_stability(distance, fine_labels,
                                                       clusters=len(set(fine_labels)))
    return result
