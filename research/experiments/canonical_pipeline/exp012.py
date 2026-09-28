"""Development-only audit of the existing blinded EXP012 region review."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from .common import load_manifest, write_json
from .surface import SurfaceStore, calibrate_quality


def _csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    area = max(0, x1 - x0) * max(0, y1 - y0)
    la = (left[2] - left[0]) * (left[3] - left[1])
    ra = (right[2] - right[0]) * (right[3] - right[1])
    return area / max(la + ra - area, 1)


def audit_exp012(store: SurfaceStore, region_manifest_path: Path, review_path: Path,
                 output: Path, *, overlap_threshold: float = 0.5,
                 minimum_precision: float = 0.90, minimum_coverage: float = 0.20,
                 manifest_path: Path | None = None) -> dict:
    """Selected crops are positive; only explicitly unusable images supply negatives.

    Other unselected crops have unknown usability and are excluded from calibration.
    This is an overlap proxy, not a claim that every proposed crop was reviewed.
    """
    regions = _csv(region_manifest_path)
    reviews = _csv(review_path)
    if not regions or not reviews:
        raise ValueError("EXP012 region manifest and completed review are required")
    by_blind: dict[str, list[dict]] = {}
    for row in regions:
        by_blind.setdefault(row["blind_id"], []).append(row)
    review_ids = [row["blind_id"] for row in reviews]
    if len(review_ids) != len(set(review_ids)) or set(review_ids) != set(by_blind):
        raise ValueError("EXP012 review must cover every blind ID exactly once")
    id_index = {id_: i for i, id_ in enumerate(store.ids)}
    representative = ({row["id"]: row["duplicate_group"]
                       for row in load_manifest(manifest_path)["rows"]}
                      if manifest_path is not None else {})
    scores, labels = [], []
    counts = {"positive_proposals": 0, "explicit_negative_proposals": 0,
              "unlabeled_proposals": 0, "images": len(reviews)}
    for review in reviews:
        blind_id = review["blind_id"]
        items = by_blind[blind_id]
        relative = items[0]["relative_path"].replace("\\", "/").removeprefix("1_Electronic/")
        image_id = representative.get(relative, relative)
        if image_id not in id_index:
            raise ValueError(f"EXP012 image absent from surface store: {relative}")
        if any(item["relative_path"] != items[0]["relative_path"] for item in items):
            raise ValueError(f"EXP012 region image mismatch: {blind_id}")
        use = review.get("usable_surface", "").strip().lower()
        if use not in {"yes", "no"}:
            raise ValueError(f"EXP012 usable_surface must be yes or no: {blind_id}")
        selected = set()
        for column in ("selected_region_1", "selected_region_2"):
            value = review.get(column, "").strip().upper()
            selected.update(re.findall(r"R\d+", value))
        possible = {item["region_id"] for item in items}
        if not selected.issubset(possible) or (use == "yes" and not selected) or (use == "no" and selected):
            raise ValueError(f"EXP012 selected regions conflict with usability: {blind_id}")
        boxes = []
        for item in items:
            if item["region_id"] in selected:
                x, y, side = (int(item[key]) for key in ("x", "y", "side"))
                boxes.append((x, y, x + side, y + side))
        for index in store.indices(id_index[image_id], threshold=0):
            proposal = tuple(map(int, store.boxes[index]))
            positive = use == "yes" and any(
                _overlap(proposal, chosen) >= overlap_threshold for chosen in boxes)
            negative = use == "no"
            if positive or negative:
                scores.append(float(store.quality[index]))
                labels.append(positive)
                counts["positive_proposals" if positive else "explicit_negative_proposals"] += 1
            else:
                counts["unlabeled_proposals"] += 1
    if not scores or not any(labels) or all(labels):
        calibration = {"passed": False, "reason": "both selected positives and explicit negatives are required"}
    else:
        calibration = calibrate_quality(
            np.asarray(scores), np.asarray(labels),
            minimum_precision=minimum_precision, minimum_coverage=minimum_coverage)
    result = {"protocol": "EXP012 development only", "counts": counts,
              "overlap_threshold": overlap_threshold, "calibration": calibration,
              "label_rule": "IoU-matched selected crop=positive; explicitly unusable image=negative; other crops=unknown",
              "warning": "gate precision is a proposal-overlap proxy and needs visual verification"}
    write_json(output, result)
    return result
