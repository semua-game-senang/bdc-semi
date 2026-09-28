"""Native-pixel region proposals, reliability, and Gabor/LBP evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageEnhance
from scipy.ndimage import laplace
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

from .common import read_json, sha256_file, unit_rows, write_json


def resolution_stratum(width: int, height: int) -> str:
    if width == height == 150:
        return "150x150"
    return "other"


def propose_boxes(width: int, height: int, minimum: int = 48) -> list[tuple[int, int, int, int]]:
    boxes = []
    for grid in (2, 3):
        for row in range(grid):
            for col in range(grid):
                box = (round(col * width / grid), round(row * height / grid),
                       round((col + 1) * width / grid), round((row + 1) * height / grid))
                if box[2] - box[0] >= minimum and box[3] - box[1] >= minimum:
                    boxes.append(box)
    side = min(width, height)
    center = ((width - side) // 2, (height - side) // 2,
              (width + side) // 2, (height + side) // 2)
    if side >= minimum:
        boxes.append(center)
    return list(dict.fromkeys(boxes))


def feature_maps(image: Image.Image) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb2gray(rgb).astype(np.float32)
    gabor_maps = []
    for frequency in (0.08, 0.16, 0.25):
        sigma = 0.56 / frequency
        side = 2 * int(np.ceil(3 * sigma)) + 1
        for theta in np.linspace(0, np.pi, 4, endpoint=False):
            real_kernel = cv2.getGaborKernel((side, side), sigma, float(theta),
                                             1 / frequency, 1.0, 0.0, ktype=cv2.CV_32F)
            imag_kernel = cv2.getGaborKernel((side, side), sigma, float(theta),
                                             1 / frequency, 1.0, np.pi / 2, ktype=cv2.CV_32F)
            real = cv2.filter2D(gray, cv2.CV_32F, real_kernel, borderType=cv2.BORDER_REFLECT)
            imag = cv2.filter2D(gray, cv2.CV_32F, imag_kernel, borderType=cv2.BORDER_REFLECT)
            gabor_maps.append(np.hypot(real, imag))
    gray_u8 = (gray * 255).astype(np.uint8)
    lbp_maps = [local_binary_pattern(gray_u8, points, radius, method="uniform")
                for points, radius in ((8, 1), (16, 2))]
    return rgb, gabor_maps, lbp_maps


def descriptors_from_maps(gabor_maps: list[np.ndarray], lbp_maps: list[np.ndarray],
                          box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box
    gabor_values = []
    for magnitude in gabor_maps:
        region = magnitude[y0:y1, x0:x1]
        gabor_values.extend((float(region.mean()), float(region.std())))
    lbp_values = []
    for pattern, points in zip(lbp_maps, (8, 16), strict=True):
        histogram = np.bincount(pattern[y0:y1, x0:x1].astype(np.int32).ravel(),
                                minlength=points + 2).astype(np.float32)
        lbp_values.extend((histogram / max(histogram.sum(), 1)).tolist())
    return np.asarray(gabor_values, dtype=np.float32), np.asarray(lbp_values, dtype=np.float32)


def descriptors(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    _, gabor_maps, lbp_maps = feature_maps(image)
    return descriptors_from_maps(gabor_maps, lbp_maps, (0, 0, image.width, image.height))


def _border_colour(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    return np.median(border, axis=0) / 255.0


def region_quality(full_rgb: np.ndarray, patch_rgb: np.ndarray,
                   box: tuple[int, int, int, int], gabor_original: np.ndarray,
                   lbp_original: np.ndarray, gabor_alt: np.ndarray,
                   lbp_alt: np.ndarray) -> tuple[float, dict]:
    area = ((box[2] - box[0]) * (box[3] - box[1])) / (full_rgb.shape[0] * full_rgb.shape[1])
    area_score = float(np.clip(area / 0.25, 0, 1))
    foreground = float(np.clip(np.linalg.norm(np.median(patch_rgb.reshape(-1, 3), axis=0)
                                              - _border_colour(full_rgb)) / 0.45, 0, 1))
    focus = float(np.var(laplace(rgb2gray(patch_rgb))))
    focus_score = float(np.clip(np.log1p(focus * 1000) / np.log1p(30), 0, 1))
    g_stability = float(np.dot(gabor_original, gabor_alt) /
                        (np.linalg.norm(gabor_original) * np.linalg.norm(gabor_alt) + 1e-12))
    l_stability = 1 - float(np.abs(lbp_original - lbp_alt).sum() / 4)
    stability = float(np.clip((g_stability + l_stability) / 2, 0, 1))
    score = float(np.clip(0.40 * stability + 0.25 * area_score +
                          0.20 * foreground + 0.15 * focus_score, 0, 1))
    return score, {"area": area_score, "foreground": foreground,
                   "focus": focus_score, "photometric_stability": stability}


@dataclass
class SurfaceStore:
    ids: list[str]
    image_index: np.ndarray
    boxes: np.ndarray
    quality: np.ndarray
    gabor: np.ndarray
    lbp: np.ndarray
    strata: list[str]
    reasons: list[str]
    _reference_source: dict | None = field(default=None, init=False, repr=False)
    _reference_arrays: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if np.any(np.diff(self.image_index) < 0):
            raise ValueError("surface regions must be grouped by image")

    def indices(self, image: int, threshold: float = 0.45) -> np.ndarray:
        start, stop = np.searchsorted(self.image_index, (image, image + 1))
        return np.arange(start, stop)[self.quality[start:stop] >= threshold]

    def reference_arrays(self, reference: dict) -> tuple[np.ndarray, np.ndarray]:
        if self._reference_source is not reference:
            self._reference_arrays = (
                np.asarray(reference["gabor_distances"]),
                np.asarray(reference["lbp_distances"]),
            )
            self._reference_source = reference
        assert self._reference_arrays is not None
        return self._reference_arrays


def extract_surfaces(rows: list[dict], output: Path, *, minimum: int = 48,
                     gate: float = 0.45) -> dict:
    image_index, boxes, quality, gabor_values, lbp_values, reasons = [], [], [], [], [], []
    for index, row in enumerate(rows):
        with Image.open(row["path"]) as opened:
            image = opened.convert("RGB")
        full_rgb = np.asarray(image, dtype=np.uint8)
        full_maps = feature_maps(image) if max(image.size) <= 512 else None
        altered_maps = feature_maps(ImageEnhance.Brightness(image).enhance(1.05)) if full_maps else None
        for proposed in propose_boxes(*image.size, minimum):
            x0, y0, x1, y1 = proposed
            if x1 - x0 > 256:
                midpoint = (x0 + x1) // 2
                x0, x1 = midpoint - 128, midpoint + 128
            if y1 - y0 > 256:
                midpoint = (y0 + y1) // 2
                y0, y1 = midpoint - 128, midpoint + 128
            box = (x0, y0, x1, y1)
            patch_rgb = full_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
            if full_maps is None:
                patch = image.crop(box)
                _, gm, lm = feature_maps(patch)
                _, ga, la = feature_maps(ImageEnhance.Brightness(patch).enhance(1.05))
                local_box = (0, 0, patch.width, patch.height)
            else:
                gm, lm = full_maps[1:]
                ga, la = altered_maps[1:]
                local_box = box
            g, l = descriptors_from_maps(gm, lm, local_box)
            g_alt, l_alt = descriptors_from_maps(ga, la, local_box)
            q, parts = region_quality(full_rgb, patch_rgb, box, g, l, g_alt, l_alt)
            image_index.append(index)
            boxes.append(box)
            quality.append(q)
            gabor_values.append(g)
            lbp_values.append(l)
            reasons.append("accepted" if q >= gate else
                           "low reliability: " + ",".join(key for key, value in parts.items() if value < 0.3))
        image.close()
    if not image_index:
        raise ValueError("no native-pixel regions met the minimum size")
    gabor_array = np.asarray(gabor_values, dtype=np.float32)
    accepted = np.asarray(quality) >= gate
    if not accepted.any():
        raise ValueError("all regions were rejected; inspect gate and image resolution")
    center = np.median(gabor_array[accepted], axis=0)
    scale = np.subtract(*np.percentile(gabor_array[accepted], [75, 25], axis=0))
    scaled = unit_rows((gabor_array - center) / np.maximum(scale, 1e-6))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, image_index=np.asarray(image_index, dtype=np.int32),
                        boxes=np.asarray(boxes, dtype=np.int32), quality=np.asarray(quality, dtype=np.float32),
                        gabor=scaled, lbp=np.asarray(lbp_values, dtype=np.float32),
                        gabor_center=center, gabor_scale=scale)
    metadata = {"ids": [row["id"] for row in rows], "strata": [
        resolution_stratum(row["width"], row["height"]) for row in rows],
        "reasons": reasons, "minimum_native_pixels": minimum, "gate": gate,
        "region_count": len(image_index), "accepted_count": int(accepted.sum()),
        "sha256": sha256_file(output)}
    write_json(output.with_suffix(".json"), metadata)
    return metadata


def load_surfaces(path: Path, expected_ids: list[str] | None = None) -> SurfaceStore:
    metadata = read_json(path.with_suffix(".json"))
    if sha256_file(path) != metadata["sha256"]:
        raise ValueError("surface artifact checksum mismatch")
    if expected_ids is not None and metadata["ids"] != expected_ids:
        raise ValueError("surface IDs differ from manifest")
    with np.load(path, allow_pickle=False) as arrays:
        return SurfaceStore(metadata["ids"], arrays["image_index"], arrays["boxes"],
                            arrays["quality"], arrays["gabor"], arrays["lbp"],
                            metadata["strata"], metadata["reasons"])


def calibrate_quality(scores: np.ndarray, human_usable: np.ndarray, *,
                      minimum_precision: float = 0.90, minimum_coverage: float = 0.20) -> dict:
    """Use EXP012 development judgments only; report both precision and coverage."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(human_usable, dtype=bool)
    if scores.shape != labels.shape or not len(scores):
        raise ValueError("quality and human usability rows differ")
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    valid = []
    for threshold in candidates:
        accepted = scores >= threshold
        if not accepted.any():
            continue
        precision = float(labels[accepted].mean())
        coverage = float(accepted.mean())
        if precision >= minimum_precision and coverage >= minimum_coverage:
            valid.append((coverage, threshold, precision))
    if not valid:
        return {"passed": False, "reason": "precision/coverage gate unmet"}
    coverage, threshold, precision = max(valid)
    return {"passed": True, "threshold": float(threshold),
            "precision": precision, "coverage": coverage}


def _chi_square(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum((a[:, None] - b[None]) ** 2 /
                        (a[:, None] + b[None] + 1e-9), axis=2)


def region_distance(store: SurfaceStore, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gabor = np.clip(1 - store.gabor[left] @ store.gabor[right].T, 0, 2)
    lbp = _chi_square(store.lbp[left], store.lbp[right])
    return gabor, lbp


def fit_distance_reference(store: SurfaceStore, *, gate: float = 0.45,
                           pairs: int = 20000, seed: int = 20260927) -> dict:
    accepted = np.flatnonzero(store.quality >= gate)
    if len(accepted) < 2:
        raise ValueError("too few accepted regions")
    rng = np.random.default_rng(seed)
    left = rng.choice(accepted, size=pairs)
    right = rng.choice(accepted, size=pairs)
    unequal = store.image_index[left] != store.image_index[right]
    left, right = left[unequal], right[unequal]
    gd = np.clip(1 - np.einsum("ij,ij->i", store.gabor[left], store.gabor[right]), 0, 2)
    ld = 0.5 * np.sum((store.lbp[left] - store.lbp[right]) ** 2 /
                      (store.lbp[left] + store.lbp[right] + 1e-9), axis=1)
    return {"gabor_distances": np.sort(gd).tolist(), "lbp_distances": np.sort(ld).tolist(),
            "sampled_pairs": len(gd), "seed": seed, "gate": gate}


def surface_pair(store: SurfaceStore, i: int, j: int, reference: dict,
                 *, gate: float = 0.45) -> tuple[float, float, list[list[int]]] | None:
    left, right = store.indices(i, gate), store.indices(j, gate)
    if not len(left) or not len(right):
        return None
    gd, ld = region_distance(store, left, right)
    gref, lref = store.reference_arrays(reference)
    sg = 1 - np.searchsorted(gref, gd, side="right") / max(len(gref), 1)
    sl = 1 - np.searchsorted(lref, ld, side="right") / max(len(lref), 1)
    scores = 0.5 * sg + 0.5 * sl
    mutual = []
    for a in range(len(left)):
        for b in np.argsort(-scores[a])[:2]:
            if a in np.argsort(-scores[:, b])[:2]:
                mutual.append((float(scores[a, b]), int(left[a]), int(right[b])))
    mutual.sort(reverse=True)
    chosen = []
    used_left, used_right = set(), set()
    for score, a, b in mutual:
        if a not in used_left and b not in used_right:
            chosen.append((score, a, b))
            used_left.add(a)
            used_right.add(b)
        if len(chosen) == 2:
            break
    if not chosen:
        return None
    weights = np.array([store.quality[a] * store.quality[b] for _, a, b in chosen])
    pair_score = float(np.average([score for score, _, _ in chosen], weights=weights))
    reliability = float(np.clip(weights.mean(), 0, 1))
    return pair_score, reliability, [[a, b] for _, a, b in chosen]
