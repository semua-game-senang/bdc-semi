"""Immutable inputs, deterministic IDs, and artifact integrity helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def readable_path(path: Path) -> Path:
    """Use Windows extended paths for the eight official long filenames."""
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path.absolute()))
    return path


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def save_matrix(path: Path, matrix: np.ndarray, ids: list[str], metadata: dict[str, Any]) -> None:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or not np.isfinite(matrix).all():
        raise ValueError("matrix shape, IDs, or finite-value check failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix, allow_pickle=False)
    write_json(path.with_suffix(".json"), {"ids": ids, "shape": list(matrix.shape), "metadata": metadata,
                                           "sha256": sha256_file(path)})


def load_matrix(path: Path, expected_ids: list[str] | None = None) -> np.ndarray:
    sidecar = read_json(path.with_suffix(".json"))
    if sha256_file(path) != sidecar["sha256"]:
        raise ValueError(f"artifact checksum mismatch: {path}")
    if expected_ids is not None and sidecar["ids"] != expected_ids:
        raise ValueError(f"artifact row IDs differ: {path}")
    matrix = np.load(path, allow_pickle=False)
    if list(matrix.shape) != sidecar["shape"] or not np.isfinite(matrix).all():
        raise ValueError(f"invalid matrix: {path}")
    return np.asarray(matrix, dtype=np.float32)


def build_manifest(image_root: Path, *, allow_fixture: bool = False, expected_count: int | None = 3961) -> dict[str, Any]:
    root = image_root.resolve(strict=True)
    if not allow_fixture and (root.name != "1_Electronic" or root.parent.name != "data_train"):
        raise ValueError("Only the official data_train/1_Electronic directory is accepted")
    paths = sorted((p for p in root.rglob("*") if readable_path(p).is_file()
                    and p.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda p: p.relative_to(root).as_posix())
    if expected_count is not None and len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} images, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    by_hash: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        io_path = readable_path(path)
        raw_hash = sha256_file(io_path)
        with Image.open(io_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            decoded_hash = hashlib.sha256(image.tobytes()).hexdigest()
        group = by_hash.setdefault(raw_hash, relative)
        rows.append({"id": relative, "path": str(io_path), "sha256": raw_hash,
                     "pixel_sha256": decoded_hash, "width": width, "height": height,
                     "duplicate_group": group, "representative": group == relative})
    manifest = {"schema": 1, "image_root": str(root), "scope": "official preliminary train electronic" if not allow_fixture else "fixture",
                "rows": rows, "unique_count": sum(row["representative"] for row in rows)}
    manifest["manifest_sha256"] = digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    expected = manifest.pop("manifest_sha256")
    if digest_json(manifest) != expected:
        raise ValueError("manifest integrity check failed")
    manifest["manifest_sha256"] = expected
    ids = [row["id"] for row in manifest["rows"]]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("manifest IDs are not unique and sorted")
    return manifest


def representatives(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in manifest["rows"] if row["representative"]]


def representative_ids(manifest: dict[str, Any]) -> list[str]:
    return [row["id"] for row in representatives(manifest)]


def reattach_labels(manifest: dict[str, Any], unique_ids: list[str], labels: np.ndarray) -> dict[str, int]:
    if len(unique_ids) != len(labels):
        raise ValueError("label count differs from unique-image count")
    lookup = dict(zip(unique_ids, map(int, labels), strict=True))
    return {row["id"]: lookup[row["duplicate_group"]] for row in manifest["rows"]}
