"""Development-only test of a CLIP-consistent region eligibility rule."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .common import (load_manifest, load_matrix, read_json,
                     representative_ids, write_json)
from .similarity import build_base, cosine_similarity, fit_excess_reference
from .surface import fit_distance_reference, load_surfaces


def run(manifest_path: Path, clip_path: Path, kec_path: Path,
        surfaces_path: Path, score_path: Path, output: Path,
        *, minimum_similarity: float = 0.8, maximum_regions: int = 3) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    rows = {row["id"]: row for row in manifest["rows"] if row["representative"]}
    index = {image_id: i for i, image_id in enumerate(ids)}
    store = load_surfaces(surfaces_path)
    selected_indices = np.asarray([index[image_id] for image_id in store.ids])
    clip = load_matrix(clip_path, ids)[selected_indices]
    kec = load_matrix(kec_path, ids)[selected_indices]
    scores = np.load(score_path, allow_pickle=False)
    if scores.shape != store.quality.shape:
        raise ValueError("semantic score and surface rows differ")
    area = np.asarray([
        (box[2] - box[0]) * (box[3] - box[1]) /
        (rows[store.ids[int(image_index)]]["width"] *
         rows[store.ids[int(image_index)]]["height"])
        for box, image_index in zip(store.boxes, store.image_index, strict=True)])
    keep = np.zeros(len(scores), dtype=bool)
    for i in range(len(store.ids)):
        indices = store.indices(i, threshold=0)
        indices = indices[(area[indices] <= 0.5) &
                          (scores[indices] >= minimum_similarity)]
        ranked = indices[np.argsort(-scores[indices])[:maximum_regions]]
        keep[ranked] = True
    quality = store.quality.copy()
    quality[~keep] = -1
    selected = replace(store, quality=quality)
    reference = fit_distance_reference(selected, gate=0.45, pairs=2000)
    excess = fit_excess_reference(selected, cosine_similarity(kec), reference,
                                  candidate_k=10, gate=0.45,
                                  samples_per_stratum=50)
    target = output.parent / "surface-pilot-semantic-gate"
    base = build_base(clip, kec, store.ids, selected, reference, excess,
                      target, candidate_k=10, strength=0.1, gate=0.45)
    changes = read_json(target / "C2_changed_edges.json")
    deltas = np.asarray([change["delta"] for change in changes])
    summary = {"images": len(store.ids), "regions": len(scores),
               "maximum_region_area_fraction": 0.5,
               "minimum_crop_to_image_clip_cosine": minimum_similarity,
               "maximum_regions_per_image": maximum_regions,
               "selected_regions": int(keep.sum()),
               "image_coverage": float(np.mean([
                   len(selected.indices(i, 0.45)) > 0 for i in range(len(store.ids))])),
               "candidate_edges": base["candidate_edges"],
               "changed_edges": base["changed_edges"],
               "delta_p10_p50_p90": np.percentile(deltas, [10, 50, 90]).tolist()
               if len(deltas) else None}
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--kec", type=Path, required=True)
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.manifest, args.clip, args.kec, args.surfaces,
              args.scores, args.output))


if __name__ == "__main__":
    main()
