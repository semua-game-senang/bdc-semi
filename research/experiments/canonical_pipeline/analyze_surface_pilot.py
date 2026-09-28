"""Small actual-image check of C2 edge coverage and correction magnitude."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import load_manifest, load_matrix, read_json, representative_ids, write_json
from .similarity import build_base, cosine_similarity, fit_excess_reference
from .surface import fit_distance_reference, load_surfaces


def run(manifest_path: Path, clip_path: Path, kec_path: Path,
        surfaces_path: Path, output: Path) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    index = {image_id: i for i, image_id in enumerate(ids)}
    store = load_surfaces(surfaces_path)
    selected = np.asarray([index[image_id] for image_id in store.ids], dtype=int)
    clip = load_matrix(clip_path, ids)[selected]
    kec = load_matrix(kec_path, ids)[selected]
    summary = {"images": len(store.ids), "regions": len(store.quality), "gates": {}}
    for gate in (0.45, 0.7):
        target = output.parent / f"surface-pilot-gate-{gate:g}"
        reference = fit_distance_reference(store, gate=gate, pairs=2000)
        excess = fit_excess_reference(store, cosine_similarity(kec), reference,
                                      candidate_k=10, gate=gate,
                                      samples_per_stratum=50)
        base = build_base(clip, kec, store.ids, store, reference, excess,
                          target, candidate_k=10, strength=0.1, gate=gate)
        changes = read_json(target / "C2_changed_edges.json")
        deltas = np.asarray([change["delta"] for change in changes])
        summary["gates"][str(gate)] = {
            "image_coverage": float(np.mean([
                len(store.indices(i, gate)) > 0 for i in range(len(store.ids))])),
            "candidate_edges": base["candidate_edges"],
            "changed_edges": base["changed_edges"],
            "delta_p10_p50_p90": np.percentile(deltas, [10, 50, 90]).tolist()
            if len(deltas) else None,
            "positive_delta_fraction": float(np.mean(deltas > 0)) if len(deltas) else None,
        }
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--kec", type=Path, required=True)
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.manifest, args.clip, args.kec, args.surfaces, args.output))


if __name__ == "__main__":
    main()
