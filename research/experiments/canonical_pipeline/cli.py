"""Run only the canonical C0, C1, and C2 Proposal 2 experiment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import (build_manifest, load_manifest, load_matrix, read_json,
                     representative_ids, representatives, save_matrix,
                     sha256_file, write_json)
from .encoders import ClipBackend, extract_clip
from .evaluation import (evaluate_method, freeze_selection, load_exp009_reviews,
                         load_reviews, verify_selection)
from .exp012 import audit_exp012
from .kec import discover_nouns, embed_and_ground, generate_concepts, wordnet_nouns
from .outputs import evidence_cards, hierarchy_plot
from .similarity import build_base, cosine_similarity, fit_excess_reference
from .surface import extract_surfaces, fit_distance_reference, load_surfaces
from .tree import average_linkage_tree


METHODS = ("C0", "C1", "C2")


def config(path: Path) -> dict:
    value = read_json(path)
    if value.get("schema") != 1:
        raise ValueError("unsupported configuration schema")
    return value


def _save_tree(directory: Path, name: str, tree: dict, ids: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{name}_linkage.npy", tree["linkage"], allow_pickle=False)
    save_matrix(directory / f"{name}_labels.npy",
                np.column_stack([tree["broad"], tree["fine"]]), ids,
                {"method": name, "columns": ["broad", "fine"]})


def run(args: argparse.Namespace) -> dict:
    cfg = config(args.config)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    stage = getattr(args, "stage", "all")
    if stage in ("all", "manifest"):
        start = time.perf_counter()
        manifest = build_manifest(args.images, allow_fixture=args.fixture,
                                  expected_count=None if args.fixture else 3961)
        if (out / "manifest.json").exists():
            prior = load_manifest(out / "manifest.json")
            if prior["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ValueError("output directory belongs to a different image manifest")
        write_json(out / "manifest.json", manifest)
        timings["manifest_s"] = time.perf_counter() - start
        if stage == "manifest":
            summary = {"manifest_sha256": manifest["manifest_sha256"],
                       "rows": len(manifest["rows"]), "unique_count": manifest["unique_count"],
                       "timings": timings, "status": "input manifest complete"}
            write_json(out / "manifest_run.json", summary)
            return summary
    else:
        manifest = load_manifest(out / "manifest.json")
        if manifest["image_root"] != str(args.images.resolve(strict=True)):
            raise ValueError("manifest image root differs from requested images")
        if manifest["scope"] == "fixture" and not args.fixture:
            raise ValueError("a fixture semantic stage requires --fixture in the surface stage")
        if manifest["scope"] != "fixture" and args.fixture:
            raise ValueError("official semantic stage cannot resume as a fixture")
    rows = representatives(manifest)
    ids = representative_ids(manifest)
    if stage in ("all", "semantic"):
        revision = args.clip_revision or cfg["clip_revision"]
        if not revision and not args.fixture:
            raise ValueError("a pinned --clip-revision is required for an official run")
        start = time.perf_counter()
        clip_path = out / "clip.npy"
        if clip_path.exists():
            cached = read_json(clip_path.with_suffix(".json"))["metadata"]
            if (cached["model"] != cfg["clip_model"] or cached["revision"] != revision
                    or cached["manifest_sha256"] != manifest["manifest_sha256"]):
                raise ValueError("cached CLIP vectors belong to a different input or encoder")
        else:
            extract_clip(manifest, clip_path, model_id=cfg["clip_model"],
                         revision=revision, device=args.device, batch_size=args.batch_size)
        x = load_matrix(out / "clip.npy", ids)
        backend = ClipBackend(cfg["clip_model"], revision, args.device)
        timings["clip_s"] = time.perf_counter() - start

        start = time.perf_counter()
        if args.noun_limit and not args.fixture:
            raise ValueError("--noun-limit is only permitted for fixtures")
        nouns = wordnet_nouns(limit=args.noun_limit)
        noun_path = out / "noun_vectors.npy"
        if noun_path.exists():
            cached = read_json(noun_path.with_suffix(".json"))["metadata"]
            if (cached["model"] != cfg["clip_model"] or cached["revision"] != revision
                    or cached["template"] != "A photo of [CLASS]"):
                raise ValueError("cached noun vectors belong to a different encoder or template")
            noun_vectors = load_matrix(noun_path, nouns)
        else:
            noun_vectors = backend.encode_texts([f"A photo of {noun}" for noun in nouns],
                                                 batch_size=args.batch_size)
            save_matrix(noun_path, noun_vectors, nouns,
                        backend.metadata() | {"template": "A photo of [CLASS]"})
        group_count = (max(2, round(len(rows) / 300))
                       if cfg["kec_seed_groups"] == "N/300"
                       else int(cfg["kec_seed_groups"]))
        discovery = discover_nouns(x, nouns, noun_vectors, seed=cfg["seed"],
                                   group_count=group_count, top_k=cfg["noun_top_k"],
                                   threshold=cfg["kec_merge_threshold"])
        write_json(out / "noun_discovery.json", discovery)
        timings["noun_discovery_s"] = time.perf_counter() - start

        start = time.perf_counter()
        if args.concepts_file:
            concepts = read_json(args.concepts_file)
            if len(concepts["groups"]) != len(discovery["groups"]):
                raise ValueError("provided concept count differs from discovered groups")
            write_json(out / "concepts.json", concepts)
        else:
            concepts = generate_concepts(discovery, out / "concepts.json", backend,
                                         model=cfg["concept_model"],
                                         temperature=cfg["concept_temperature"])
        embed_and_ground(backend, concepts, x, ids, out / "kec.npy")
        timings["kec_s"] = time.perf_counter() - start
        if stage == "semantic":
            summary = {"manifest_sha256": manifest["manifest_sha256"], "ids": len(ids),
                       "timings": timings, "config_sha256": sha256_file(args.config),
                       "status": "semantic stage complete; surface stage pending"}
            write_json(out / "semantic_run.json", summary)
            return summary
    else:
        semantic = read_json(out / "semantic_run.json")
        if semantic["manifest_sha256"] != manifest["manifest_sha256"] or semantic["config_sha256"] != sha256_file(args.config):
            raise ValueError("semantic stage manifest or configuration differs")
        x = load_matrix(out / "clip.npy", ids)
    kec = load_matrix(out / "kec.npy", ids)

    start = time.perf_counter()
    surface_path = out / "surfaces.npz"
    if not surface_path.exists():
        extract_surfaces(rows, surface_path, minimum=cfg["min_native_region"],
                         gate=cfg["region_quality_threshold"])
    surfaces = load_surfaces(surface_path, ids)
    surface_metadata = read_json(surface_path.with_suffix(".json"))
    if (surface_metadata["minimum_native_pixels"] != cfg["min_native_region"]
            or surface_metadata["gate"] != cfg["region_quality_threshold"]):
        raise ValueError("cached surface regions use different proposal or gate settings")
    if stage == "surface_extract":
        summary = {"manifest_sha256": manifest["manifest_sha256"],
                   "ids": len(ids), "regions": surface_metadata["region_count"],
                   "accepted": surface_metadata["accepted_count"],
                   "timings": {"surface_extraction_s": time.perf_counter() - start},
                   "config_sha256": sha256_file(args.config),
                   "status": "surface regions extracted; clustering pending"}
        write_json(out / "surface_extract_run.json", summary)
        return summary
    distance_reference = fit_distance_reference(
        surfaces, gate=cfg["region_quality_threshold"],
        pairs=cfg["distance_reference_pairs"], seed=cfg["seed"])
    write_json(out / "surface_distance_reference.json", distance_reference)
    excess_reference = fit_excess_reference(
        surfaces, cosine_similarity(kec), distance_reference,
        candidate_k=cfg["candidate_k"], gate=cfg["region_quality_threshold"],
        samples_per_stratum=cfg["reference_pairs_per_stratum"], seed=cfg["seed"])
    write_json(out / "surface_excess_reference.json", excess_reference)
    timings["surface_s"] = time.perf_counter() - start

    start = time.perf_counter()
    base = build_base(x, kec, ids, surfaces, distance_reference,
                      excess_reference, out / "base",
                      candidate_k=cfg["candidate_k"],
                      strength=cfg["surface_lambda"],
                      gate=cfg["region_quality_threshold"])
    for name in METHODS:
        distance = load_matrix(out / "base" / f"{name}_distance.npy", ids)
        tree = average_linkage_tree(distance, broad=cfg["broad_clusters"],
                                    fine=cfg["fine_clusters"])
        _save_tree(out / "trees", name, tree, ids)
        hierarchy_plot(tree["linkage"], out / "figures" / f"{name}_hierarchy.png",
                       title=f"{name} average-linkage hierarchy")
        if name == "C2":
            evidence_cards(rows, distance, tree["broad"], tree["fine"],
                           out / "figures" / "evidence_cards",
                           surface=surfaces, gate=cfg["region_quality_threshold"])
    timings["base_and_trees_s"] = time.perf_counter() - start
    summary = {"manifest_sha256": manifest["manifest_sha256"],
               "ids": len(ids), "methods": list(METHODS), "timings": timings,
               "base": base, "config_sha256": sha256_file(args.config),
               "status": "unreviewed development artifacts; no improvement claim"}
    write_json(out / "run.json", summary)
    return summary


def evaluate(args: argparse.Namespace) -> dict:
    cfg = config(args.config)
    out = args.out.resolve()
    manifest = load_manifest(out / "manifest.json")
    ids = representative_ids(manifest)
    frozen = verify_selection(args.selection, config_path=args.config,
                              review_id_path=args.review_ids)
    if frozen["method"] not in METHODS:
        raise ValueError("selection is outside the canonical C0/C1/C2 methods")
    required = [out / "base" / f"{method}_{kind}.npy"
                for method in METHODS for kind in ("similarity", "distance")]
    required += [out / "trees" / f"{method}_labels.npy" for method in METHODS]
    missing = [str(path) for path in required
               if str(path.resolve()) not in frozen["artifacts"]]
    if missing:
        raise ValueError(f"evaluation artifacts were not frozen: {missing}")
    if args.review_ids.suffix.lower() == ".csv":
        reviews = load_exp009_reviews(args.review_ids, args.reviews, manifest)
    else:
        expected_ids = set(read_json(args.review_ids)["ids"])
        reviews = load_reviews(args.reviews,
                               {row["id"] for row in manifest["rows"]})
        if {row["id"] for row in reviews} != expected_ids:
            raise ValueError("human reviews do not exactly match locked IDs")
    rep_by_id = {row["id"]: row["duplicate_group"] for row in manifest["rows"]}
    unique: dict[str, dict] = {}
    for row in reviews:
        representative = rep_by_id[row["id"]]
        row["id"] = representative
        if representative in unique:
            prior = unique[representative]
            fields = ("family", "visible_surfaces", "surface_observable",
                      "source_stratum", "mixed")
            if any(prior[key] != row[key] for key in fields):
                raise ValueError("duplicate-group reviews disagree")
        else:
            unique[representative] = row
    reviews = list(unique.values())
    results = {}
    for method in METHODS:
        similarity = load_matrix(out / "base" / f"{method}_similarity.npy", ids)
        distance = load_matrix(out / "base" / f"{method}_distance.npy", ids)
        labels = load_matrix(out / "trees" / f"{method}_labels.npy", ids).astype(int)
        results[method] = evaluate_method(similarity, ids, reviews,
                                          labels[:, 0], labels[:, 1],
                                          candidate_k=cfg["candidate_k"],
                                          distance=distance)
    result = {"selection_sha256": frozen["selection_sha256"],
              "review_count": len(reviews), "methods": results,
              "note": "visible surfaces are proxies, not chemistry or treatment labels"}
    write_json(out / "locked_evaluation.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="build C0, C1, C2 and their trees")
    command.add_argument("--images", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    command.add_argument("--clip-revision")
    command.add_argument("--device", default="cpu")
    command.add_argument("--stage", choices=("all", "manifest", "semantic", "surface_extract", "surface"), default="all",
                         help="manifest and surface run on CPU; semantic runs on GPU")
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--noun-limit", type=int,
                         help="fixture only; omit for an official run")
    command.add_argument("--concepts-file", type=Path,
                         help="audited concept JSON instead of API generation")
    command.add_argument("--fixture", action="store_true",
                         help="permit a small isolated smoke-test folder")
    audit = commands.add_parser("audit-exp012", help="audit region gate on development review")
    audit.add_argument("--surfaces", type=Path, required=True)
    audit.add_argument("--region-manifest", type=Path, required=True)
    audit.add_argument("--review", type=Path, required=True)
    audit.add_argument("--manifest", type=Path,
                       help="verified manifest for exact-duplicate ID mapping")
    audit.add_argument("--out", type=Path, required=True)
    freeze = commands.add_parser("freeze", help="lock one C0/C1/C2 choice before review")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--run", type=Path, required=True)
    freeze.add_argument("--artifact", type=Path, action="append", default=[])
    freeze.add_argument("--review-ids", type=Path, required=True)
    freeze.add_argument("--method", choices=METHODS, required=True)
    freeze.add_argument("--broad", type=int, required=True)
    freeze.add_argument("--fine", type=int, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="score locked EXP009 human review")
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--selection", type=Path, required=True)
    evaluation.add_argument("--review-ids", type=Path, required=True)
    evaluation.add_argument("--reviews", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args)
    elif args.command == "audit-exp012":
        result = audit_exp012(load_surfaces(args.surfaces), args.region_manifest,
                              args.review, args.out, manifest_path=args.manifest)
    elif args.command == "freeze":
        settings = config(args.config)
        if (args.broad, args.fine) != (settings["broad_clusters"],
                                       settings["fine_clusters"]):
            raise ValueError("frozen cuts must match the run configuration")
        run_dir = args.run.resolve()
        if not (run_dir / "run.json").exists():
            raise FileNotFoundError("completed canonical run is required")
        paths = [run_dir / name for name in
                 ("manifest.json", "run.json", "concepts.json",
                  "surfaces.npz", "surfaces.json")]
        for subdir in ("base", "trees"):
            paths.extend(path for path in (run_dir / subdir).rglob("*")
                         if path.is_file() and path.suffix in {".npy", ".npz", ".json"})
        paths.extend(args.artifact)
        result = freeze_selection(args.out, config_path=args.config,
                                  artifact_paths=sorted(set(paths)),
                                  method=args.method, broad=args.broad,
                                  fine=args.fine, review_id_path=args.review_ids)
    else:
        result = evaluate(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
