"""Modal entrypoints for the canonical experiment in the deltakrist-k workspace.

Run `modal run research/experiments/canonical_pipeline/modal_runner.py::check_gemini`
before uploading data or starting a GPU job. Never print secret values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import modal

app = modal.App("bdc-canonical-experiment")
gemini_secret = modal.Secret.from_name("gemini-api-key", required_keys=["GEMINI_API_KEY"])
check_image = modal.Image.debian_slim(python_version="3.11").pip_install("google-genai>=1,<2")
source = Path(__file__).resolve().parent
manifest_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26,<3", "Pillow>=10,<13")
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(str(source), "/root/research/experiments/canonical_pipeline")
)
experiment_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(source / "requirements.txt"))
    .run_commands("python -m nltk.downloader -d /usr/local/nltk_data wordnet")
    .env({"PYTHONPATH": "/root", "NLTK_DATA": "/usr/local/nltk_data"})
    .add_local_dir(str(source), "/root/research/experiments/canonical_pipeline")
)
volume = modal.Volume.from_name("bdc-canonical-v1", create_if_missing=True)
official_data = modal.Volume.from_name("satriadata-official-data")
remote_out = "/data/outputs/canonical-v1"
remote_images = "/official/data_train/1_Electronic"
remote_config = "/root/research/experiments/canonical_pipeline/config.json"
clip_revision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


@app.function(image=check_image, secrets=[gemini_secret], timeout=120)
def check_gemini() -> dict:
    """One tiny structured request; returns metadata only, never the key."""
    import os
    from google import genai
    from google.genai import types

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is missing from Modal secret")
    model = "gemini-3.5-flash-lite"
    client = genai.Client()
    response = client.models.generate_content(
        model=model, contents="Return the concept name for nouns: cable, wire.",
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema={"type": "object", "properties": {"concept": {"type": "string"}},
                             "required": ["concept"]}))
    if not response.text:
        raise RuntimeError("Gemini returned no structured text")
    return {"secret_present": True, "model": model,
            "json_response_present": True,
            "input_tokens": getattr(response.usage_metadata, "prompt_token_count", None)}


def _arguments(stage: str, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        images=Path(remote_images), out=Path(remote_out), config=Path(remote_config),
        clip_revision=clip_revision, device=device, batch_size=32,
        noun_limit=None, concepts_file=None, fixture=False, stage=stage)


@app.function(image=manifest_image, cpu=2, memory=4096,
              volumes={"/data": volume, "/official": official_data}, timeout=7200)
def manifest_job() -> dict:
    import time
    from research.experiments.canonical_pipeline.common import (
        build_manifest, load_manifest, write_json)

    try:
        started = time.perf_counter()
        manifest = build_manifest(Path(remote_images), expected_count=3961)
        remote_out_path = Path(remote_out)
        remote_out_path.mkdir(parents=True, exist_ok=True)
        target = remote_out_path / "manifest.json"
        if target.exists() and load_manifest(target)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("output directory belongs to a different image manifest")
        write_json(target, manifest)
        summary = {"manifest_sha256": manifest["manifest_sha256"],
                   "rows": len(manifest["rows"]), "unique_count": manifest["unique_count"],
                   "manifest_s": time.perf_counter() - started,
                   "status": "input manifest complete"}
        write_json(remote_out_path / "manifest_run.json", summary)
        return summary
    finally:
        volume.commit()


@app.function(image=experiment_image, gpu="T4", secrets=[gemini_secret],
              volumes={"/data": volume, "/official": official_data},
              timeout=7200)
def semantic_job() -> dict:
    from research.experiments.canonical_pipeline.cli import run

    try:
        return run(_arguments("semantic", "cuda"))
    finally:
        volume.commit()


@app.function(image=experiment_image, cpu=4, memory=8192,
              volumes={"/data": volume, "/official": official_data},
              timeout=10800)
def surface_extract_job() -> dict:
    from research.experiments.canonical_pipeline.cli import run

    try:
        return run(_arguments("surface_extract", "cpu"))
    finally:
        volume.commit()


@app.function(image=experiment_image, cpu=4, memory=8192,
              volumes={"/data": volume, "/official": official_data},
              timeout=10800)
def surface_job() -> dict:
    from research.experiments.canonical_pipeline.cli import run

    try:
        return run(_arguments("surface", "cpu"))
    finally:
        volume.commit()


@app.function(image=experiment_image, cpu=4, memory=8192,
              volumes={"/data": volume}, timeout=3600)
def probe_noun_granularity() -> dict:
    """Check the proposal's N/300, N/100, and N/50 settings without new LLM calls."""
    import numpy as np
    from research.experiments.canonical_pipeline.common import (
        load_matrix, load_manifest, read_json, representative_ids, write_json)
    from research.experiments.canonical_pipeline.kec import _components, discover_nouns

    out = Path(remote_out)
    try:
        manifest = load_manifest(out / "manifest.json")
        ids = representative_ids(manifest)
        x = load_matrix(out / "clip.npy", ids)
        nouns = read_json(out / "noun_vectors.json")["ids"]
        noun_vectors = load_matrix(out / "noun_vectors.npy", nouns)
        cfg = read_json(Path(remote_config))
        summary = {}
        for denominator in (300, 100, 50):
            count = max(2, round(len(ids) / denominator))
            discovery = discover_nouns(
                x, nouns, noun_vectors, seed=cfg["seed"], group_count=count,
                top_k=cfg["noun_top_k"], threshold=0.8)
            matrix = np.asarray(discovery["similarity"], dtype=np.float32)
            summary[str(denominator)] = {
                "seed_groups": count, "concept_groups_beta_08": len(discovery["groups"]),
                "concept_groups_beta_09": len(_components(matrix, 0.9)),
            }
            write_json(out / "sensitivity" / f"noun_discovery_n{denominator}.json", discovery)
        write_json(out / "sensitivity" / "noun_granularity_summary.json", summary)
        return summary
    finally:
        volume.commit()


@app.function(image=experiment_image, cpu=2, memory=4096, secrets=[gemini_secret],
              volumes={"/data": volume}, timeout=3600)
def n50_semantic_job() -> dict:
    """Build the prespecified N/50, beta=0.8 sensitivity from cached embeddings."""
    from research.experiments.canonical_pipeline.common import (
        load_manifest, load_matrix, read_json, representative_ids, write_json)
    from research.experiments.canonical_pipeline.encoders import ClipBackend
    from research.experiments.canonical_pipeline.kec import embed_and_ground, generate_concepts

    out = Path(remote_out)
    target = out / "sensitivity" / "n50_beta08"
    try:
        cfg = read_json(Path(remote_config))
        discovery = read_json(out / "sensitivity" / "noun_discovery_n50.json")
        if (discovery["seed_group_count"] != 79 or discovery["threshold"] != 0.8
                or len(discovery["groups"]) != 3):
            raise ValueError("unexpected N/50 discovery; inspect before generating concepts")
        ids = representative_ids(load_manifest(out / "manifest.json"))
        x = load_matrix(out / "clip.npy", ids)
        target.mkdir(parents=True, exist_ok=True)
        backend = ClipBackend(cfg["clip_model"], clip_revision, "cpu")
        concepts = generate_concepts(discovery, target / "concepts.json", backend,
                                     model=cfg["concept_model"],
                                     temperature=cfg["concept_temperature"])
        metadata = embed_and_ground(backend, concepts, x, ids, target / "kec.npy")
        summary = {"setting": "N/50 seed groups, beta=0.8", "ids": len(ids),
                   "concepts": len(concepts["groups"]), "llm_calls": len(concepts["calls"]),
                   "kec_metadata": metadata, "status": "semantic sensitivity complete"}
        write_json(target / "semantic_run.json", summary)
        return {key: value for key, value in summary.items() if key != "kec_metadata"}
    finally:
        volume.commit()


@app.function(image=experiment_image, cpu=2, memory=4096, secrets=[gemini_secret],
              volumes={"/data": volume}, timeout=3600)
def n300_complete_semantic_job() -> dict:
    """Test complete-linkage noun merging at the original N/300 and beta=0.8."""
    from research.experiments.canonical_pipeline.common import (
        load_manifest, load_matrix, read_json, representative_ids, write_json)
    from research.experiments.canonical_pipeline.encoders import ClipBackend
    from research.experiments.canonical_pipeline.kec import (
        discover_nouns, embed_and_ground, generate_concepts)

    out = Path(remote_out)
    target = out / "sensitivity" / "n300_complete08"
    try:
        cfg = read_json(Path(remote_config))
        ids = representative_ids(load_manifest(out / "manifest.json"))
        x = load_matrix(out / "clip.npy", ids)
        nouns = read_json(out / "noun_vectors.json")["ids"]
        noun_vectors = load_matrix(out / "noun_vectors.npy", nouns)
        discovery = discover_nouns(
            x, nouns, noun_vectors, seed=cfg["seed"], group_count=13,
            top_k=cfg["noun_top_k"], threshold=0.8, merge_mode="complete")
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "noun_discovery.json", discovery)
        backend = ClipBackend(cfg["clip_model"], clip_revision, "cpu")
        concepts = generate_concepts(discovery, target / "concepts.json", backend,
                                     model=cfg["concept_model"],
                                     temperature=cfg["concept_temperature"])
        metadata = embed_and_ground(backend, concepts, x, ids, target / "kec.npy")
        summary = {"setting": "N/300, complete-linkage noun merge at beta=0.8",
                   "ids": len(ids), "concepts": len(concepts["groups"]),
                   "llm_calls": len(concepts["calls"]),
                   "kec_metadata": metadata, "status": "semantic sensitivity complete"}
        write_json(target / "semantic_run.json", summary)
        return {key: value for key, value in summary.items() if key != "kec_metadata"}
    finally:
        volume.commit()


@app.function(image=experiment_image, cpu=2, memory=4096,
              volumes={"/data": volume}, timeout=3600)
def n50_temperature_job() -> dict:
    """Test concept-grounding temperatures; no new noun or LLM requests."""
    from research.experiments.canonical_pipeline.common import (
        load_manifest, load_matrix, read_json, representative_ids, write_json)
    from research.experiments.canonical_pipeline.encoders import ClipBackend
    from research.experiments.canonical_pipeline.kec import embed_and_ground

    out = Path(remote_out)
    parent = out / "sensitivity" / "n50_beta08"
    try:
        cfg = read_json(Path(remote_config))
        ids = representative_ids(load_manifest(out / "manifest.json"))
        x = load_matrix(out / "clip.npy", ids)
        concepts = read_json(parent / "concepts.json")
        backend = ClipBackend(cfg["clip_model"], clip_revision, "cpu")
        summary = {}
        for temperature in (0.5, 0.25, 0.1):
            target = parent / f"grounding_t{temperature:g}"
            target.mkdir(parents=True, exist_ok=True)
            metadata = embed_and_ground(
                backend, concepts, x, ids, target / "kec.npy",
                grounding_temperature=temperature)
            summary[str(temperature)] = {"ids": len(ids), "concepts": len(concepts["groups"]),
                                         "kec_metadata": metadata}
        write_json(parent / "grounding_temperature_run.json", summary)
        return {key: {"ids": value["ids"], "concepts": value["concepts"]}
                for key, value in summary.items()}
    finally:
        volume.commit()


@app.function(image=experiment_image, cpu=2, memory=8192,
              volumes={"/data": volume}, timeout=3600)
def complete_linkage_job() -> dict:
    """Fit a matched complete-linkage sensitivity from the finished C0/C1/C2 distances."""
    import numpy as np
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from research.experiments.canonical_pipeline.common import (
        load_manifest, load_matrix, read_json, representative_ids,
        representatives, save_matrix, write_json)
    from research.experiments.canonical_pipeline.tree import hierarchy_tree

    out = Path(remote_out)
    target = out / "sensitivity" / "complete_linkage"
    try:
        if not (out / "run.json").exists():
            raise ValueError("finish canonical C0/C1/C2 distances before comparing linkage")
        if (target / "run.json").exists():
            return read_json(target / "run.json")
        cfg = read_json(Path(remote_config))
        manifest = load_manifest(out / "manifest.json")
        ids = representative_ids(manifest)
        resolution = np.asarray([int(row["width"] == row["height"] == 150)
                                 for row in representatives(manifest)])
        target.mkdir(parents=True, exist_ok=True)
        trees = {}
        for name in ("C0", "C1", "C2"):
            distance = load_matrix(out / "base" / f"{name}_distance.npy", ids)
            tree = hierarchy_tree(distance, broad=cfg["broad_clusters"],
                                  fine=cfg["fine_clusters"], method="complete")
            trees[name] = tree
            np.save(target / f"{name}_linkage.npy", tree["linkage"], allow_pickle=False)
            save_matrix(target / f"{name}_labels.npy",
                        np.column_stack([tree["broad"], tree["fine"]]), ids,
                        {"method": name, "linkage": "complete", "columns": ["broad", "fine"]})
        summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
                   "linkage": "complete", "methods": {}}
        for name, tree in trees.items():
            summary["methods"][name] = {
                "broad_sizes": sorted(np.bincount(tree["broad"])[1:].tolist(), reverse=True),
                "fine_sizes": sorted(np.bincount(tree["fine"])[1:].tolist(), reverse=True),
                "broad_resolution_ami": float(adjusted_mutual_info_score(
                    resolution, tree["broad"])),
                "fine_resolution_ami": float(adjusted_mutual_info_score(
                    resolution, tree["fine"])),
                "broad_ari_vs_c0": float(adjusted_rand_score(
                    trees["C0"]["broad"], tree["broad"])),
                "fine_ari_vs_c0": float(adjusted_rand_score(
                    trees["C0"]["fine"], tree["fine"])),
            }
        write_json(target / "run.json", summary)
        return summary
    finally:
        volume.commit()


@app.function(image=experiment_image, gpu="T4",
              volumes={"/data": volume, "/official": official_data}, timeout=3600)
def pilot_region_clip_job() -> dict:
    """Encode only the fixed 40-image development pilot's native crop proposals."""
    from PIL import Image
    from research.experiments.canonical_pipeline.common import (
        load_manifest, read_json, save_matrix, write_json)
    from research.experiments.canonical_pipeline.encoders import ClipBackend
    from research.experiments.canonical_pipeline.surface import propose_boxes

    out = Path(remote_out)
    target = out / "sensitivity" / "pilot_region_clip"
    try:
        if (target / "region_vectors.npy").exists():
            return read_json(target / "run.json")
        manifest = load_manifest(out / "manifest.json")
        rows = {row["id"]: row for row in manifest["rows"] if row["representative"]}
        pilot = read_json(Path(remote_config).with_name("pilot_ids.json"))
        ids = pilot["ids"]
        if len(ids) != 40 or len(set(ids)) != 40 or not set(ids).issubset(rows):
            raise ValueError("pilot IDs are missing or not distinct representatives")
        backend = ClipBackend("openai/clip-vit-base-patch32", clip_revision, "cuda")
        crops, crop_ids, image_index, boxes = [], [], [], []
        for i, image_id in enumerate(ids):
            row = rows[image_id]
            with Image.open(row["path"]) as opened:
                image = opened.convert("RGB")
            for j, proposed in enumerate(propose_boxes(*image.size, 48)):
                x0, y0, x1, y1 = proposed
                if x1 - x0 > 256:
                    mid = (x0 + x1) // 2
                    x0, x1 = mid - 128, mid + 128
                if y1 - y0 > 256:
                    mid = (y0 + y1) // 2
                    y0, y1 = mid - 128, mid + 128
                box = (x0, y0, x1, y1)
                crops.append(image.crop(box))
                crop_ids.append(f"{image_id}#region{j}")
                image_index.append(i)
                boxes.append(box)
            image.close()
        vectors = backend.encode_pil_images(crops)
        for crop in crops:
            crop.close()
        target.mkdir(parents=True, exist_ok=True)
        save_matrix(target / "region_vectors.npy", vectors, crop_ids,
                    backend.metadata() | {"manifest_sha256": manifest["manifest_sha256"]})
        write_json(target / "regions.json", {
            "ids": ids, "image_index": image_index, "boxes": boxes,
            "pilot": "40-image development surface sample"})
        summary = {"images": len(ids), "regions": len(crops),
                   "status": "CLIP crop vectors for semantic foreground diagnostic"}
        write_json(target / "run.json", summary)
        return summary
    finally:
        volume.commit()


@app.function(image=experiment_image, cpu=2, memory=4096,
              volumes={"/data": volume}, timeout=3600)
def prompt_text_bank_job() -> dict:
    """Encode fixed object prompts with the same pinned CLIP text encoder."""
    from research.experiments.canonical_pipeline.common import save_matrix, write_json
    from research.experiments.canonical_pipeline.encoders import ClipBackend
    from research.experiments.canonical_pipeline.prompt_relation_audit import PROMPTS

    target = Path(remote_out) / "sensitivity" / "text_bank_v1"
    try:
        if (target / "text_vectors.npy").exists():
            raise ValueError("text bank v1 already exists; use a new versioned path")
        backend = ClipBackend("openai/clip-vit-base-patch32", clip_revision, "cpu")
        vectors = backend.encode_texts(list(PROMPTS))
        target.mkdir(parents=True, exist_ok=False)
        save_matrix(target / "text_vectors.npy", vectors, list(PROMPTS), backend.metadata())
        summary = {"prompts": len(PROMPTS), "clip_revision": backend.revision,
                   "status": "fixed prompt bank encoded"}
        write_json(target / "run.json", summary)
        return summary
    finally:
        volume.commit()


@app.local_entrypoint()
def main(stage: str = "check") -> None:
    if stage == "check":
        print(check_gemini.remote())
    elif stage == "manifest":
        print(manifest_job.remote())
    elif stage == "semantic":
        print(semantic_job.remote())
    elif stage == "probe":
        print(probe_noun_granularity.remote())
    elif stage == "n50_semantic":
        print(n50_semantic_job.remote())
    elif stage == "n300_complete_semantic":
        print(n300_complete_semantic_job.remote())
    elif stage == "n50_temperature":
        print(n50_temperature_job.remote())
    elif stage == "complete_linkage":
        print(complete_linkage_job.remote())
    elif stage == "pilot_region_clip":
        print(pilot_region_clip_job.remote())
    elif stage == "prompt_text_bank":
        print(prompt_text_bank_job.remote())
    elif stage == "surface_extract":
        print(surface_extract_job.remote())
    elif stage == "surface":
        print(surface_job.remote())
    else:
        raise ValueError("stage must be check, manifest, semantic, probe, n50_semantic, n300_complete_semantic, n50_temperature, pilot_region_clip, prompt_text_bank, surface_extract, surface, or complete_linkage")
