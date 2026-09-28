"""KEC noun discovery, concept construction, and frozen image grounding."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .common import digest_json, load_matrix, read_json, save_matrix, unit_rows, write_json
from .encoders import ClipBackend


def wordnet_nouns(limit: int | None = None) -> list[str]:
    from nltk.corpus import wordnet as wn
    try:
        names = {lemma.name().replace("_", " ").strip().lower()
                 for synset in wn.all_synsets(pos=wn.NOUN) for lemma in synset.lemmas()}
    except LookupError as error:
        raise RuntimeError("NLTK WordNet is missing; run: python -m nltk.downloader wordnet") from error
    names = sorted(name for name in names if name and len(name) < 70)
    return names[:limit] if limit else names


def spherical_kmeans(x: np.ndarray, k: int, seed: int, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    x = unit_rows(x)
    if not 1 <= k <= len(x):
        raise ValueError("invalid spherical K-means group count")
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    labels = np.full(len(x), -1, dtype=np.int32)
    for _ in range(iterations):
        proposed = np.argmax(x @ centers.T, axis=1).astype(np.int32)
        if np.array_equal(proposed, labels):
            break
        labels = proposed
        updated = []
        for group in range(k):
            members = x[labels == group]
            updated.append(members.mean(axis=0) if len(members) else x[rng.integers(len(x))])
        centers = unit_rows(np.stack(updated))
    return labels, centers


def _components(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    n = len(similarity)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _complete_groups(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    """Prevent a chain of weak links from collapsing all noun seed groups."""
    if len(similarity) == 1:
        return [[0]]
    distance = np.clip(1 - np.asarray(similarity, dtype=np.float64), 0, 2)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=True), method="complete")
    labels = fcluster(tree, t=1 - threshold, criterion="distance")
    groups = [[int(i) for i in np.flatnonzero(labels == label)]
              for label in np.unique(labels)]
    return sorted(groups, key=lambda group: group[0])


def discover_nouns(x: np.ndarray, nouns: list[str], noun_vectors: np.ndarray, *,
                   seed: int, group_count: int, top_k: int = 5,
                   threshold: float = 0.8, merge_mode: str = "components") -> dict:
    if len(nouns) != len(noun_vectors):
        raise ValueError("noun/vector alignment failed")
    _, image_centers = spherical_kmeans(x, group_count, seed)
    noun_vectors = unit_rows(noun_vectors)
    similarities = image_centers @ noun_vectors.T
    indices = np.argsort(-similarities, axis=1, kind="stable")[:, :top_k]
    noun_centers = unit_rows(np.stack([noun_vectors[ids].mean(axis=0) for ids in indices]))
    r = 0.8 * (image_centers @ image_centers.T) + 0.2 * (noun_centers @ noun_centers.T)
    if merge_mode == "components":
        merged = _components(r, threshold)
    elif merge_mode == "complete":
        merged = _complete_groups(r, threshold)
    else:
        raise ValueError("merge_mode must be components or complete")
    groups = []
    for group_id, seed_groups in enumerate(merged):
        words = sorted({nouns[int(index)] for seed_group in seed_groups for index in indices[seed_group]})
        groups.append({"id": group_id, "seed_groups": seed_groups, "nouns": words})
    return {"seed": seed, "seed_group_count": group_count, "top_k": top_k,
            "threshold": threshold, "merge_mode": merge_mode,
            "seed_nouns": [[nouns[int(index)] for index in row] for row in indices],
            "groups": groups, "similarity": r.tolist()}


CONCEPT_SCHEMA = {
    "type": "object", "properties": {
        "concept": {"type": "string"}, "description": {"type": "string"}},
    "required": ["concept", "description"], "additionalProperties": False}
UNI_SCHEMA = {
    "type": "object", "properties": {
        "attributes": {"type": "array", "items": {"type": "string"}}},
    "required": ["attributes"], "additionalProperties": False}
PAIR_SCHEMA = {
    "type": "object", "properties": {"attribute": {"type": "string"}},
    "required": ["attribute"], "additionalProperties": False}


def select_concept_pairs(concept_vectors: np.ndarray, threshold: float = 0.8) -> list[tuple[int, int]]:
    """KEC's cumulative normalized-similarity neighbor selection, symmetrized."""
    vectors = unit_rows(concept_vectors)
    if not 0 < threshold <= 1:
        raise ValueError("pair-selection threshold must be in (0,1]")
    similarity = vectors @ vectors.T
    pairs = set()
    for i in range(len(vectors)):
        others = [j for j in range(len(vectors)) if j != i]
        if not others:
            continue
        probabilities = softmax(similarity[i, others])
        order = np.argsort(-probabilities, kind="stable")
        cumulative = 0.0
        for position in order:
            j = others[int(position)]
            pairs.add((min(i, j), max(i, j)))
            cumulative += float(probabilities[position])
            if cumulative >= threshold:
                break
    return sorted(pairs)


def _request_json(client: Any, *, model: str, temperature: float, prompt: str,
                  schema: dict, name: str) -> tuple[dict, dict]:
    import json

    system = "Return concise, visually grounded clustering knowledge as JSON. Do not infer chemical composition or hazards."
    if model.startswith("gemini-"):
        from google.genai import types

        response = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, system_instruction=system,
                response_mime_type="application/json",
                response_schema={key: value for key, value in schema.items()
                                 if key != "additionalProperties"}))
        raw = response.text
        if not raw:
            raise RuntimeError("Gemini concept request returned no text")
        usage = response.usage_metadata.model_dump(exclude_none=True) if response.usage_metadata else None
        response_id = getattr(response, "response_id", None)
    else:
        response = client.responses.create(
            model=model, temperature=temperature,
            input=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}})
        if response.status != "completed" or not response.output_text:
            raise RuntimeError(f"concept request incomplete: {response.status}")
        raw = response.output_text
        usage = response.usage.model_dump() if response.usage else None
        response_id = response.id
    data = json.loads(raw)
    if not isinstance(data, dict) or any(key not in data for key in schema["required"]):
        raise ValueError("concept response does not match required JSON fields")
    record = {"model": model, "response_id": response_id, "raw_output": raw,
              "prompt": prompt, "usage": usage}
    return data, record


def generate_concepts(discovery: dict, output: Path, backend: ClipBackend, *, model: str = "gemini-3.5-flash-lite",
                      temperature: float = 0.1) -> dict:
    """Resume each concept and pair. The API sees noun groups, never individual images."""
    if model.startswith("gemini-"):
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is absent; attach Modal secret gemini-api-key")
        from google import genai
        client = genai.Client()
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is absent; set it locally or supply an audited concepts JSON")
        from openai import OpenAI
        client = OpenAI(max_retries=2)
    current = read_json(output) if output.exists() else {
        "model": model, "temperature": temperature, "groups": {}, "pairs": {}, "calls": []}
    if current["model"] != model or current["temperature"] != temperature:
        raise ValueError("concept cache model/settings differ")
    if set(current["groups"]) - {str(group["id"]) for group in discovery["groups"]}:
        raise ValueError("concept cache has groups absent from noun discovery")
    for group in discovery["groups"]:
        cached = current["groups"].get(str(group["id"]))
        if cached is not None and cached["nouns"] != group["nouns"]:
            raise ValueError("concept cache nouns differ from noun discovery")
    for group in discovery["groups"]:
        key = str(group["id"])
        if key in current["groups"]:
            continue
        prompt = ("Represent these CLIP-retrieved WordNet nouns by one broad visible-object "
                  "concept and a concise visual description. "
                  f"Nouns: {group['nouns']}")
        data, record = _request_json(client, model=model, temperature=temperature,
                                     prompt=prompt, schema=CONCEPT_SCHEMA, name="kec_concept")
        current["groups"][key] = data | {"nouns": group["nouns"]}
        current["calls"].append(record)
        write_json(output, current)
    ordered = [current["groups"][str(i)] for i in range(len(discovery["groups"]))]
    for i, group in enumerate(ordered):
        if "attributes" in group:
            continue
        others = [item["concept"] for j, item in enumerate(ordered) if j != i]
        prompt = ("Give exactly two visible attributes that help distinguish instances of "
                  f"{group['concept']} from the other concepts {others}. "
                  "Avoid chemistry, value, and hazard claims.")
        data, record = _request_json(client, model=model, temperature=temperature,
                                     prompt=prompt, schema=UNI_SCHEMA, name="kec_uni_attributes")
        if len(data["attributes"]) != 2:
            raise ValueError("uni-concept request must produce exactly two attributes")
        current["groups"][str(i)]["attributes"] = data["attributes"]
        current["calls"].append(record)
        write_json(output, current)
    concept_vectors = backend.encode_texts([item["concept"] for item in ordered])
    for i, j in select_concept_pairs(concept_vectors, discovery["threshold"]):
        key = f"{i}:{j}"
        if key in current["pairs"]:
            continue
        prompt = ("Give one visible attribute that distinguishes these two visual concepts: "
                  f"A: {ordered[i]['concept']} ({ordered[i]['description']}); "
                  f"B: {ordered[j]['concept']} ({ordered[j]['description']}). "
                  "Avoid chemistry, value, and hazard claims.")
        data, record = _request_json(client, model=model, temperature=temperature,
                                     prompt=prompt, schema=PAIR_SCHEMA, name="kec_pair")
        current["pairs"][key] = data
        current["calls"].append(record)
        write_json(output, current)
    return current


def ground_kec(x: np.ndarray, names: np.ndarray, descriptions: np.ndarray,
               attribute_means: np.ndarray, *,
               grounding_temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Paper-style c, a, kappa, and raw [x;kappa] with no block rescaling."""
    x = unit_rows(x)
    if names.shape != descriptions.shape or names.shape != attribute_means.shape:
        raise ValueError("concept embedding dimensions differ")
    if grounding_temperature <= 0:
        raise ValueError("grounding temperature must be positive")
    zeta = names + descriptions
    weights = softmax((x @ zeta.T) / grounding_temperature, axis=1)
    context = weights @ zeta
    attributes = x * (weights @ attribute_means)
    raw = np.concatenate([x, context + attributes], axis=1).astype(np.float32)
    return raw, weights.astype(np.float32)


def embed_and_ground(backend: ClipBackend, concepts: dict, x: np.ndarray,
                     ids: list[str], output: Path, *,
                     grounding_temperature: float = 1.0) -> dict:
    keys = sorted(concepts["groups"], key=int)
    groups = [concepts["groups"][key] for key in keys]
    names = backend.encode_texts([group["concept"] for group in groups])
    descriptions = backend.encode_texts([group["description"] for group in groups])
    attributes = [list(group["attributes"]) for group in groups]
    for pair_key, item in concepts["pairs"].items():
        i, j = map(int, pair_key.split(":"))
        attributes[i].append(item["attribute"])
        attributes[j].append(item["attribute"])
    flat = [text for item in attributes for text in item]
    embedded = backend.encode_texts(flat)
    means = []
    cursor = 0
    for item in attributes:
        means.append(embedded[cursor:cursor + len(item)].mean(axis=0))
        cursor += len(item)
    raw, weights = ground_kec(x, names, descriptions, np.stack(means),
                              grounding_temperature=grounding_temperature)
    metadata = backend.metadata() | {
        "concept_count": len(groups), "source": "KEC paper-style representation",
        "concept_model": concepts.get("model"),
        "concept_temperature": concepts.get("temperature"),
        "grounding_temperature": grounding_temperature,
        "concepts_sha256": digest_json(concepts),
    }
    save_matrix(output.with_name(output.stem + "_raw.npy"), raw, ids, metadata)
    save_matrix(output, unit_rows(raw), ids, metadata | {"l2_normalized_for_cosine": True})
    save_matrix(output.with_name(output.stem + "_weights.npy"), weights, ids, metadata)
    return metadata
