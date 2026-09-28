"""Audit CLIP crop-to-image agreement as a possible foreground-region gate."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import (load_manifest, load_matrix, read_json, readable_path,
                     representative_ids, write_json)
from .surface import load_surfaces


def analyze(manifest_path: Path, clip_path: Path, surfaces_path: Path,
            regions_path: Path, region_vectors_path: Path,
            image_root: Path, output: Path) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    index = {image_id: i for i, image_id in enumerate(ids)}
    rows = {row["id"]: row for row in manifest["rows"] if row["representative"]}
    clip = load_matrix(clip_path, ids)
    store = load_surfaces(surfaces_path)
    remote = read_json(regions_path)
    if (remote["ids"] != store.ids or
            not np.array_equal(np.asarray(remote["boxes"]), store.boxes) or
            not np.array_equal(np.asarray(remote["image_index"]), store.image_index)):
        raise ValueError("CLIP crops and Gabor/LBP regions are misaligned")
    crops = load_matrix(region_vectors_path)
    if len(crops) != len(store.quality):
        raise ValueError("crop vector count differs from surface regions")
    whole = clip[np.asarray([index[image_id] for image_id in store.ids])]
    scores = np.einsum("ij,ij->i", crops, whole[store.image_index])
    area = np.asarray([
        (box[2] - box[0]) * (box[3] - box[1]) /
        (rows[store.ids[int(image_index)]]["width"] *
         rows[store.ids[int(image_index)]]["height"])
        for box, image_index in zip(store.boxes, store.image_index, strict=True)])
    eligible = area <= 0.5
    top3 = np.zeros(len(scores), dtype=bool)
    top1 = np.zeros(len(scores), dtype=bool)
    for i in range(len(store.ids)):
        indices = store.indices(i, threshold=0)
        indices = indices[eligible[indices]]
        ranked = indices[np.argsort(-scores[indices])]
        top3[ranked[:3]] = True
        top1[ranked[0]] = True
    np.save(output.with_suffix(".npy"), scores.astype(np.float32), allow_pickle=False)
    columns, width, height = 5, 240, 240
    sheet = Image.new("RGB", (columns * width,
                              ((len(store.ids) + columns - 1) // columns) * height),
                      "#eaeef1")
    draw = ImageDraw.Draw(sheet)
    same_top = 0
    for i, image_id in enumerate(store.ids):
        with Image.open(readable_path(image_root / image_id)) as opened:
            image = opened.convert("RGB")
        original_width, original_height = image.size
        image.thumbnail((width - 16, height - 52))
        x = i % columns * width + (width - image.width) // 2
        y = i // columns * height + 8
        sheet.paste(image, (x, y))
        indices = store.indices(i, threshold=0)
        indices = indices[eligible[indices]]
        best_q = indices[np.argmax(store.quality[indices])]
        best_clip = indices[np.argmax(scores[indices])]
        same_top += int(best_q == best_clip)
        for region, colour in ((best_q, "#15ef96"), (best_clip, "#2680ff")):
            x0, y0, x1, y1 = store.boxes[region]
            draw.rectangle((x + x0 * image.width / original_width,
                            y + y0 * image.height / original_height,
                            x + x1 * image.width / original_width,
                            y + y1 * image.height / original_height),
                           outline=colour, width=3)
        draw.text((i % columns * width + 8, (i // columns + 1) * height - 40),
                  f"{i:02d} {image_id[:23]}", fill="#101a22")
        draw.text((i % columns * width + 8, (i // columns + 1) * height - 23),
                  f"q {store.quality[best_q]:.2f}  CLIP {scores[best_clip]:.2f}",
                  fill="#101a22")
    sheet.save(output.with_suffix(".png"))
    summary = {"images": len(store.ids), "regions": len(scores),
               "top1_agreement_with_quality": same_top / len(store.ids),
               "quality_clip_pearson": float(np.corrcoef(
                   store.quality[eligible], scores[eligible])[0, 1]),
               "crop_similarity_p10_p50_p90": np.percentile(
                   scores[eligible], [10, 50, 90]).tolist(),
               "eligible_area_fraction_max": 0.5,
               "top3_regions": int(top3.sum()),
               "top3_quality_gate_045": int(np.sum(top3 & (store.quality >= 0.45)))}
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--region-vectors", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.manifest, args.clip, args.surfaces, args.regions,
                  args.region_vectors, args.images, args.output))


if __name__ == "__main__":
    main()
