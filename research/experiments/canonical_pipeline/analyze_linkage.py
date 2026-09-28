"""Compare portable linkage rules before applying the canonical C2 distance."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score

from .common import (load_manifest, load_matrix, readable_path,
                     representative_ids, representatives, write_json)


def _gallery(rows: list[dict], vectors: np.ndarray, labels: np.ndarray,
             image_root: Path, output: Path) -> None:
    groups = np.unique(labels)
    tile_width, tile_height = 84, 84
    panel_width, panel_height = 4 * tile_width + 12, 2 * tile_height + 40
    columns = 3
    sheet = Image.new("RGB", (columns * panel_width,
                              ((len(groups) + columns - 1) // columns) * panel_height),
                      "#eef1f4")
    draw = ImageDraw.Draw(sheet)
    for position, group in enumerate(groups):
        members = np.flatnonzero(labels == group)
        center = vectors[members].mean(axis=0)
        scores = vectors[members] @ center
        selected = members[np.argsort(-scores)[:8]]
        origin_x = (position % columns) * panel_width
        origin_y = (position // columns) * panel_height
        draw.text((origin_x + 5, origin_y + 4), f"Cluster {int(group)}  n={len(members)}",
                  fill="#12202a")
        for index, selected_index in enumerate(selected):
            image_id = rows[int(selected_index)]["id"]
            with Image.open(readable_path(image_root / image_id)) as opened:
                image = opened.convert("RGB")
            image.thumbnail((tile_width - 4, tile_height - 4))
            x = origin_x + 2 + (index % 4) * tile_width
            y = origin_y + 28 + (index // 4) * tile_height
            sheet.paste(image, (x + (tile_width - image.width) // 2,
                                y + (tile_height - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def analyze(manifest_path: Path, clip_path: Path,
            image_root: Path | None, output: Path) -> dict:
    manifest = load_manifest(manifest_path)
    ids = representative_ids(manifest)
    rows = representatives(manifest)
    vectors = load_matrix(clip_path, ids)
    resolution = np.asarray([int(row["width"] == row["height"] == 150)
                             for row in rows])
    condensed = pdist(vectors, metric="cosine")
    summary = {"ids": len(ids), "manifest_sha256": manifest["manifest_sha256"],
               "metric": "cosine distance", "methods": {}}
    for method in ("average", "complete", "weighted"):
        tree = linkage(condensed, method=method)
        summary["methods"][method] = {}
        for count in (12, 24):
            labels = fcluster(tree, count, criterion="maxclust")
            sizes = np.bincount(labels)[1:]
            summary["methods"][method][str(count)] = {
                "actual_clusters": len(sizes),
                "largest": int(sizes.max()),
                "median_size": float(np.median(sizes)),
                "clusters_below_five": int(np.sum(sizes < 5)),
                "resolution_ami": float(adjusted_mutual_info_score(resolution, labels)),
                "sizes": sorted(sizes.tolist(), reverse=True),
            }
            if method == "complete" and count == 12 and image_root is not None:
                _gallery(rows, vectors, labels, image_root, output.with_suffix(".png"))
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.manifest, args.clip, args.images, args.output)
    for method, cuts in result["methods"].items():
        for cut, values in cuts.items():
            print(method, cut, "largest", values["largest"],
                  "small clusters", values["clusters_below_five"],
                  "resolution AMI", round(values["resolution_ami"], 4))


if __name__ == "__main__":
    main()
