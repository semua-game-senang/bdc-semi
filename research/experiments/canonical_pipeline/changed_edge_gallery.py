"""Show measured C2 corrections with the actual matched image regions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import read_json, readable_path
from .surface import load_surfaces


def _image_with_box(path: Path, box: np.ndarray, size: tuple[int, int]) -> Image.Image:
    with Image.open(readable_path(path)) as opened:
        image = opened.convert("RGB")
    original_width, original_height = image.size
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    draw.rectangle((x + x0 * image.width / original_width,
                    y + y0 * image.height / original_height,
                    x + x1 * image.width / original_width,
                    y + y1 * image.height / original_height),
                   outline="#ff3355", width=3)
    return canvas


def make_gallery(surface_path: Path, changes_path: Path,
                 image_root: Path, output: Path, *, seed: int = 20260928) -> dict:
    store = load_surfaces(surface_path)
    changes = read_json(changes_path)
    if not changes:
        raise ValueError("no changed C2 edges to visualize")
    ordered = sorted(changes, key=lambda item: item["delta"])
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(changes), size=min(4, len(changes)), replace=False)
    selected = ([('negative', item) for item in ordered[:4]] +
                [('positive', item) for item in ordered[-4:][::-1]] +
                [('random', changes[int(index)]) for index in random_indices])
    columns, cell_width, cell_height = 2, 444, 205
    sheet = Image.new("RGB", (columns * cell_width,
                              ((len(selected) + columns - 1) // columns) * cell_height),
                      "#edf1f4")
    draw = ImageDraw.Draw(sheet)
    image_size = (210, 155)
    for position, (selection, edge) in enumerate(selected):
        origin_x = position % columns * cell_width
        origin_y = position // columns * cell_height
        i, j = edge["i"], edge["j"]
        a, b = edge["regions"][0]
        if store.image_index[a] != i or store.image_index[b] != j:
            raise ValueError("matched region indices do not belong to edge images")
        for side, index, region in ((0, i, a), (1, j, b)):
            image = _image_with_box(image_root / store.ids[index],
                                    store.boxes[region], image_size)
            sheet.paste(image, (origin_x + 8 + side * 218, origin_y + 31))
        draw.text((origin_x + 8, origin_y + 6),
                  f"{selection}  delta={edge['delta']:+.4f}  "
                  f"score={edge['score']:.3f}  expected={edge['expected']:.3f}",
                  fill="#13202c")
        draw.text((origin_x + 8, origin_y + 188),
                  f"{store.ids[i][:22]}  |  {store.ids[j][:22]}", fill="#13202c")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {"edges": len(changes), "displayed": len(selected), "seed": seed,
            "selection": "4 lowest, 4 highest, 4 seeded random deltas"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--changes", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(make_gallery(args.surfaces, args.changes, args.images, args.output))


if __name__ == "__main__":
    main()
