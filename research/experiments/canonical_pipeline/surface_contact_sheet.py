"""Draw highest/lowest scoring candidate regions for a development visual audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .common import readable_path
from .surface import load_surfaces


def make_sheet(surfaces: Path, image_root: Path, output: Path) -> None:
    store = load_surfaces(surfaces)
    width, height, columns = 240, 240, 5
    rows = (len(store.ids) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "#eaeef1")
    draw = ImageDraw.Draw(sheet)
    for i, image_id in enumerate(store.ids):
        with Image.open(readable_path(image_root / image_id)) as opened:
            image = opened.convert("RGB")
        original_width, original_height = image.size
        image.thumbnail((width - 16, height - 52))
        x = (i % columns) * width + (width - image.width) // 2
        y = (i // columns) * height + 8
        sheet.paste(image, (x, y))
        indices = store.indices(i, 0.0)
        for index, colour in ((indices[store.quality[indices].argmax()], "#15ef96"),
                              (indices[store.quality[indices].argmin()], "#ff4371")):
            x0, y0, x1, y1 = store.boxes[index]
            draw.rectangle((x + x0 * image.width / original_width,
                            y + y0 * image.height / original_height,
                            x + x1 * image.width / original_width,
                            y + y1 * image.height / original_height),
                           outline=colour, width=3)
        draw.text(((i % columns) * width + 8, (i // columns + 1) * height - 40),
                  f"{i:02d} {image_id[:23]}", fill="#101a22")
        low, high = store.quality[indices].min(), store.quality[indices].max()
        draw.text(((i % columns) * width + 8, (i // columns + 1) * height - 23),
                  f"low {low:.2f}  high {high:.2f}", fill="#101a22")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_sheet(args.surfaces, args.images, args.output)


if __name__ == "__main__":
    main()
