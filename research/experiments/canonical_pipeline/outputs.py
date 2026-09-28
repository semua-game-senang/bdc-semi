"""Auditable galleries, evidence cards, and paper figures from actual images."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import write_json
from .surface import SurfaceStore


def _thumbnail(path: str, size: tuple[int, int] = (160, 120)) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def contact_sheet(rows: list[dict], indices: list[int], output: Path, *,
                  title: str, columns: int = 5, blinded: bool = False) -> None:
    cell_width, cell_height = 180, 150
    lines = (len(indices) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, 34 + max(lines, 1) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for position, index in enumerate(indices):
        x, y = (position % columns) * cell_width, 34 + (position // columns) * cell_height
        sheet.paste(_thumbnail(rows[index]["path"]), (x + 10, y))
        caption = f"item {position + 1}" if blinded else rows[index]["id"][:25]
        draw.text((x + 10, y + 123), caption, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def evidence_cards(rows: list[dict], distance: np.ndarray, broad: np.ndarray,
                   fine: np.ndarray, output_dir: Path, *,
                   surface: SurfaceStore | None = None, gate: float = 0.45,
                   per_card: int = 8) -> list[dict]:
    """Visible exemplars and counterexamples; no automatic material labels."""
    if len(rows) != len(distance) or len(rows) != len(broad):
        raise ValueError("evidence-card rows are not aligned")
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for group in sorted(set(map(int, broad))):
        members = np.flatnonzero(broad == group)
        medoid = int(members[np.argmin(distance[np.ix_(members, members)].mean(axis=1))])
        order = sorted(map(int, members), key=lambda index: (distance[medoid, index], rows[index]["id"]))
        outsiders = np.flatnonzero(broad != group)
        counter = int(outsiders[np.argmin(distance[medoid, outsiders])]) if len(outsiders) else None
        chosen = order[:per_card]
        if counter is not None:
            chosen.append(counter)
        contact_sheet(rows, chosen, output_dir / f"group_{group:03d}.png",
                      title=f"Group {group}: {len(members)} unique images; last item is nearest counterexample")
        strata = Counter("150x150" if rows[index]["width"] == rows[index]["height"] == 150 else "other"
                         for index in members)
        coverage = (float(np.mean([len(surface.indices(int(index), gate)) > 0 for index in members]))
                    if surface is not None else None)
        cards.append({"group": group, "unique_images": len(members),
                      "medoid_id": rows[medoid]["id"],
                      "representative_ids": [rows[index]["id"] for index in order[:per_card]],
                      "counterexample_id": rows[counter]["id"] if counter is not None else None,
                      "fine_groups": dict(Counter(map(int, fine[members]))),
                      "source_strata": dict(strata), "surface_coverage": coverage,
                      "interpretation": "Requires blinded human description; visible evidence only"})
    write_json(output_dir / "cards.json", cards)
    return cards


def hierarchy_plot(linkage_matrix: np.ndarray, output: Path, *,
                   title: str = "Average-linkage hierarchy", truncate: int = 40) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=truncate,
               no_labels=True, ax=ax, color_threshold=None)
    ax.set(title=title, xlabel="Merged groups", ylabel="Linkage distance")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
