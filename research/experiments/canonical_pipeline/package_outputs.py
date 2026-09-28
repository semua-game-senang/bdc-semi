"""Package selected canonical run artifacts without raw images or credentials."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .common import sha256_file, write_json


RUN_FILES = (
    "clip.npy", "clip.json", "kec.npy", "kec.json", "concepts.json",
    "noun_discovery.json", "surfaces.npz", "surfaces.json",
    "C2_similarity.npy", "C2_similarity.json", "C2_changed_edges.json",
    "run.json", "full-c2-audit.json", "c2-stability-audit.json",
    "luna-exp012-audit.json", "luna-confirmation-audit.json",
    "full-edge-gallery.png", "linkage-audit.png",
)


def package(run_dir: Path, manifest: Path, confirmation: Path,
            regions: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {f"run/{name}": run_dir / name for name in RUN_FILES}
    files.update({"manifest.json": manifest,
                  "reviews/exp009_model_luna.csv": confirmation,
                  "reviews/exp012_model_luna.csv": regions})
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required selected outputs missing: {missing}")
    zip_path = output_dir / "canonical-selected-v1.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    result = {
        "bundle": zip_path.name,
        "sha256": sha256_file(zip_path),
        "size_bytes": zip_path.stat().st_size,
        "producing_git_commit": "f671643",
        "scope": "selected C0/C1/C2 development artifacts; no raw images",
        "review_provenance": "GPT-6 Luna model-assisted, owner-approved; not independent human ground truth",
        "review_pack_source": "semua-game-senang/bdc@c408a7ffde16af527fe6a69768e7469d17462da9:outputs/automated-exploratory-exp009-exp013-v1/",
        "files": {name: {"sha256": sha256_file(path),
                         "size_bytes": path.stat().st_size}
                  for name, path in sorted(files.items())},
    }
    write_json(output_dir / "bundle_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "confirmation", "regions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = package(args.run, args.manifest, args.confirmation,
                     args.regions, args.output)
    print(result["bundle"], result["size_bytes"], result["sha256"])


if __name__ == "__main__":
    main()
