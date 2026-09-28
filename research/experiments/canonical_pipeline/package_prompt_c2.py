"""Package selected EXP043–EXP045 evidence without images or credentials."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .common import sha256_file, write_json


def package(run_dir: Path, text_dir: Path, scan: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    names = ("run.json", "full-audit.json", "C2_similarity.npy", "C2_similarity.json",
             "C2_labels.npy", "C2_labels.json", "candidate_edges.json",
             "excess_reference.json")
    files = {f"run/{name}": run_dir / name for name in names}
    files.update({"text/text_vectors.npy": text_dir / "text_vectors.npy",
                  "text/text_vectors.json": text_dir / "text_vectors.json",
                  "text/run.json": text_dir / "run.json",
                  "selection/prompt-weight-scan.json": scan})
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"selected evidence is incomplete: {missing}")
    target = output_dir / "prompt-selective-c2-v2.zip"
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    metadata = {
        "bundle": target.name, "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "producing_git_commits": ["7589b57", "d44c3a1"],
        "scope": "selected full-data prompt-control and reliability-gated C2 development evidence",
        "provenance": "EXP043 review reused for selection; GPT-6 Luna labels are model-assisted, not independent validation",
        "no_raw_images_or_review_csv": True,
        "files": {name: {"sha256": sha256_file(path),
                         "size_bytes": path.stat().st_size}
                  for name, path in sorted(files.items())},
    }
    write_json(output_dir / "bundle_manifest.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run_dir", "text_dir", "scan", "output_dir"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = package(args.run_dir, args.text_dir, args.scan, args.output_dir)
    print(result["bundle"], result["size_bytes"], result["sha256"])


if __name__ == "__main__":
    main()
