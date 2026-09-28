"""Compare the canonical electronic input manifest with the official-data inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_manifest


def verify(official_path: Path, canonical_path: Path) -> dict:
    official = json.loads(official_path.read_text(encoding="utf-8"))
    canonical = load_manifest(canonical_path)
    expected = {
        Path(row["path"]).name: row["sha256"].lower()
        for row in official["train"] if row["label"] == 1
    }
    actual = {row["id"]: row["sha256"].lower() for row in canonical["rows"]}
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    mismatched = sorted(name for name in expected.keys() & actual.keys()
                        if expected[name] != actual[name])
    report = {"official_rows": len(expected), "canonical_rows": len(actual),
              "canonical_unique": canonical["unique_count"],
              "missing": len(missing), "extra": len(extra),
              "hash_mismatches": len(mismatched),
              "manifest_sha256": canonical["manifest_sha256"]}
    if missing or extra or mismatched:
        raise ValueError(f"input inventory differs: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.official, args.canonical), indent=2))


if __name__ == "__main__":
    main()
