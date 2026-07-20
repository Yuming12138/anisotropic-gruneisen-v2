#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build provenance for imported elastic results")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    created = 0
    skipped = 0
    for material_dir in sorted(results_root.iterdir()):
        task_status_path = material_dir / "task_status.json"
        validation_path = material_dir / "validation.json"
        elastic_dir = material_dir / "elastic"
        metadata_path = elastic_dir / "calculation_metadata.json"
        required = [elastic_dir / "POSCAR", elastic_dir / "CONTCAR", elastic_dir / "ELASTIC_TENSOR"]
        if metadata_path.is_file():
            skipped += 1
            continue
        if not task_status_path.is_file() or not validation_path.is_file() or not all(
            path.is_file() for path in required
        ):
            skipped += 1
            continue
        task_status = read_json(task_status_path)
        validation = read_json(validation_path)
        if task_status.get("status") != "success" or not validation.get("passed"):
            skipped += 1
            continue
        metadata = {
            "schema_version": 1,
            "provenance_type": "imported_local_campaign",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_results_root": str(results_root),
            "material_id": material_dir.name,
            "task_status": task_status,
            "validation": validation,
            "current_artifacts": {
                path.name: {
                    "path": str(path),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in required
            },
            "source_records": {
                "task_status": {
                    "path": str(task_status_path),
                    "sha256": sha256(task_status_path),
                },
                "validation": {
                    "path": str(validation_path),
                    "sha256": sha256(validation_path),
                },
            },
        }
        if not args.dry_run:
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        created += 1
    print(json.dumps({"created": created, "skipped": skipped, "dry_run": args.dry_run}))


if __name__ == "__main__":
    main()
