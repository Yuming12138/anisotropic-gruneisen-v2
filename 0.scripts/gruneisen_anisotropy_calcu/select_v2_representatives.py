#!/usr/bin/env python
"""Select one low-cost NTE and PTE representative per crystal system for v2 validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CLASSIFICATION = PROJECT_ROOT / "isotropy_classification" / "layer1_all_materials.csv"
CRYSTAL_SYSTEM_ORDER = (
    "cubic",
    "hexagonal",
    "trigonal",
    "tetragonal",
    "orthorhombic",
    "monoclinic",
    "triclinic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--result-subdir", default="gruneisen_aniso_1M_v2")
    parser.add_argument("--out-csv", type=Path, default=SCRIPT_DIR / "v2_representative_materials.csv")
    parser.add_argument("--out-list", type=Path, default=SCRIPT_DIR / "v2_representative_materials.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.classification.open(newline="", encoding="utf-8-sig") as handle:
        classification = list(csv.DictReader(handle))

    candidates = []
    for row in classification:
        material_dir = Path(row["material_dir"])
        report_path = material_dir / args.result_subdir / "preflight_report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "failed":
            continue
        if report.get("axis_mapping", {}).get("status") != "ok":
            continue
        supercell = np.asarray(report["supercell_matrix"], dtype=int)
        supercell_atoms = int(report["elastic_atom_count"] * round(abs(np.linalg.det(supercell))))
        candidates.append(
            {
                "dataset_class": row["dataset_class"],
                "material": row["material"],
                "crystal_system": row["root_crystal_system_primary"],
                "spacegroup_number": row["root_spacegroup_number_primary"],
                "spacegroup_symbol": row["root_spacegroup_symbol_primary"],
                "material_dir": str(material_dir),
                "supercell_matrix": json.dumps(report["supercell_matrix"], separators=(",", ":")),
                "supercell_atom_count": supercell_atoms,
                "elastic_min_eigenvalue_GPa": report["elastic"]["min_eigenvalue_GPa"],
                "elastic_condition_number": report["elastic"]["condition_number"],
                "selection_reason": "lowest estimated v2 supercell atom count in class/system",
            }
        )

    selected = []
    for crystal_system in CRYSTAL_SYSTEM_ORDER:
        for dataset_class in ("NTE", "PTE"):
            group = [
                row
                for row in candidates
                if row["crystal_system"] == crystal_system and row["dataset_class"] == dataset_class
            ]
            if not group:
                continue
            selected.append(
                min(
                    group,
                    key=lambda row: (
                        row["supercell_atom_count"],
                        -float(row["elastic_min_eigenvalue_GPa"]),
                        row["material"],
                    ),
                )
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    args.out_list.write_text(
        "\n".join(row["material"] for row in selected) + "\n",
        encoding="utf-8",
    )
    print(f"selected={len(selected)}")
    print(f"csv={args.out_csv.resolve()}")
    print(f"list={args.out_list.resolve()}")
    for row in selected:
        print(
            row["dataset_class"],
            row["crystal_system"],
            row["material"],
            f"supercell_atoms={row['supercell_atom_count']}",
        )


if __name__ == "__main__":
    main()
