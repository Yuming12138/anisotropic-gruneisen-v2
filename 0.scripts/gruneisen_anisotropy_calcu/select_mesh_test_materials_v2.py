#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from gruneisen_v2_core import (
    choose_supercell_matrix,
    read_elastic_tensor,
    structure_axis_mapping,
    validate_elastic_tensor,
)
from run_gruneisen_thermal_expansion_v2 import make_phonon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank valid materials for mesh convergence tests")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for material_dir in sorted(args.results_root.expanduser().resolve().iterdir()):
        try:
            root_poscar = material_dir / "POSCAR"
            elastic_poscar = material_dir / "elastic" / "POSCAR"
            elastic_tensor = material_dir / "elastic" / "ELASTIC_TENSOR"
            if not all(path.is_file() for path in (root_poscar, elastic_poscar, elastic_tensor)):
                continue
            _, elastic = validate_elastic_tensor(read_elastic_tensor(elastic_tensor))
            if not elastic["positive_definite"]:
                continue
            root_structure = Structure.from_file(root_poscar)
            elastic_structure = Structure.from_file(elastic_poscar)
            if structure_axis_mapping(root_structure, elastic_structure).get("status") != "ok":
                continue
            supercell = choose_supercell_matrix(
                elastic_structure.lattice.matrix,
                minimum_length_A=12.0,
            )
            atoms = read(elastic_poscar)
            phonon = make_phonon(atoms, supercell)
            phonon.generate_displacements(distance=0.01)
            displacement_count = len(phonon.supercells_with_displacements or [])
            unit_atoms = len(atoms)
            supercell_atoms = unit_atoms * round(abs(np.linalg.det(supercell)))
            crystal_system = SpacegroupAnalyzer(
                elastic_structure,
                symprec=1.0e-3,
            ).get_crystal_system()
            rows.append(
                {
                    "material": material_dir.name,
                    "unit_atoms": unit_atoms,
                    "supercell_atoms": supercell_atoms,
                    "displacement_count": displacement_count,
                    "force_cost_proxy": supercell_atoms * displacement_count,
                    "crystal_system": crystal_system,
                    "elastic_min_eigenvalue_GPa": elastic["min_eigenvalue_GPa"],
                    "supercell": " ".join(str(value) for value in np.diag(supercell)),
                }
            )
        except Exception:
            continue
    rows.sort(key=lambda row: (row["force_cost_proxy"], row["supercell_atoms"]))
    if not rows:
        raise RuntimeError("no_valid_materials_found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"valid={len(rows)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
