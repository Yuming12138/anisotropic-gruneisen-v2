#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 20^3 and 24^3 across materials")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--systems-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base.expanduser().resolve()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    systems_root = args.systems_root.expanduser().resolve()
    paths = {
        "0003": base / "bench_agv2_meshconv_20260717",
        "0236": benchmark_root / "mesh_0236",
        "0171": benchmark_root / "mesh_0171",
        "0091": benchmark_root / "mesh_0091",
        "0157": systems_root / "mesh_0157",
        "0292": systems_root / "mesh_0292",
        "0223": systems_root / "mesh_0223",
    }
    metadata = {
        "0003": ("orthorhombic", 28, 336, 25),
        "0236": ("cubic", 2, 54, 2),
        "0171": ("monoclinic", 7, 84, 14),
        "0091": ("trigonal", 42, 336, 21),
        "0157": ("hexagonal", 4, 192, 4),
        "0292": ("tetragonal", 7, 56, 5),
        "0223": ("triclinic", 9, 162, 54),
    }
    rows = []
    for material, root in paths.items():
        summaries = {
            mesh: json.loads((root / f"mesh_{mesh}" / "summary.json").read_text())
            for mesh in (20, 24)
        }
        tables = {
            mesh: np.loadtxt(root / f"mesh_{mesh}" / "thermal_expansion_directional.dat")
            for mesh in (20, 24)
        }
        index_300K = int(np.argmin(np.abs(tables[20][:, 0] - 300.0)))
        alpha_volume_20 = float(tables[20][index_300K, 4])
        alpha_volume_24 = float(tables[24][index_300K, 4])
        crystal_system, unit_atoms, supercell_atoms, displacement_count = metadata[material]
        rows.append(
            {
                "material": material,
                "formula": Structure.from_file(
                    base / "run_20260717_batch1024_all10" / "results" / material / "POSCAR"
                ).composition.reduced_formula,
                "crystal_system": crystal_system,
                "unit_atoms": unit_atoms,
                "supercell_atoms": supercell_atoms,
                "displacement_count": displacement_count,
                "mesh_seconds": {
                    str(mesh): summaries[mesh]["mesh_seconds"] for mesh in (20, 24)
                },
                "alpha_volume_300K_micro_per_K": {
                    "20": alpha_volume_20,
                    "24": alpha_volume_24,
                },
                "alpha_volume_300K_difference_micro_per_K": alpha_volume_20
                - alpha_volume_24,
                "alpha_volume_300K_relative_difference_percent": abs(
                    alpha_volume_20 - alpha_volume_24
                )
                / max(abs(alpha_volume_24), 1.0e-12)
                * 100.0,
                "max_alpha_volume_difference_all_T_micro_per_K": float(
                    np.max(np.abs(tables[20][:, 4] - tables[24][:, 4]))
                ),
                "max_directional_difference_all_T_micro_per_K": float(
                    np.max(np.abs(tables[20][:, 1:4] - tables[24][:, 1:4]))
                ),
                "F_ani_300K": {
                    str(mesh): float(tables[mesh][index_300K, 5]) for mesh in (20, 24)
                },
                "status": {
                    str(mesh): summaries[mesh]["effective_isotropy_screen"]["status"]
                    for mesh in (20, 24)
                },
                "max_abs_gamma": {
                    str(mesh): max(
                        item["abs_max"] for item in summaries[mesh]["quality"]["gamma_statistics"]
                    )
                    for mesh in (20, 24)
                },
                "max_excluded_heat_capacity_fraction": {
                    str(mesh): summaries[mesh]["quality"][
                        "max_excluded_heat_capacity_fraction"
                    ]
                    for mesh in (20, 24)
                },
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
