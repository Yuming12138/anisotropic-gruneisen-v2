#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from gruneisen_v2_core import compute_thermal_response, summarize_effective_isotropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mesh convergence at low-frequency cutoffs"
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--mesh-array", action="append", required=True, metavar="MESH=NPZ")
    parser.add_argument("--cutoffs", type=float, nargs="+", required=True)
    return parser.parse_args()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "frequencies": data[
            "frequencies" if "frequencies" in data.files else "frequencies_0_thz"
        ],
        "gammas": data["gammas" if "gammas" in data.files else "gamma_voigt"],
        "weights": data["weights"],
    }


def main() -> None:
    args = parse_args()
    source_result = args.source_result.expanduser().resolve()
    report = json.loads((source_result / "preflight_report.json").read_text(encoding="utf-8"))
    parameters = report["parameters"]
    temperatures = np.arange(
        float(parameters["temperature_min_K"]),
        float(parameters["temperature_max_K"]) + 0.5 * float(parameters["temperature_step_K"]),
        float(parameters["temperature_step_K"]),
    )
    volume_A3 = float(read(source_result / "work" / "strain_0" / "POSCAR").get_volume())
    mesh_arrays = {}
    for value in args.mesh_array:
        mesh, separator, path = value.partition("=")
        if not separator:
            raise SystemExit(f"invalid --mesh-array value: {value}")
        mesh_arrays[int(mesh)] = load_arrays(Path(path).expanduser().resolve())

    rows = []
    for cutoff in args.cutoffs:
        for mesh in sorted(mesh_arrays):
            arrays = mesh_arrays[mesh]
            response, quality = compute_thermal_response(
                temperatures_K=temperatures,
                frequencies_THz=arrays["frequencies"],
                gammas=arrays["gammas"],
                weights=arrays["weights"],
                compliance_1_per_GPa=np.asarray(report["elastic"]["compliance_1_per_GPa"]),
                volume_A3=volume_A3,
                frequency_cutoff_THz=cutoff,
                axis_mapping=report["axis_mapping"],
            )
            screen = summarize_effective_isotropy(
                response,
                fani_threshold=float(parameters["fani_threshold"]),
                sign_tolerance_micro_per_K=float(parameters["sign_tolerance_micro_per_K"]),
            )
            index_300K = int(np.argmin(np.abs(response["temperatures_K"] - 300.0)))
            rows.append(
                {
                    "cutoff_THz": cutoff,
                    "mesh": mesh,
                    "alpha_directional_300K_micro_per_K": (
                        response["alpha_directional_per_K"][index_300K] * 1.0e6
                    ).tolist(),
                    "alpha_volume_300K_micro_per_K": float(
                        response["alpha_volume_per_K"][index_300K] * 1.0e6
                    ),
                    "F_ani_300K": float(response["F_ani"][index_300K]),
                    "max_excluded_heat_capacity_fraction": quality[
                        "max_excluded_heat_capacity_fraction"
                    ],
                    "status": screen["status"],
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
