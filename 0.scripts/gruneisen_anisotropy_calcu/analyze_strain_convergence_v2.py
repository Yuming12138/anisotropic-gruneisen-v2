#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from gruneisen_v2_core import compute_thermal_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare h=0.005 and h=0.0025 responses")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fallback-root", type=Path, required=True)
    return parser.parse_args()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "frequencies": data["frequencies"],
        "gammas": data["gammas"],
        "weights": data["weights"],
    }


def response_from_arrays(arrays: dict[str, np.ndarray], report: dict, volume_A3: float):
    parameters = report["parameters"]
    temperatures = np.arange(
        float(parameters["temperature_min_K"]),
        float(parameters["temperature_max_K"]) + 0.5 * float(parameters["temperature_step_K"]),
        float(parameters["temperature_step_K"]),
    )
    return compute_thermal_response(
        temperatures_K=temperatures,
        frequencies_THz=arrays["frequencies"],
        gammas=arrays["gammas"],
        weights=arrays["weights"],
        compliance_1_per_GPa=np.asarray(report["elastic"]["compliance_1_per_GPa"]),
        volume_A3=volume_A3,
        frequency_cutoff_THz=float(parameters["frequency_cutoff_THz"]),
        axis_mapping=report["axis_mapping"],
    )


def main() -> None:
    args = parse_args()
    base = args.base.expanduser().resolve()
    fallback_root = args.fallback_root.expanduser().resolve()
    primary_arrays = {
        "0003": base / "bench_agv2_meshconv_20260717" / "mesh_20" / "mesh_arrays.npz",
        "0236": base
        / "bench_agv2_multimat_20260717"
        / "mesh_0236"
        / "mesh_20"
        / "mesh_arrays.npz",
        "0171": base
        / "bench_agv2_multimat_20260717"
        / "mesh_0171"
        / "mesh_20"
        / "mesh_arrays.npz",
        "0091": base
        / "bench_agv2_multimat_20260717"
        / "mesh_0091"
        / "mesh_20"
        / "mesh_arrays.npz",
    }
    rows = []
    for material, primary_path in primary_arrays.items():
        fallback_result = (
            base
            / "run_20260717_batch1024_all10"
            / "results"
            / material
            / "agv2_br_h0025_m6"
        )
        report = json.loads((fallback_result / "preflight_report.json").read_text())
        volume_A3 = float(read(fallback_result / "work" / "strain_0" / "POSCAR").get_volume())
        primary_response, primary_quality = response_from_arrays(
            load_arrays(primary_path), report, volume_A3
        )
        fallback_response, fallback_quality = response_from_arrays(
            load_arrays(fallback_root / f"mesh_{material}" / "mesh_20" / "mesh_arrays.npz"),
            report,
            volume_A3,
        )
        index_100K = int(np.argmin(np.abs(primary_response["temperatures_K"] - 100.0)))
        index_300K = int(np.argmin(np.abs(primary_response["temperatures_K"] - 300.0)))
        primary_integrals = primary_response["gruneisen_integrals_J_per_K"][index_100K]
        fallback_integrals = fallback_response["gruneisen_integrals_J_per_K"][index_100K]
        integral_scale = np.maximum(
            np.maximum(np.abs(primary_integrals), np.abs(fallback_integrals)),
            1.0e-25,
        )
        integral_relative = np.abs(primary_integrals - fallback_integrals) / integral_scale
        primary_alpha_volume = primary_response["alpha_volume_per_K"] * 1.0e6
        fallback_alpha_volume = fallback_response["alpha_volume_per_K"] * 1.0e6
        primary_directional = primary_response["alpha_directional_per_K"] * 1.0e6
        fallback_directional = fallback_response["alpha_directional_per_K"] * 1.0e6
        rows.append(
            {
                "material": material,
                "integrals_100K_J_per_K": {
                    "primary": primary_integrals.tolist(),
                    "fallback": fallback_integrals.tolist(),
                    "relative_difference": integral_relative.tolist(),
                    "max_relative_difference": float(np.max(integral_relative)),
                },
                "alpha_volume_micro_per_K": {
                    "primary_100K": float(primary_alpha_volume[index_100K]),
                    "fallback_100K": float(fallback_alpha_volume[index_100K]),
                    "difference_100K": float(
                        primary_alpha_volume[index_100K] - fallback_alpha_volume[index_100K]
                    ),
                    "relative_difference_100K": float(
                        abs(primary_alpha_volume[index_100K] - fallback_alpha_volume[index_100K])
                        / max(abs(fallback_alpha_volume[index_100K]), 1.0e-12)
                    ),
                    "primary_300K": float(primary_alpha_volume[index_300K]),
                    "fallback_300K": float(fallback_alpha_volume[index_300K]),
                    "difference_300K": float(
                        primary_alpha_volume[index_300K] - fallback_alpha_volume[index_300K]
                    ),
                    "max_difference_all_T": float(
                        np.max(np.abs(primary_alpha_volume - fallback_alpha_volume))
                    ),
                },
                "max_directional_difference_all_T_micro_per_K": float(
                    np.max(np.abs(primary_directional - fallback_directional))
                ),
                "max_abs_gamma": {
                    "primary": max(
                        item["abs_max"] for item in primary_quality["gamma_statistics"]
                    ),
                    "fallback": max(
                        item["abs_max"] for item in fallback_quality["gamma_statistics"]
                    ),
                },
                "max_excluded_heat_capacity_fraction": {
                    "primary": primary_quality["max_excluded_heat_capacity_fraction"],
                    "fallback": fallback_quality["max_excluded_heat_capacity_fraction"],
                },
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
