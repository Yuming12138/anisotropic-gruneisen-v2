#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read
from phonopy.file_IO import parse_FORCE_CONSTANTS

from gruneisen_v2_core import (
    compute_thermal_response,
    runtime_versions,
    rows_to_text_table,
    sha256_file,
    stable_json_hash,
    summarize_effective_isotropy,
    write_json,
)
from run_gruneisen_thermal_expansion_v2 import DiagnosticGruneisenMesh, make_phonon


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
CORE_PATH = SCRIPT_DIR / "gruneisen_v2_core.py"
RUNNER_PATH = SCRIPT_DIR / "run_gruneisen_thermal_expansion_v2.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute anisotropic response from saved force constants"
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--mesh", type=int, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_phonon(state_dir: Path, supercell_matrix: np.ndarray):
    phonon = make_phonon(read(state_dir / "POSCAR"), supercell_matrix)
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(state_dir / "FORCE_CONSTANTS"))
    return phonon


def main() -> None:
    args = parse_args()
    source_result = args.source_result.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((source_result / "preflight_report.json").read_text(encoding="utf-8"))
    parameters = report["parameters"]
    supercell_matrix = np.asarray(report["supercell_matrix"], dtype=int)
    work_dir = source_result / "work"
    phonon_zero = load_phonon(work_dir / "strain_0", supercell_matrix)
    strained_phonons = {
        (component, sign): load_phonon(
            work_dir / f"eta{component}_{'minus' if sign < 0 else 'plus'}",
            supercell_matrix,
        )
        for component in range(1, 7)
        for sign in (-1, 1)
    }

    start = time.perf_counter()
    gammas = []
    qpoints_ref = None
    weights_ref = None
    frequencies_ref = None
    for component in range(1, 7):
        mesh = DiagnosticGruneisenMesh(
            phonon_zero.dynamical_matrix,
            strained_phonons[(component, 1)].dynamical_matrix,
            strained_phonons[(component, -1)].dynamical_matrix,
            mesh=tuple(args.mesh),
            delta_strain=2.0 * float(parameters["strain"]),
            is_time_reversal=True,
            is_gamma_center=True,
            is_mesh_symmetry=False,
            imaginary_cutoff_THz=float(parameters["frequency_cutoff_THz"]),
        )
        qpoints = np.asarray(mesh.get_qpoints(), dtype=float)
        weights = np.asarray(mesh.get_weights(), dtype=float)
        frequencies = np.asarray(mesh.get_frequencies(), dtype=float)
        if qpoints_ref is None:
            qpoints_ref = qpoints
            weights_ref = weights
            frequencies_ref = frequencies
        gammas.append(np.asarray(mesh.get_gruneisen(), dtype=float))
    mesh_seconds = time.perf_counter() - start

    temperatures = np.arange(
        float(parameters["temperature_min_K"]),
        float(parameters["temperature_max_K"]) + 0.5 * float(parameters["temperature_step_K"]),
        float(parameters["temperature_step_K"]),
    )
    gamma_array = np.asarray(gammas)
    response, quality = compute_thermal_response(
        temperatures_K=temperatures,
        frequencies_THz=frequencies_ref,
        gammas=gamma_array,
        weights=weights_ref,
        compliance_1_per_GPa=np.asarray(report["elastic"]["compliance_1_per_GPa"]),
        volume_A3=float(phonon_zero.primitive.volume),
        frequency_cutoff_THz=float(parameters["frequency_cutoff_THz"]),
        axis_mapping=report["axis_mapping"],
    )
    screen = summarize_effective_isotropy(
        response,
        fani_threshold=float(parameters["fani_threshold"]),
        sign_tolerance_micro_per_K=float(parameters["sign_tolerance_micro_per_K"]),
    )
    np.savez_compressed(
        output_dir / "mesh_arrays.npz",
        qpoints=qpoints_ref,
        weights=weights_ref,
        frequencies=frequencies_ref,
        gammas=gamma_array,
        temperatures=response["temperatures_K"],
        alpha_voigt=response["alpha_voigt_per_K"],
        alpha_volume=response["alpha_volume_per_K"],
        alpha_directional=response["alpha_directional_per_K"],
        F_ani=response["F_ani"],
        excluded_heat_capacity_fraction=response["excluded_heat_capacity_fraction"],
    )
    cartesian_rows = np.column_stack(
        [
            response["temperatures_K"],
            response["alpha_voigt_per_K"] * 1.0e6,
            response["alpha_volume_per_K"] * 1.0e6,
        ]
    )
    (output_dir / "thermal_expansion_cartesian.dat").write_text(
        rows_to_text_table(
            [
                "T_K",
                "alpha_xx",
                "alpha_yy",
                "alpha_zz",
                "alpha_yz_eng",
                "alpha_xz_eng",
                "alpha_xy_eng",
                "alpha_volume",
            ],
            cartesian_rows,
        ),
        encoding="utf-8",
    )
    directional_rows = np.column_stack(
        [
            response["temperatures_K"],
            response["alpha_directional_per_K"] * 1.0e6,
            response["alpha_volume_per_K"] * 1.0e6,
            response["F_ani"],
        ]
    )
    (output_dir / "thermal_expansion_directional.dat").write_text(
        rows_to_text_table(
            ["T_K", "alpha_a", "alpha_b", "alpha_c", "alpha_volume", "F_ani"],
            directional_rows,
        ),
        encoding="utf-8",
    )
    integral_rows = np.column_stack(
        [response["temperatures_K"], response["gruneisen_integrals_J_per_K"]]
    )
    (output_dir / "gruneisen_integrals.dat").write_text(
        rows_to_text_table(
            ["T_K", "I_xx", "I_yy", "I_zz", "I_yz", "I_xz", "I_xy"],
            integral_rows,
        ),
        encoding="utf-8",
    )
    fani_rows = np.column_stack(
        [
            response["temperatures_K"],
            response["alpha_volume_hyd_per_K"] * 1.0e6,
            response["alpha_volume_dev_per_K"] * 1.0e6,
            response["alpha_volume_per_K"] * 1.0e6,
            response["F_ani"],
            response["excluded_heat_capacity_fraction"],
        ]
    )
    (output_dir / "fani_temperature.dat").write_text(
        rows_to_text_table(
            [
                "T_K",
                "alphaV_hyd",
                "alphaV_dev",
                "alphaV_total",
                "F_ani",
                "excluded_Cv_fraction",
            ],
            fani_rows,
        ),
        encoding="utf-8",
    )
    index_300K = int(np.argmin(np.abs(response["temperatures_K"] - 300.0)))
    source_complete = json.loads(
        (source_result / "calculation_complete.json").read_text(encoding="utf-8")
    )
    fingerprint = {
        "source_result": str(source_result),
        "source_fingerprint_sha256": source_complete["fingerprint_sha256"],
        "mesh": list(args.mesh),
        "files": {
            "script": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
            "core": {"path": str(CORE_PATH), "sha256": sha256_file(CORE_PATH)},
            "runner": {"path": str(RUNNER_PATH), "sha256": sha256_file(RUNNER_PATH)},
        },
        "runtime_versions": runtime_versions(),
    }
    fingerprint["fingerprint_sha256"] = stable_json_hash(fingerprint)
    summary = {
        "schema_version": 1,
        "source_result": str(source_result),
        "mesh": list(args.mesh),
        "qpoint_count": int(len(qpoints_ref)),
        "mesh_seconds": mesh_seconds,
        "quality": quality,
        "effective_isotropy_screen": screen,
        "fingerprint": fingerprint,
        "at_300K": {
            "alpha_directional_micro_per_K": (
                response["alpha_directional_per_K"][index_300K] * 1.0e6
            ).tolist(),
            "alpha_volume_micro_per_K": float(
                response["alpha_volume_per_K"][index_300K] * 1.0e6
            ),
            "F_ani": float(response["F_ani"][index_300K]),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_json(
        output_dir / "calculation_complete.json",
        {
            "status": "complete",
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "summary": str(output_dir / "summary.json"),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
