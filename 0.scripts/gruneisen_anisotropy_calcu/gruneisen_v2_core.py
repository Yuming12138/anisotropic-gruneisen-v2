#!/usr/bin/env python
"""Pure scientific and data-contract utilities for anisotropic Gruneisen v2.

This module deliberately does not import MatterSim.  Preflight, tensor checks,
thermal integration, and result post-processing therefore remain usable from a
lightweight Python environment.  MatterSim is imported lazily by the v2 runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ase import Atoms
from pymatgen.core import Structure


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
HYDROSTATIC_VOIGT = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

KB_J_PER_K = 1.380649e-23
H_J_S = 6.62607015e-34
ANGSTROM3_TO_M3 = 1.0e-30
GPA_TO_PA = 1.0e9


@dataclass(frozen=True)
class V2Parameters:
    schema_version: int = 1
    model_size: str = "1M"
    dtype: str = "float64"
    strain: float = 0.005
    fallback_strain: float = 0.0025
    displacement_A: float = 0.01
    mesh: tuple[int, int, int] = (30, 30, 30)
    min_supercell_length_A: float = 12.0
    internal_relax_fmax_eV_A: float = 1.0e-3
    internal_relax_max_steps: int = 1000
    frequency_cutoff_THz: float = 1.0e-4
    temperature_min_K: float = 10.0
    temperature_max_K: float = 1000.0
    temperature_step_K: float = 10.0
    fani_threshold: float = 0.20
    sign_tolerance_micro_per_K: float = 1.0e-3

    def temperatures(self) -> np.ndarray:
        return np.arange(
            self.temperature_min_K,
            self.temperature_max_K + 0.5 * self.temperature_step_K,
            self.temperature_step_K,
            dtype=float,
        )


@lru_cache(maxsize=4096)
def _sha256_file_cached(path_text: str, size: int, mtime_ns: int, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _sha256_file_cached(str(resolved), stat.st_size, stat.st_mtime_ns, chunk_size)


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for module_name in ("scipy", "ase", "phonopy", "spglib", "pymatgen", "torch", "mattersim"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "not_importable"
    return versions


def read_elastic_tensor(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(values) >= 6:
            rows.append(values[-6:])
        if len(rows) == 6:
            break
    if len(rows) != 6:
        raise ValueError(f"elastic_parse_error:{path}")
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"elastic_parse_error:{path}")
    return matrix


def validate_elastic_tensor(
    stiffness_GPa_raw: np.ndarray,
    symmetry_tolerance_relative: float = 1.0e-6,
    condition_limit: float = 1.0e5,
) -> tuple[np.ndarray, dict[str, Any]]:
    C_raw = np.asarray(stiffness_GPa_raw, dtype=float)
    if C_raw.shape != (6, 6) or not np.all(np.isfinite(C_raw)):
        raise ValueError("elastic_parse_error")
    norm = float(np.linalg.norm(C_raw))
    asymmetry = float(np.linalg.norm(C_raw - C_raw.T) / norm) if norm else math.inf
    if asymmetry > symmetry_tolerance_relative:
        raise ValueError(f"elastic_tensor_asymmetry_too_large:{asymmetry:.6g}")
    C = 0.5 * (C_raw + C_raw.T)
    eigenvalues = np.linalg.eigvalsh(C)
    min_eigenvalue = float(eigenvalues[0])
    condition_number = float(np.linalg.cond(C))
    positive_definite = bool(min_eigenvalue > 0.0)
    ill_conditioned = bool(condition_number > condition_limit)
    compliance = np.linalg.inv(C) if positive_definite else None
    report = {
        "unit": "GPa",
        "voigt_order": list(VOIGT_LABELS),
        "asymmetry_relative": asymmetry,
        "symmetrized": bool(asymmetry > 0.0),
        "eigenvalues_GPa": [float(value) for value in eigenvalues],
        "min_eigenvalue_GPa": min_eigenvalue,
        "positive_definite": positive_definite,
        "condition_number": condition_number,
        "ill_conditioned": ill_conditioned,
        "condition_limit": condition_limit,
    }
    if compliance is not None:
        report["compliance_1_per_GPa"] = compliance.tolist()
    return C, report


def engineering_strain_tensor(component: int, amplitude: float) -> np.ndarray:
    """Return the Cartesian symmetric strain tensor for Voigt component 1..6."""

    if component < 1 or component > 6:
        raise ValueError("strain component must be in 1..6")
    eta = np.zeros(6, dtype=float)
    eta[component - 1] = float(amplitude)
    return np.asarray(
        [
            [eta[0], eta[5] / 2.0, eta[4] / 2.0],
            [eta[5] / 2.0, eta[1], eta[3] / 2.0],
            [eta[4] / 2.0, eta[3] / 2.0, eta[2]],
        ],
        dtype=float,
    )


def apply_engineering_strain(atoms: Atoms, component: int, amplitude: float) -> Atoms:
    """Apply a Cartesian engineering strain to an ASE row-vector cell."""

    strained = atoms.copy()
    E = engineering_strain_tensor(component, amplitude)
    F = np.eye(3) + E
    new_cell = np.asarray(atoms.cell.array, dtype=float) @ F.T
    strained.set_cell(new_cell, scale_atoms=True)
    return strained


def choose_supercell_matrix(
    cell: np.ndarray,
    minimum_length_A: float = 12.0,
    minimum_repeat: int = 2,
) -> np.ndarray:
    lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
    if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise ValueError("invalid_reference_cell")
    repeats = [
        max(int(minimum_repeat), int(math.ceil(float(minimum_length_A) / float(length))))
        for length in lengths
    ]
    return np.diag(repeats).astype(int)


def structure_axis_mapping(
    source_structure: Structure,
    target_structure: Structure,
    ltol: float = 0.25,
    atol: float = 5.0,
) -> dict[str, Any]:
    """Map source crystallographic a/b/c directions into target Cartesian frame."""

    mapping = target_structure.lattice.find_mapping(
        source_structure.lattice,
        ltol=ltol,
        atol=atol,
    )
    if mapping is None:
        return {
            "status": "cte_axis_to_elastic_lattice_mapping_failed",
            "axis_unit_vectors_in_elastic_frame": None,
        }
    _aligned_lattice, rotation_matrix, scale_matrix = mapping
    if rotation_matrix is None:
        return {
            "status": "cte_axis_to_elastic_rotation_missing",
            "axis_unit_vectors_in_elastic_frame": None,
        }
    rotation = np.asarray(rotation_matrix, dtype=float)
    source_cell = np.asarray(source_structure.lattice.matrix, dtype=float)
    vectors = {}
    for index, axis in enumerate(("a", "b", "c")):
        vector = np.inner(source_cell[index], rotation)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            return {
                "status": "cte_axis_to_elastic_zero_vector",
                "axis_unit_vectors_in_elastic_frame": None,
            }
        vectors[axis] = (vector / norm).tolist()
    return {
        "status": "ok",
        "rotation_matrix": rotation.tolist(),
        "scale_matrix": np.asarray(scale_matrix).tolist(),
        "axis_unit_vectors_in_elastic_frame": vectors,
    }


def mode_heat_capacity_J_K(temperature_K: float, frequencies_THz: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies_THz, dtype=float)
    energy = H_J_S * frequencies * 1.0e12
    valid = energy > 0.0
    x = np.zeros_like(energy)
    x[valid] = energy[valid] / (KB_J_PER_K * float(temperature_K))
    cv = np.zeros_like(energy)
    small = valid & (x < 1.0e-5)
    regular = valid & (x >= 1.0e-5) & (x < 100.0)
    cv[small] = KB_J_PER_K * (1.0 - x[small] ** 2 / 12.0)
    denominator = np.expm1(x[regular])
    cv[regular] = (
        KB_J_PER_K
        * x[regular] ** 2
        * (denominator + 1.0)
        / denominator**2
    )
    return cv


def strain_voigt_to_tensor(engineering_voigt: np.ndarray) -> np.ndarray:
    value = np.asarray(engineering_voigt, dtype=float)
    if value.shape != (6,):
        raise ValueError("Expected six engineering strain components")
    return np.asarray(
        [
            [value[0], value[5] / 2.0, value[4] / 2.0],
            [value[5] / 2.0, value[1], value[3] / 2.0],
            [value[4] / 2.0, value[3] / 2.0, value[2]],
        ]
    )


def gamma_statistics(gamma: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(gamma, dtype=float)[np.asarray(valid_mask, dtype=bool)]
    if values.size == 0:
        return {"count": 0}
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "negative_count": int(np.sum(values < 0.0)),
        "positive_count": int(np.sum(values > 0.0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "abs_median": float(np.median(absolute)),
        "abs_p90": float(np.quantile(absolute, 0.90)),
        "abs_p95": float(np.quantile(absolute, 0.95)),
        "abs_p99": float(np.quantile(absolute, 0.99)),
        "abs_max": float(np.max(absolute)),
    }


def compute_thermal_response(
    temperatures_K: np.ndarray,
    frequencies_THz: np.ndarray,
    gammas: np.ndarray,
    weights: np.ndarray,
    compliance_1_per_GPa: np.ndarray,
    volume_A3: float,
    frequency_cutoff_THz: float,
    axis_mapping: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Integrate six strain Gruneisen components and calculate alpha and F_ani."""

    frequencies = np.asarray(frequencies_THz, dtype=float)
    gamma = np.asarray(gammas, dtype=float)
    qweights = np.asarray(weights, dtype=float)
    if gamma.shape[0] != 6 or gamma.shape[1:] != frequencies.shape:
        raise ValueError("gamma_shape_mismatch")
    if qweights.shape != (frequencies.shape[0],):
        raise ValueError("weight_shape_mismatch")
    if np.asarray(compliance_1_per_GPa).shape != (6, 6):
        raise ValueError("compliance_shape_mismatch")

    weight_sum = float(np.sum(qweights))
    if weight_sum <= 0.0:
        raise ValueError("invalid_qpoint_weights")
    compliance_Pa = np.asarray(compliance_1_per_GPa, dtype=float) / GPA_TO_PA
    volume_m3 = float(volume_A3) * ANGSTROM3_TO_M3
    base_valid = np.isfinite(frequencies) & (frequencies > float(frequency_cutoff_THz))
    gamma_finite = np.all(np.isfinite(gamma), axis=0)
    valid = base_valid & gamma_finite

    axis_vectors = None
    if axis_mapping and axis_mapping.get("status") == "ok":
        axis_vectors = axis_mapping.get("axis_unit_vectors_in_elastic_frame")

    ntemp = len(temperatures_K)
    integrals = np.full((ntemp, 6), np.nan)
    macro_gamma = np.full((ntemp, 6), np.nan)
    alpha_voigt = np.full((ntemp, 6), np.nan)
    alpha_volume = np.full(ntemp, np.nan)
    alpha_volume_hyd = np.full(ntemp, np.nan)
    alpha_volume_dev = np.full(ntemp, np.nan)
    F_ani = np.full(ntemp, np.nan)
    alpha_directional = np.full((ntemp, 3), np.nan)
    excluded_cv_fraction = np.full(ntemp, np.nan)

    for index, temperature in enumerate(np.asarray(temperatures_K, dtype=float)):
        cv = mode_heat_capacity_J_K(float(temperature), frequencies)
        weighted_cv_all = qweights[:, None] * cv
        cv_all_sum = float(np.sum(weighted_cv_all))
        cv_valid_sum = float(np.sum(weighted_cv_all[valid]))
        excluded_cv_fraction[index] = (
            1.0 - cv_valid_sum / cv_all_sum if cv_all_sum > 0.0 else math.nan
        )
        for component in range(6):
            integrals[index, component] = float(
                np.sum(weighted_cv_all[valid] * gamma[component][valid]) / weight_sum
            )
            macro_gamma[index, component] = (
                float(np.sum(weighted_cv_all[valid] * gamma[component][valid]) / cv_valid_sum)
                if cv_valid_sum > 0.0
                else math.nan
            )

        I = integrals[index]
        I_hyd = HYDROSTATIC_VOIGT * float(np.mean(I[:3]))
        I_dev = I - I_hyd
        alpha = compliance_Pa @ I / volume_m3
        alpha_hyd = compliance_Pa @ I_hyd / volume_m3
        alpha_dev = compliance_Pa @ I_dev / volume_m3
        alpha_voigt[index] = alpha
        alpha_volume[index] = float(np.dot(HYDROSTATIC_VOIGT, alpha))
        alpha_volume_hyd[index] = float(np.dot(HYDROSTATIC_VOIGT, alpha_hyd))
        alpha_volume_dev[index] = float(np.dot(HYDROSTATIC_VOIGT, alpha_dev))
        denominator = abs(alpha_volume_hyd[index]) + abs(alpha_volume_dev[index])
        F_ani[index] = abs(alpha_volume_dev[index]) / denominator if denominator > 0.0 else math.nan

        if axis_vectors:
            alpha_tensor = strain_voigt_to_tensor(alpha)
            for axis_index, axis in enumerate(("a", "b", "c")):
                direction = np.asarray(axis_vectors[axis], dtype=float)
                alpha_directional[index, axis_index] = float(
                    direction @ alpha_tensor @ direction
                )

    result = {
        "temperatures_K": np.asarray(temperatures_K, dtype=float),
        "gruneisen_integrals_J_per_K": integrals,
        "macro_strain_gruneisen": macro_gamma,
        "alpha_voigt_per_K": alpha_voigt,
        "alpha_volume_per_K": alpha_volume,
        "alpha_volume_hyd_per_K": alpha_volume_hyd,
        "alpha_volume_dev_per_K": alpha_volume_dev,
        "F_ani": F_ani,
        "alpha_directional_per_K": alpha_directional,
        "excluded_heat_capacity_fraction": excluded_cv_fraction,
    }
    finite_excluded = excluded_cv_fraction[np.isfinite(excluded_cv_fraction)]
    quality = {
        "frequency_cutoff_THz": float(frequency_cutoff_THz),
        "reference_mode_count": int(frequencies.size),
        "reference_imaginary_or_zero_count": int(np.sum(~base_valid)),
        "nonfinite_gamma_mode_count": int(np.sum(base_valid & ~gamma_finite)),
        "valid_mode_count": int(np.sum(valid)),
        "gamma_statistics": [
            gamma_statistics(gamma[component], base_valid & np.isfinite(gamma[component]))
            for component in range(6)
        ],
        "max_excluded_heat_capacity_fraction": (
            float(np.max(finite_excluded)) if finite_excluded.size else None
        ),
        "axis_mapping_status": axis_mapping.get("status") if axis_mapping else "not_requested",
    }
    return result, quality


def summarize_effective_isotropy(
    response: dict[str, np.ndarray],
    fani_threshold: float = 0.20,
    sign_tolerance_micro_per_K: float = 1.0e-3,
) -> dict[str, Any]:
    """Apply the temperature-wide effective-isotropy screen to a response."""

    tolerance = float(sign_tolerance_micro_per_K) * 1.0e-6
    alpha_volume = np.asarray(response["alpha_volume_per_K"], dtype=float)
    fani = np.asarray(response["F_ani"], dtype=float)
    macro = np.asarray(response["macro_strain_gruneisen"], dtype=float)
    directional = np.asarray(response["alpha_directional_per_K"], dtype=float)

    finite_fani = fani[np.isfinite(fani)]
    fani_complete = bool(finite_fani.size == fani.size and fani.size > 0)
    fani_max = float(np.max(finite_fani)) if finite_fani.size else None
    fani_pass = bool(fani_complete and fani_max is not None and fani_max <= fani_threshold)

    gamma_bar = np.mean(macro[:, :3], axis=1)
    sign_mask = (np.abs(alpha_volume) > tolerance) & np.isfinite(gamma_bar)
    sign_match_values = np.sign(alpha_volume[sign_mask]) == np.sign(gamma_bar[sign_mask])
    sign_match_complete = bool(np.any(sign_mask))
    sign_match_pass = bool(sign_match_complete and np.all(sign_match_values))

    directional_available = bool(np.all(np.isfinite(directional)))
    mixed_sign_any = None
    if directional_available:
        mixed = []
        for row in directional:
            significant = row[np.abs(row) > tolerance]
            mixed.append(bool(significant.size > 1 and np.min(significant) < 0.0 < np.max(significant)))
        mixed_sign_any = bool(any(mixed))
    directional_pass = directional_available and not bool(mixed_sign_any)

    passed = bool(fani_pass and sign_match_pass and directional_pass)
    if passed:
        status = "effective_isotropic_candidate"
    elif not fani_complete or not sign_match_complete or not directional_available:
        status = "unresolved"
    else:
        status = "anisotropic_mechanism"
    return {
        "status": status,
        "passed": passed,
        "fani_threshold": float(fani_threshold),
        "fani_max": fani_max,
        "fani_pass": fani_pass,
        "sign_tolerance_micro_per_K": float(sign_tolerance_micro_per_K),
        "alphaV_sign_matches_mean_gamma": sign_match_pass,
        "alphaV_sign_match_evaluated_temperature_count": int(np.sum(sign_mask)),
        "directional_alpha_available": directional_available,
        "directional_mixed_sign_any": mixed_sign_any,
        "directional_no_mixed_sign_pass": directional_pass,
    }


def input_fingerprint(
    root_poscar: Path,
    elastic_poscar: Path,
    elastic_tensor: Path,
    model_path: Path | None,
    parameters: V2Parameters,
    runner_path: Path,
    core_path: Path,
) -> dict[str, Any]:
    files = {
        "root_poscar": {"path": str(root_poscar), "sha256": sha256_file(root_poscar)},
        "elastic_poscar": {"path": str(elastic_poscar), "sha256": sha256_file(elastic_poscar)},
        "elastic_tensor": {"path": str(elastic_tensor), "sha256": sha256_file(elastic_tensor)},
        "runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
        "core": {"path": str(core_path), "sha256": sha256_file(core_path)},
    }
    if model_path is not None and model_path.is_file():
        files["model"] = {"path": str(model_path), "sha256": sha256_file(model_path)}
    else:
        files["model"] = {"path": str(model_path) if model_path else None, "sha256": None}
    payload = {"files": files, "parameters": asdict(parameters)}
    payload["fingerprint_sha256"] = stable_json_hash(payload)
    return payload


def rows_to_text_table(header: Iterable[str], rows: np.ndarray, fmt: str = "%.10e") -> str:
    from io import StringIO

    buffer = StringIO()
    np.savetxt(buffer, np.asarray(rows), header=" ".join(header), fmt=fmt)
    return buffer.getvalue()
