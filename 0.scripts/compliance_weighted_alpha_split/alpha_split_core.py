#!/usr/bin/env python
"""Pure scientific utilities for compliance-weighted alphaV sign splitting."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from ase import Atoms

from v2_runtime_adapter import (
    ANGSTROM3_TO_M3,
    GPA_TO_PA,
    HYDROSTATIC_VOIGT,
    mode_heat_capacity_J_K,
    strain_voigt_to_tensor,
)


def compliance_weighted_path(compliance_1_per_GPa: np.ndarray) -> dict[str, Any]:
    """Return a dimensionless mixed-strain direction for volumetric alpha.

    For a mode with six strain Gruneisen components ``gamma``, its actual
    volumetric thermal-expansion weight is

        chi = e.T @ S @ gamma = d.T @ gamma,

    where ``e=(1,1,1,0,0,0)`` and ``d=S.T@e``.  We normalize ``d`` so that the
    largest absolute principal strain of the resulting strain tensor is one.
    A calculation at scalar amplitude ``h`` therefore never exceeds ``h`` in
    principal-strain magnitude, while ``path_scale * gamma_path`` restores
    ``chi`` in 1/GPa.
    """

    compliance = np.asarray(compliance_1_per_GPa, dtype=float)
    if compliance.shape != (6, 6) or not np.all(np.isfinite(compliance)):
        raise ValueError("compliance_shape_or_finiteness_error")
    asymmetry_scale = max(float(np.linalg.norm(compliance)), 1.0e-30)
    asymmetry = float(np.linalg.norm(compliance - compliance.T) / asymmetry_scale)
    if asymmetry > 1.0e-8:
        raise ValueError(f"compliance_asymmetry_too_large:{asymmetry:.6g}")
    compliance = 0.5 * (compliance + compliance.T)

    e = np.asarray(HYDROSTATIC_VOIGT, dtype=float)
    raw_direction = compliance.T @ e
    beta = float(e @ compliance @ e)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"nonpositive_volumetric_compliance:{beta}")

    raw_tensor = strain_voigt_to_tensor(raw_direction)
    principal = np.linalg.eigvalsh(raw_tensor)
    scale = float(np.max(np.abs(principal)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid_path_scale:{scale}")
    direction = raw_direction / scale
    direction_tensor = strain_voigt_to_tensor(direction)
    direction_principal = np.linalg.eigvalsh(direction_tensor)
    trace = float(np.dot(e, direction))
    if trace <= 0.0:
        raise ValueError(f"nonpositive_path_trace:{trace}")

    return {
        "hydrostatic_selector": e,
        "compliance_1_per_GPa": compliance,
        "raw_direction_1_per_GPa": raw_direction,
        "volumetric_compliance_beta_1_per_GPa": beta,
        "path_scale_1_per_GPa": scale,
        "normalized_direction": direction,
        "normalized_strain_tensor": direction_tensor,
        "normalized_principal_strains": direction_principal,
        "normalized_trace": trace,
        "normalization": "max_abs_principal_strain_equals_one",
        "identity_e_dot_direction_times_scale": float(np.dot(e, direction) * scale),
        "identity_beta": beta,
    }


def apply_engineering_strain_vector(atoms: Atoms, engineering_voigt: np.ndarray) -> Atoms:
    """Apply a general Cartesian engineering strain to an ASE row-vector cell."""

    eta = np.asarray(engineering_voigt, dtype=float)
    if eta.shape != (6,) or not np.all(np.isfinite(eta)):
        raise ValueError("engineering_strain_vector_error")
    strain_tensor = strain_voigt_to_tensor(eta)
    deformation = np.eye(3) + strain_tensor
    determinant = float(np.linalg.det(deformation))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError(f"nonpositive_deformation_determinant:{determinant}")
    strained = atoms.copy()
    strained.set_cell(np.asarray(atoms.cell.array, dtype=float) @ deformation.T, scale_atoms=True)
    return strained


def path_state_report(path: dict[str, Any], scalar_strain: float) -> dict[str, Any]:
    """Describe the plus/minus deformation gradients before running forces."""

    direction = np.asarray(path["normalized_direction"], dtype=float)
    states: dict[str, Any] = {}
    for sign, name in ((-1.0, "minus"), (1.0, "plus")):
        eta = sign * float(scalar_strain) * direction
        tensor = strain_voigt_to_tensor(eta)
        deformation = np.eye(3) + tensor
        states[name] = {
            "engineering_voigt": eta,
            "strain_tensor": tensor,
            "principal_strains": np.linalg.eigvalsh(tensor),
            "deformation_determinant": float(np.linalg.det(deformation)),
            "maximum_absolute_principal_strain": float(
                np.max(np.abs(np.linalg.eigvalsh(tensor)))
            ),
        }
    return states


def effective_gamma_1_per_GPa(
    gamma_path: np.ndarray, path_scale_1_per_GPa: float
) -> np.ndarray:
    """Recover chi=e.T@S@gamma from the dimensionless path derivative."""

    values = np.asarray(gamma_path, dtype=float)
    scale = float(path_scale_1_per_GPa)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid_path_scale")
    return values * scale


def _signed_statistics(values: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values, dtype=float)[np.asarray(valid, dtype=bool)]
    if selected.size == 0:
        return {"count": 0}
    absolute = np.abs(selected)
    return {
        "count": int(selected.size),
        "negative_count": int(np.sum(selected < 0.0)),
        "positive_count": int(np.sum(selected > 0.0)),
        "zero_count": int(np.sum(selected == 0.0)),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
        "abs_median": float(np.median(absolute)),
        "abs_p90": float(np.quantile(absolute, 0.90)),
        "abs_p99": float(np.quantile(absolute, 0.99)),
        "abs_max": float(np.max(absolute)),
    }


def compute_alpha_volume_split(
    temperatures_K: np.ndarray,
    frequencies_THz: np.ndarray,
    gamma_path: np.ndarray,
    weights: np.ndarray,
    path_scale_1_per_GPa: float,
    volume_A3: float,
    frequency_cutoff_THz: float,
    effective_gamma_zero_tolerance_1_per_GPa: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Integrate per-mode volumetric contributions and split by final sign."""

    temperatures = np.asarray(temperatures_K, dtype=float)
    frequencies = np.asarray(frequencies_THz, dtype=float)
    gamma = np.asarray(gamma_path, dtype=float)
    qweights = np.asarray(weights, dtype=float)
    if gamma.shape != frequencies.shape:
        raise ValueError("gamma_frequency_shape_mismatch")
    if qweights.shape != (frequencies.shape[0],):
        raise ValueError("weight_shape_mismatch")
    if np.any(qweights < 0.0) or not np.all(np.isfinite(qweights)):
        raise ValueError("invalid_qpoint_weights")
    weight_sum = float(np.sum(qweights))
    if weight_sum <= 0.0:
        raise ValueError("invalid_qpoint_weights")
    volume_m3 = float(volume_A3) * ANGSTROM3_TO_M3
    if not math.isfinite(volume_m3) or volume_m3 <= 0.0:
        raise ValueError("invalid_reference_volume")
    tolerance = float(effective_gamma_zero_tolerance_1_per_GPa)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("invalid_effective_gamma_zero_tolerance")

    gamma_effective = effective_gamma_1_per_GPa(gamma, path_scale_1_per_GPa)
    base_valid = np.isfinite(frequencies) & (frequencies > float(frequency_cutoff_THz))
    valid = base_valid & np.isfinite(gamma_effective)
    positive = valid & (gamma_effective > tolerance)
    negative = valid & (gamma_effective < -tolerance)
    exact_zero = valid & (gamma_effective == 0.0)
    unresolved = valid & ~(positive | negative | exact_zero)

    ntemp = len(temperatures)
    alpha_positive = np.zeros(ntemp, dtype=float)
    alpha_negative = np.zeros(ntemp, dtype=float)
    alpha_unresolved_signed = np.zeros(ntemp, dtype=float)
    alpha_unresolved_absolute_bound = np.zeros(ntemp, dtype=float)
    alpha_total = np.zeros(ntemp, dtype=float)
    ratio = np.full(ntemp, np.nan)
    ratio_lower_bound = np.full(ntemp, np.nan)
    ratio_upper_bound = np.full(ntemp, np.nan)
    excluded_cv_fraction = np.full(ntemp, np.nan)
    unresolved_cv_fraction = np.full(ntemp, np.nan)
    unresolved_alpha_fraction = np.full(ntemp, np.nan)
    integral_positive = np.zeros(ntemp, dtype=float)
    integral_negative = np.zeros(ntemp, dtype=float)
    integral_unresolved = np.zeros(ntemp, dtype=float)

    gamma_effective_per_Pa = gamma_effective / GPA_TO_PA
    for index, temperature in enumerate(temperatures):
        cv = mode_heat_capacity_J_K(float(temperature), frequencies)
        weighted_cv = qweights[:, None] * cv
        cv_all = float(np.sum(weighted_cv))
        cv_valid = float(np.sum(weighted_cv[valid]))
        cv_unresolved = float(np.sum(weighted_cv[unresolved]))
        excluded_cv_fraction[index] = 1.0 - cv_valid / cv_all if cv_all > 0.0 else math.nan
        unresolved_cv_fraction[index] = cv_unresolved / cv_valid if cv_valid > 0.0 else math.nan

        mode_integrand = weighted_cv * gamma_effective_per_Pa / weight_sum
        integral_positive[index] = float(np.sum(mode_integrand[positive]))
        integral_negative[index] = float(np.sum(mode_integrand[negative]))
        integral_unresolved[index] = float(np.sum(mode_integrand[unresolved]))
        unresolved_abs = float(np.sum(np.abs(mode_integrand[unresolved])))

        alpha_positive[index] = integral_positive[index] / volume_m3
        alpha_negative[index] = integral_negative[index] / volume_m3
        alpha_unresolved_signed[index] = integral_unresolved[index] / volume_m3
        alpha_unresolved_absolute_bound[index] = unresolved_abs / volume_m3
        alpha_total[index] = (
            alpha_positive[index]
            + alpha_negative[index]
            + alpha_unresolved_signed[index]
        )
        if alpha_positive[index] > 0.0:
            ratio[index] = abs(alpha_negative[index]) / alpha_positive[index]
            ratio_lower_bound[index] = max(
                0.0,
                abs(alpha_negative[index]) - alpha_unresolved_absolute_bound[index],
            ) / (alpha_positive[index] + alpha_unresolved_absolute_bound[index])
            if alpha_positive[index] > alpha_unresolved_absolute_bound[index]:
                ratio_upper_bound[index] = (
                    abs(alpha_negative[index]) + alpha_unresolved_absolute_bound[index]
                ) / (alpha_positive[index] - alpha_unresolved_absolute_bound[index])
            else:
                ratio_upper_bound[index] = math.inf
        resolved_magnitude = alpha_positive[index] + abs(alpha_negative[index])
        bound_denominator = resolved_magnitude + alpha_unresolved_absolute_bound[index]
        unresolved_alpha_fraction[index] = (
            alpha_unresolved_absolute_bound[index] / bound_denominator
            if bound_denominator > 0.0
            else 0.0
        )

    identity_error = np.max(
        np.abs(
            alpha_total
            - alpha_positive
            - alpha_negative
            - alpha_unresolved_signed
        )
    )
    result = {
        "temperatures_K": temperatures,
        "alpha_volume_positive_per_K": alpha_positive,
        "alpha_volume_negative_per_K": alpha_negative,
        "alpha_volume_unresolved_signed_per_K": alpha_unresolved_signed,
        "alpha_volume_unresolved_absolute_bound_per_K": alpha_unresolved_absolute_bound,
        "alpha_volume_total_per_K": alpha_total,
        "ratio_abs_negative_to_positive": ratio,
        "ratio_lower_bound": ratio_lower_bound,
        "ratio_upper_bound": ratio_upper_bound,
        "excluded_heat_capacity_fraction": excluded_cv_fraction,
        "unresolved_heat_capacity_fraction": unresolved_cv_fraction,
        "unresolved_alpha_fraction": unresolved_alpha_fraction,
        "effective_integral_positive_J_per_K_per_Pa": integral_positive,
        "effective_integral_negative_J_per_K_per_Pa": integral_negative,
        "effective_integral_unresolved_J_per_K_per_Pa": integral_unresolved,
        "effective_gamma_1_per_GPa": gamma_effective,
        "valid_mask": valid,
        "positive_mask": positive,
        "negative_mask": negative,
        "exact_zero_mask": exact_zero,
        "unresolved_mask": unresolved,
    }
    finite_excluded = excluded_cv_fraction[np.isfinite(excluded_cv_fraction)]
    finite_unresolved = unresolved_cv_fraction[np.isfinite(unresolved_cv_fraction)]
    finite_unresolved_alpha = unresolved_alpha_fraction[
        np.isfinite(unresolved_alpha_fraction)
    ]
    quality = {
        "frequency_cutoff_THz": float(frequency_cutoff_THz),
        "effective_gamma_zero_tolerance_1_per_GPa": tolerance,
        "reference_mode_count": int(frequencies.size),
        "reference_imaginary_or_zero_count": int(np.sum(~base_valid)),
        "nonfinite_effective_gamma_mode_count": int(np.sum(base_valid & ~np.isfinite(gamma_effective))),
        "valid_mode_count": int(np.sum(valid)),
        "positive_mode_count": int(np.sum(positive)),
        "negative_mode_count": int(np.sum(negative)),
        "exact_zero_mode_count": int(np.sum(exact_zero)),
        "unresolved_mode_count": int(np.sum(unresolved)),
        "effective_gamma_statistics_1_per_GPa": _signed_statistics(gamma_effective, valid),
        "max_excluded_heat_capacity_fraction": (
            float(np.max(finite_excluded)) if finite_excluded.size else None
        ),
        "max_unresolved_heat_capacity_fraction": (
            float(np.max(finite_unresolved)) if finite_unresolved.size else None
        ),
        "max_unresolved_alpha_fraction": (
            float(np.max(finite_unresolved_alpha))
            if finite_unresolved_alpha.size
            else None
        ),
        "split_identity_max_abs_error_per_K": float(identity_error),
    }
    return result, quality


def summarize_at_temperature(
    response: dict[str, np.ndarray],
    target_temperature_K: float = 300.0,
    temperature_tolerance_K: float = 1.0e-8,
) -> dict[str, Any]:
    temperatures = np.asarray(response["temperatures_K"], dtype=float)
    if temperatures.size == 0:
        raise ValueError("empty_temperature_grid")
    index = int(np.argmin(np.abs(temperatures - float(target_temperature_K))))
    difference = abs(float(temperatures[index]) - float(target_temperature_K))
    if difference > float(temperature_tolerance_K):
        raise ValueError(
            f"target_temperature_not_on_grid:{target_temperature_K}:nearest={temperatures[index]}"
        )
    ratio = float(response["ratio_abs_negative_to_positive"][index])
    ratio_lower = float(response["ratio_lower_bound"][index])
    ratio_upper = float(response["ratio_upper_bound"][index])
    alpha_positive = float(response["alpha_volume_positive_per_K"][index])
    alpha_negative = float(response["alpha_volume_negative_per_K"][index])
    if alpha_positive > 0.0 and math.isfinite(ratio):
        ratio_raw_status = "finite"
    elif alpha_positive <= 0.0 and alpha_negative < 0.0:
        ratio_raw_status = "denominator_zero_negative_present"
    elif alpha_positive <= 0.0:
        ratio_raw_status = "denominator_zero"
    else:
        ratio_raw_status = "nonfinite"
    return {
        "target_temperature_K": float(target_temperature_K),
        "sampled_temperature_K": float(temperatures[index]),
        "alphaV_positive_micro_per_K": float(
            response["alpha_volume_positive_per_K"][index] * 1.0e6
        ),
        "alphaV_negative_micro_per_K": float(
            response["alpha_volume_negative_per_K"][index] * 1.0e6
        ),
        "alphaV_negative_abs_micro_per_K": float(
            abs(response["alpha_volume_negative_per_K"][index]) * 1.0e6
        ),
        "alphaV_unresolved_signed_micro_per_K": float(
            response["alpha_volume_unresolved_signed_per_K"][index] * 1.0e6
        ),
        "alphaV_unresolved_absolute_bound_micro_per_K": float(
            response["alpha_volume_unresolved_absolute_bound_per_K"][index] * 1.0e6
        ),
        "alphaV_total_micro_per_K": float(
            response["alpha_volume_total_per_K"][index] * 1.0e6
        ),
        "ratio_abs_negative_to_positive": ratio if math.isfinite(ratio) else None,
        "ratio_lower_bound": ratio_lower if math.isfinite(ratio_lower) else None,
        "ratio_upper_bound": ratio_upper if math.isfinite(ratio_upper) else None,
        "ratio_raw_status": ratio_raw_status,
        "excluded_heat_capacity_fraction": float(
            response["excluded_heat_capacity_fraction"][index]
        ),
        "unresolved_heat_capacity_fraction": float(
            response["unresolved_heat_capacity_fraction"][index]
        ),
        "unresolved_alpha_fraction": float(
            response["unresolved_alpha_fraction"][index]
        ),
    }


def compare_split_responses(
    primary: dict[str, np.ndarray],
    fallback: dict[str, np.ndarray],
    *,
    target_temperature_K: float = 100.0,
    relative_tolerance: float = 0.10,
    absolute_tolerance_micro_per_K: float = 0.5,
    ratio_absolute_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Compare h and h/2 aggregate contributions without band-index matching."""

    temperatures = np.asarray(primary["temperatures_K"], dtype=float)
    fallback_temperatures = np.asarray(fallback["temperatures_K"], dtype=float)
    if temperatures.shape != fallback_temperatures.shape or not np.allclose(
        temperatures, fallback_temperatures
    ):
        raise ValueError("strain_temperature_grid_mismatch")
    index = int(np.argmin(np.abs(temperatures - float(target_temperature_K))))
    difference = abs(float(temperatures[index]) - float(target_temperature_K))
    if difference > 1.0e-8:
        raise ValueError(
            f"target_temperature_not_on_grid:{target_temperature_K}:nearest={temperatures[index]}"
        )
    fields = {
        "positive": "alpha_volume_positive_per_K",
        "negative": "alpha_volume_negative_per_K",
        "total": "alpha_volume_total_per_K",
    }
    comparisons: dict[str, Any] = {}
    reasons: list[str] = []
    for label, field in fields.items():
        left = float(primary[field][index])
        right = float(fallback[field][index])
        difference = abs(left - right)
        scale = max(abs(left), abs(right), 1.0e-12)
        relative = difference / scale
        absolute_micro = difference * 1.0e6
        comparisons[label] = {
            "primary_micro_per_K": left * 1.0e6,
            "fallback_micro_per_K": right * 1.0e6,
            "relative_difference": relative,
            "absolute_difference_micro_per_K": absolute_micro,
        }
        if relative > relative_tolerance and absolute_micro > absolute_tolerance_micro_per_K:
            reasons.append(f"alphaV_{label}_not_converged")

    primary_ratio = float(primary["ratio_abs_negative_to_positive"][index])
    fallback_ratio = float(fallback["ratio_abs_negative_to_positive"][index])
    ratio_difference = (
        abs(primary_ratio - fallback_ratio)
        if math.isfinite(primary_ratio) and math.isfinite(fallback_ratio)
        else math.inf
    )
    if ratio_difference > ratio_absolute_tolerance:
        reasons.append("alphaV_ratio_not_converged")
    return {
        "status": "converged" if not reasons else "strain_derivative_unresolved",
        "reasons": reasons,
        "target_temperature_K": float(temperatures[index]),
        "components": comparisons,
        "ratio": {
            "primary": primary_ratio if math.isfinite(primary_ratio) else None,
            "fallback": fallback_ratio if math.isfinite(fallback_ratio) else None,
            "absolute_difference": ratio_difference if math.isfinite(ratio_difference) else None,
        },
        "thresholds": {
            "relative_tolerance": float(relative_tolerance),
            "absolute_tolerance_micro_per_K": float(absolute_tolerance_micro_per_K),
            "ratio_absolute_tolerance": float(ratio_absolute_tolerance),
        },
    }
