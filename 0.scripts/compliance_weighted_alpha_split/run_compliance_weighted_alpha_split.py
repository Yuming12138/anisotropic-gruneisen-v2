#!/usr/bin/env python
"""Run a direct compliance-weighted alphaV positive/negative decomposition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import shutil
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from alpha_split_core import (
    apply_engineering_strain_vector,
    compliance_weighted_path,
    compute_alpha_volume_split,
    path_state_report,
    summarize_at_temperature,
)
from plot_alpha_split_results import (
    ALPHA_SPLIT_PNG,
    PLOT_METADATA_JSON,
    QHA_COMPARISON_PNG,
    generate_result_plots,
)
from v2_runtime_adapter import (
    DiagnosticGruneisenMesh,
    V2Parameters,
    V2_CORE_PATH,
    V2_RUNNER_PATH,
    batch_fixed_cell_relax,
    calculate_force_constants,
    choose_device,
    choose_supercell_matrix,
    fixed_cell_relax,
    input_fingerprint,
    read_elastic_tensor,
    reference_force_stress_report,
    resolve_model_path,
    rows_to_text_table,
    runtime_versions,
    sha256_file,
    structure_axis_mapping,
    validate_elastic_tensor,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
CORE_PATH = SCRIPT_DIR / "alpha_split_core.py"
ADAPTER_PATH = SCRIPT_DIR / "v2_runtime_adapter.py"
DEFAULT_RESULT_SUBDIR = "gruneisen_alpha_split_1M_v1"
COMPLETE_ARTIFACTS = (
    "quality_report.json",
    "run_metadata.json",
    "effective_strain_path.json",
    "alpha_volume_split.dat",
    "alpha_volume_split_target.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-dir", type=Path, required=True)
    parser.add_argument("--result-subdir", default=DEFAULT_RESULT_SUBDIR)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-size", choices=("1M", "5M"), default="1M")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--strain", type=float, default=0.005)
    parser.add_argument("--fallback-strain", type=float, default=0.0025)
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--mesh", type=int, nargs=3, default=(20, 20, 20))
    parser.add_argument("--min-supercell-length", type=float, default=12.0)
    parser.add_argument("--supercell", type=int, nargs=3, default=None)
    parser.add_argument("--fmax", type=float, default=1.0e-3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-internal-relax-displacement", type=float, default=1.0)
    parser.add_argument("--batch-relax", action="store_true")
    parser.add_argument("--batch-relax-atom-cap", type=int, default=1024)
    parser.add_argument("--frequency-cutoff", type=float, default=1.0e-4)
    parser.add_argument("--effective-gamma-zero-tolerance", type=float, default=0.0)
    parser.add_argument("--tmin", type=float, default=10.0)
    parser.add_argument("--tmax", type=float, default=1000.0)
    parser.add_argument("--tstep", type=float, default=10.0)
    parser.add_argument("--target-temperature", type=float, default=300.0)
    parser.add_argument("--minimum-reportable-contribution-micro", type=float, default=1.0e-3)
    parser.add_argument("--max-excluded-cv-fraction", type=float, default=0.05)
    parser.add_argument("--max-unresolved-cv-fraction", type=float, default=0.01)
    parser.add_argument("--max-unresolved-alpha-fraction", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-internal-relax", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--qha-thermal-expansion", type=Path, default=None)
    parser.add_argument("--plot-dpi", type=int, default=200)
    args = parser.parse_args(argv)

    result_subdir = Path(args.result_subdir)
    if result_subdir.is_absolute() or ".." in result_subdir.parts:
        raise SystemExit("--result-subdir must be relative to the material directory")
    if args.strain <= 0.0 or args.fallback_strain <= 0.0:
        raise SystemExit("strain amplitudes must be positive")
    if any(value <= 0 for value in args.mesh):
        raise SystemExit("mesh values must be positive")
    if args.tmax < args.tmin or args.tstep <= 0.0:
        raise SystemExit("invalid temperature range")
    if args.effective_gamma_zero_tolerance < 0.0:
        raise SystemExit("--effective-gamma-zero-tolerance must be non-negative")
    if not 0.0 <= args.max_excluded_cv_fraction <= 1.0:
        raise SystemExit("--max-excluded-cv-fraction must be between zero and one")
    if not 0.0 <= args.max_unresolved_cv_fraction <= 1.0:
        raise SystemExit("--max-unresolved-cv-fraction must be between zero and one")
    if not 0.0 <= args.max_unresolved_alpha_fraction <= 1.0:
        raise SystemExit("--max-unresolved-alpha-fraction must be between zero and one")
    if args.minimum_reportable_contribution_micro < 0.0:
        raise SystemExit("--minimum-reportable-contribution-micro must be non-negative")
    if args.batch_relax_atom_cap <= 0:
        raise SystemExit("--batch-relax-atom-cap must be positive")
    if args.max_internal_relax_displacement <= 0.0:
        raise SystemExit("--max-internal-relax-displacement must be positive")
    if args.plot_dpi <= 0:
        raise SystemExit("--plot-dpi must be positive")
    grid_position = (args.target_temperature - args.tmin) / args.tstep
    if (
        args.target_temperature < args.tmin
        or args.target_temperature > args.tmax
        or abs(grid_position - round(grid_position)) > 1.0e-8
    ):
        raise SystemExit("--target-temperature must lie exactly on the temperature grid")
    return args


def build_parameters(args: argparse.Namespace) -> V2Parameters:
    return V2Parameters(
        model_size=args.model_size,
        dtype=args.dtype,
        strain=float(args.strain),
        fallback_strain=float(args.fallback_strain),
        displacement_A=float(args.displacement),
        mesh=tuple(int(value) for value in args.mesh),
        min_supercell_length_A=float(args.min_supercell_length),
        internal_relax_fmax_eV_A=float(args.fmax),
        internal_relax_max_steps=int(args.max_steps),
        frequency_cutoff_THz=float(args.frequency_cutoff),
        temperature_min_K=float(args.tmin),
        temperature_max_K=float(args.tmax),
        temperature_step_K=float(args.tstep),
    )


def strict_phase_match(root: Structure, elastic: Structure) -> dict[str, Any]:
    formula_match = root.composition.reduced_composition == elastic.composition.reduced_composition
    atom_count_match = len(root) == len(elastic)
    matcher = StructureMatcher(
        ltol=0.25,
        stol=0.5,
        angle_tol=5.0,
        primitive_cell=False,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )
    try:
        structure_match = bool(matcher.fit(root, elastic))
    except Exception as error:
        structure_match = False
        matcher_error = f"{type(error).__name__}:{error}"
    else:
        matcher_error = None
    return {
        "formula_match": formula_match,
        "atom_count_match": atom_count_match,
        "structure_matcher_fit": structure_match,
        "structure_matcher_error": matcher_error,
        "status": (
            "ok"
            if formula_match and atom_count_match and structure_match
            else "root_elastic_phase_mismatch"
        ),
    }


def relaxation_branch_report(
    initial: Any, relaxed: Any, max_displacement_A: float
) -> dict[str, Any]:
    symbols_match = initial.get_chemical_symbols() == relaxed.get_chemical_symbols()
    cell_match = bool(
        np.allclose(initial.cell.array, relaxed.cell.array, rtol=1.0e-10, atol=1.0e-10)
    )
    fractional_delta = relaxed.get_scaled_positions(wrap=False) - initial.get_scaled_positions(
        wrap=False
    )
    fractional_delta -= np.rint(fractional_delta)
    cartesian_delta = fractional_delta @ np.asarray(initial.cell.array, dtype=float)
    displacement = np.linalg.norm(cartesian_delta, axis=1)
    maximum = float(np.max(displacement)) if displacement.size else 0.0
    rms = float(np.sqrt(np.mean(displacement**2))) if displacement.size else 0.0
    matcher = StructureMatcher(
        ltol=0.05,
        stol=0.5,
        angle_tol=1.0,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
        allow_subset=False,
    )
    try:
        structure_match = bool(
            matcher.fit(
                AseAtomsAdaptor.get_structure(initial),
                AseAtomsAdaptor.get_structure(relaxed),
            )
        )
    except Exception as error:
        structure_match = False
        matcher_error = f"{type(error).__name__}:{error}"
    else:
        matcher_error = None
    passed = bool(
        symbols_match
        and cell_match
        and structure_match
        and maximum <= float(max_displacement_A)
    )
    return {
        "status": "ok" if passed else "relaxation_structure_branch_changed",
        "symbols_match": symbols_match,
        "cell_match": cell_match,
        "structure_matcher_fit": structure_match,
        "structure_matcher_error": matcher_error,
        "maximum_mapped_displacement_A": maximum,
        "rms_mapped_displacement_A": rms,
        "maximum_allowed_displacement_A": float(max_displacement_A),
    }


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    material_dir = args.material_dir.expanduser().resolve()
    result_dir = material_dir / args.result_subdir
    original_root_poscar = material_dir / "POSCAR"
    elastic_dir = material_dir / "elastic"
    elastic_poscar = elastic_dir / "POSCAR"
    elastic_tensor = elastic_dir / "ELASTIC_TENSOR"
    missing = [
        str(path)
        for path in (original_root_poscar, elastic_poscar, elastic_tensor)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing_required_inputs:" + ";".join(missing))

    original_root_structure = Structure.from_file(original_root_poscar)
    elastic_structure = Structure.from_file(elastic_poscar)
    stiffness_raw = read_elastic_tensor(elastic_tensor)
    stiffness, elastic_report = validate_elastic_tensor(stiffness_raw)
    compliance = np.linalg.inv(stiffness) if elastic_report["positive_definite"] else None
    original_phase_report = strict_phase_match(
        original_root_structure, elastic_structure
    )
    original_axis_mapping = structure_axis_mapping(
        original_root_structure, elastic_structure
    )
    optimized_poscar = material_dir / "opt" / "CONTCAR"
    if optimized_poscar.is_file():
        optimized_structure = Structure.from_file(optimized_poscar)
        optimized_phase_report = strict_phase_match(
            optimized_structure, elastic_structure
        )
        optimized_axis_mapping = structure_axis_mapping(
            optimized_structure, elastic_structure
        )
    else:
        optimized_phase_report = {"status": "optimized_structure_missing"}
        optimized_axis_mapping = {"status": "optimized_structure_missing"}
    root_poscar = elastic_poscar
    phase_report = strict_phase_match(elastic_structure, elastic_structure)
    axis_mapping = structure_axis_mapping(elastic_structure, elastic_structure)
    structure_source = "elastic_poscar"
    phase_report["structure_source"] = structure_source
    phase_report["structure_path"] = str(root_poscar)
    phase_report["original_root_poscar"] = str(original_root_poscar)
    axis_mapping["structure_source"] = structure_source
    axis_mapping["structure_path"] = str(root_poscar)
    axis_mapping["original_root_poscar"] = str(original_root_poscar)
    path = compliance_weighted_path(compliance) if compliance is not None else None
    states = path_state_report(path, args.strain) if path is not None else None

    if args.supercell is None:
        supercell = choose_supercell_matrix(
            elastic_structure.lattice.matrix,
            minimum_length_A=float(args.min_supercell_length),
        )
        supercell_source = "automatic_minimum_length"
    else:
        supercell = np.diag(np.asarray(args.supercell, dtype=int))
        supercell_source = "explicit_cli"
    model_path = resolve_model_path(args.model, args.model_size)
    parameters = build_parameters(args)
    versions = runtime_versions()

    issues: list[str] = []
    if not elastic_report["positive_definite"]:
        issues.append("elastic_not_positive_definite")
    if elastic_report["ill_conditioned"]:
        issues.append("elastic_ill_conditioned")
    if original_phase_report["status"] != "ok":
        issues.append("original_root_elastic_phase_mismatch")
    if original_axis_mapping["status"] != "ok":
        issues.append("original_root_elastic_axis_mapping_failed")
    if optimized_poscar.is_file() and optimized_phase_report["status"] != "ok":
        issues.append("optimized_elastic_phase_mismatch")
    if optimized_poscar.is_file() and optimized_axis_mapping["status"] != "ok":
        issues.append("optimized_elastic_axis_mapping_failed")
    if not model_path.is_file():
        issues.append("mattersim_model_missing_in_current_environment")
    if not (elastic_dir / "calculation_metadata.json").is_file():
        issues.append("elastic_calculation_metadata_missing")
    if states is not None:
        if any(item["deformation_determinant"] <= 0.0 for item in states.values()):
            issues.append("nonpositive_path_deformation_determinant")

    blocking_names = {
        "elastic_not_positive_definite",
        "elastic_ill_conditioned",
        "mattersim_model_missing_in_current_environment",
        "nonpositive_path_deformation_determinant",
    }
    blocking = [issue for issue in issues if issue in blocking_names]
    execution = {
        "method": "direct_compliance_weighted_volumetric_path",
        "path_normalization": "max_abs_principal_strain_equals_one",
        "effective_gamma_zero_tolerance_1_per_GPa": float(
            args.effective_gamma_zero_tolerance
        ),
        "target_temperature_K": float(args.target_temperature),
        "minimum_reportable_contribution_micro_per_K": float(
            args.minimum_reportable_contribution_micro
        ),
        "max_excluded_cv_fraction": float(args.max_excluded_cv_fraction),
        "max_unresolved_cv_fraction": float(args.max_unresolved_cv_fraction),
        "max_unresolved_alpha_fraction": float(args.max_unresolved_alpha_fraction),
        "max_internal_relax_displacement_A": float(
            args.max_internal_relax_displacement
        ),
        "supercell_matrix": supercell.tolist(),
        "device": args.device,
        "internal_relax_method": (
            "skipped"
            if args.skip_internal_relax
            else (
                "fixed_cell_batch_bfgs"
                if args.batch_relax
                else "fixed_cell_sequential_bfgs"
            )
        ),
        "batch_relax_atom_cap": args.batch_relax_atom_cap if args.batch_relax else None,
        "adapter_sha256": sha256_file(ADAPTER_PATH),
        "shared_v2_core_sha256": sha256_file(V2_CORE_PATH),
        "shared_v2_runner_sha256": sha256_file(V2_RUNNER_PATH),
        "runtime_versions": versions,
    }
    fingerprint = input_fingerprint(
        root_poscar=elastic_poscar,
        elastic_poscar=elastic_poscar,
        elastic_tensor=elastic_tensor,
        model_path=model_path if model_path.is_file() else None,
        parameters=parameters,
        runner_path=SCRIPT_PATH,
        core_path=CORE_PATH,
        execution=execution,
    )
    report = {
        "schema_version": 1,
        "method": "direct_compliance_weighted_volumetric_path",
        "material": material_dir.name,
        "material_dir": str(material_dir),
        "result_dir": str(result_dir),
        "status": "failed" if blocking else ("warning" if issues else "ok"),
        "blocking_issues": blocking,
        "issues": issues,
        "root_poscar": str(root_poscar),
        "original_root_poscar": str(original_root_poscar),
        "structure_source": structure_source,
        "elastic_poscar": str(elastic_poscar),
        "elastic_tensor": str(elastic_tensor),
        "phase_consistency": phase_report,
        "original_phase_consistency": original_phase_report,
        "optimized_poscar": str(optimized_poscar),
        "optimized_phase_consistency": optimized_phase_report,
        "axis_mapping": axis_mapping,
        "original_axis_mapping": original_axis_mapping,
        "optimized_axis_mapping": optimized_axis_mapping,
        "elastic": elastic_report,
        "effective_strain_path": path,
        "path_states": states,
        "supercell_matrix": supercell.tolist(),
        "supercell_source": supercell_source,
        "model_path": str(model_path),
        "model_present": model_path.is_file(),
        "parameters": asdict(parameters),
        "split_parameters": {
            "effective_gamma_zero_tolerance_1_per_GPa": float(
                args.effective_gamma_zero_tolerance
            ),
            "target_temperature_K": float(args.target_temperature),
            "minimum_reportable_contribution_micro_per_K": float(
                args.minimum_reportable_contribution_micro
            ),
            "max_excluded_cv_fraction": float(args.max_excluded_cv_fraction),
            "max_unresolved_cv_fraction": float(args.max_unresolved_cv_fraction),
            "max_unresolved_alpha_fraction": float(args.max_unresolved_alpha_fraction),
            "max_internal_relax_displacement_A": float(
                args.max_internal_relax_displacement
            ),
        },
        "fingerprint": fingerprint,
        "runtime_versions": versions,
    }
    context = {
        "material_dir": material_dir,
        "result_dir": result_dir,
        "root_poscar": root_poscar,
        "original_root_poscar": original_root_poscar,
        "elastic_poscar": elastic_poscar,
        "elastic_tensor": elastic_tensor,
        "stiffness_GPa": stiffness,
        "compliance_1_per_GPa": compliance,
        "effective_strain_path": path,
        "path_states": states,
        "supercell_matrix": supercell,
        "model_path": model_path,
        "parameters": parameters,
        "axis_mapping": axis_mapping,
        "phase_consistency": phase_report,
        "fingerprint": fingerprint,
    }
    return report, context


def write_preflight_outputs(report: dict[str, Any], context: dict[str, Any]) -> None:
    result_dir: Path = context["result_dir"]
    reference_dir = result_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(context["elastic_poscar"], reference_dir / "POSCAR")
    shutil.copy2(context["root_poscar"], reference_dir / "ROOT_POSCAR")
    shutil.copy2(
        context["original_root_poscar"], reference_dir / "ORIGINAL_ROOT_POSCAR"
    )
    write_json(reference_dir / "structure_mapping.json", context["axis_mapping"])
    write_json(reference_dir / "phase_consistency.json", context["phase_consistency"])
    write_json(result_dir / "preflight_report.json", report)
    if context["effective_strain_path"] is None:
        path_payload = {"status": "unavailable_invalid_elastic_tensor", "states": None}
    else:
        path_payload = {
            **context["effective_strain_path"],
            "status": "ok",
            "states": context["path_states"],
        }
    write_json(result_dir / "effective_strain_path.json", path_payload)
    np.savetxt(
        result_dir / "elastic_tensor_used.dat",
        context["stiffness_GPa"],
        header="Stiffness tensor C_ij in GPa; Voigt order xx yy zz yz xz xy",
        fmt="%16.8f",
    )
    metadata_path = result_dir / "run_metadata.json"
    preserve_complete = False
    if metadata_path.is_file():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            preserve_complete = bool(
                existing.get("calculation_status") == "complete"
                and existing.get("fingerprint", {}).get("fingerprint_sha256")
                == context["fingerprint"]["fingerprint_sha256"]
            )
        except Exception:
            preserve_complete = False
    if not preserve_complete:
        write_json(
            metadata_path,
            {
                "schema_version": 1,
                "calculation_status": "preflight_complete",
                "method": report["method"],
                "fingerprint": context["fingerprint"],
                "parameters": report["parameters"],
                "split_parameters": report["split_parameters"],
                "runtime_versions": report["runtime_versions"],
            },
        )


def completed_result_matches(result_dir: Path, fingerprint_sha256: str) -> bool:
    complete_path = result_dir / "calculation_complete.json"
    if not complete_path.is_file():
        return False
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        complete.get("status") == "complete"
        and complete.get("fingerprint_sha256") == fingerprint_sha256
        and all((result_dir / name).is_file() for name in COMPLETE_ARTIFACTS)
    )


def update_result_plots(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    if args.skip_plots:
        return None
    result_dir: Path = context["result_dir"]
    try:
        return generate_result_plots(
            result_dir,
            material_dir=context["material_dir"],
            qha_thermal_expansion=args.qha_thermal_expansion,
            target_temperature_K=float(args.target_temperature),
            dpi=int(args.plot_dpi),
        )
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "failure": f"{type(error).__name__}:{error}",
        }
        write_json(result_dir / PLOT_METADATA_JSON, report)
        print(
            f"[alpha-split] plotting warning: {report['failure']}",
            file=sys.stderr,
        )
        return report


def assess_split_readiness(
    quality: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    hard_failures: list[str] = []
    fallback_reasons: list[str] = []
    reference = quality.get("reference_force_stress", {})
    reference_force = float(reference.get("max_force_eV_A", math.inf))
    reference_stress = float(reference.get("max_abs_stress_GPa", math.inf))
    if not math.isfinite(reference_force) or reference_force > 1.0e-3:
        hard_failures.append("reference_residual_force_too_large")
    if (
        reference.get("stress_status") != "ok"
        or not math.isfinite(reference_stress)
        or reference_stress > 0.1
    ):
        hard_failures.append("reference_residual_stress_too_large")
    if quality.get("phase_consistency_status") != "ok":
        hard_failures.append("root_elastic_phase_mismatch")
    relaxations = quality.get("internal_relaxation", {})
    acceptable_relax = {"converged"}
    if args.skip_internal_relax:
        acceptable_relax.add("skipped_by_user")
    if any(report.get("status") not in acceptable_relax for report in relaxations.values()):
        hard_failures.append("internal_relaxation_not_converged")
    if any(
        report.get("branch_consistency", {}).get("status") != "ok"
        for report in relaxations.values()
    ):
        hard_failures.append("relaxation_structure_branch_changed")
    excluded = quality.get("max_excluded_heat_capacity_fraction")
    if excluded is None or float(excluded) > float(args.max_excluded_cv_fraction):
        hard_failures.append("excluded_heat_capacity_fraction_too_large")
    unresolved = quality.get("max_unresolved_heat_capacity_fraction")
    if unresolved is None or float(unresolved) > float(args.max_unresolved_cv_fraction):
        hard_failures.append("unresolved_heat_capacity_fraction_too_large")
    unresolved_alpha = quality.get("max_unresolved_alpha_fraction")
    if unresolved_alpha is None or float(unresolved_alpha) > float(
        args.max_unresolved_alpha_fraction
    ):
        hard_failures.append("unresolved_alpha_fraction_too_large")
    if int(quality.get("reference_imaginary_or_zero_count", 0)) > 3:
        fallback_reasons.append("reference_nonacoustic_imaginary_modes")
    imaginary = quality.get("strained_imaginary_diagnostics", {})
    if int(imaginary.get("minus_imaginary_mode_count", 0)) + int(
        imaginary.get("plus_imaginary_mode_count", 0)
    ) > 0:
        fallback_reasons.append("strain_induced_imaginary_modes")
    stats = quality.get("effective_gamma_statistics_1_per_GPa", {})
    path_scale = float(quality.get("path_scale_1_per_GPa", 0.0))
    gamma_path_abs_max = (
        float(stats.get("abs_max", 0.0)) / path_scale if path_scale > 0.0 else math.inf
    )
    if gamma_path_abs_max > 500.0:
        fallback_reasons.append("extreme_path_gruneisen_parameter")
    status = "failed" if hard_failures else (
        "requires_strain_check" if fallback_reasons else "ready"
    )
    return {
        "status": status,
        "hard_failures": hard_failures,
        "fallback_required": bool(fallback_reasons),
        "fallback_reasons": fallback_reasons,
        "observed": {
            "max_excluded_heat_capacity_fraction": excluded,
            "max_unresolved_heat_capacity_fraction": unresolved,
            "max_unresolved_alpha_fraction": unresolved_alpha,
            "max_abs_gamma_path": gamma_path_abs_max,
            "reference_max_force_eV_A": reference.get("max_force_eV_A"),
            "reference_max_abs_stress_GPa": reference.get("max_abs_stress_GPa"),
        },
    }


def run_calculation(
    args: argparse.Namespace, report: dict[str, Any], context: dict[str, Any]
) -> None:
    if report["blocking_issues"]:
        raise RuntimeError("preflight_blocked:" + ";".join(report["blocking_issues"]))
    model_path: Path = context["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"MatterSim model not found: {model_path}")

    from mattersim.forcefield.potential import MatterSimCalculator

    device = choose_device(args.device)
    calculator = MatterSimCalculator(
        load_path=str(model_path),
        device=device,
        dtype=args.dtype,
        compute_stress=True,
        batch_converter=False,
    )
    result_dir: Path = context["result_dir"]
    work_dir = result_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    reference_atoms = read(context["elastic_poscar"])
    reference_report = reference_force_stress_report(reference_atoms, calculator)
    write_json(result_dir / "reference" / "residual_force_stress.json", reference_report)
    parameters: V2Parameters = context["parameters"]
    path = context["effective_strain_path"]
    direction = np.asarray(path["normalized_direction"], dtype=float)
    common_state = {
        "input_fingerprint_sha256": context["fingerprint"]["fingerprint_sha256"],
        "method": "direct_compliance_weighted_volumetric_path",
        "path_normalization": path["normalization"],
        "normalized_direction": direction.tolist(),
        "path_scale_1_per_GPa": float(path["path_scale_1_per_GPa"]),
        "volumetric_compliance_beta_1_per_GPa": float(
            path["volumetric_compliance_beta_1_per_GPa"]
        ),
        "supercell_matrix": context["supercell_matrix"].tolist(),
        "displacement_A": parameters.displacement_A,
        "primitive_matrix": np.eye(3).tolist(),
        "model_sha256": sha256_file(model_path),
        "dtype": args.dtype,
        "internal_relax_method": (
            "skipped"
            if args.skip_internal_relax
            else (
                "fixed_cell_batch_bfgs"
                if args.batch_relax
                else "fixed_cell_sequential_bfgs"
            )
        ),
    }
    reference_state = {
        "method": "shared_reference_force_constants",
        "reference_poscar_sha256": sha256_file(context["elastic_poscar"]),
        "supercell_matrix": context["supercell_matrix"].tolist(),
        "displacement_A": parameters.displacement_A,
        "primitive_matrix": np.eye(3).tolist(),
        "model_sha256": sha256_file(model_path),
        "dtype": args.dtype,
        "device": device,
        "calculator": {
            "compute_stress": True,
            "batch_converter": False,
        },
        "runtime_versions": runtime_versions(),
        "alpha_split_core_sha256": sha256_file(CORE_PATH),
        "shared_v2_core_sha256": sha256_file(V2_CORE_PATH),
        "shared_v2_runner_sha256": sha256_file(V2_RUNNER_PATH),
    }

    print("[alpha-split] reference force constants", flush=True)
    phonon_zero = calculate_force_constants(
        atoms=reference_atoms,
        calculator=calculator,
        run_dir=work_dir / "strain_0",
        supercell_matrix=context["supercell_matrix"],
        displacement_A=parameters.displacement_A,
        state_fingerprint=reference_state,
        resume=args.resume,
        force=args.force,
    )

    state_specs: list[tuple[int, str, np.ndarray, Any, Path]] = []
    for sign, tag in ((-1, "minus"), (1, "plus")):
        eta = sign * float(parameters.strain) * direction
        state_name = f"cw_{tag}"
        print(
            f"[alpha-split] {state_name}: max principal strain "
            f"{sign * parameters.strain:+.6f}",
            flush=True,
        )
        strained = apply_engineering_strain_vector(reference_atoms, eta)
        state_specs.append((sign, state_name, eta, strained, work_dir / state_name))

    if args.batch_relax and not args.skip_internal_relax:
        relaxed_states = batch_fixed_cell_relax(
            [(name, atoms, run_dir) for _, name, _, atoms, run_dir in state_specs],
            calculator.potential,
            fmax=parameters.internal_relax_fmax_eV_A,
            max_steps=parameters.internal_relax_max_steps,
            max_natoms_per_batch=args.batch_relax_atom_cap,
        )
    else:
        relaxed_states = {
            name: fixed_cell_relax(
                atoms,
                calculator,
                run_dir,
                fmax=parameters.internal_relax_fmax_eV_A,
                max_steps=parameters.internal_relax_max_steps,
                skip=args.skip_internal_relax,
            )
            for _, name, _, atoms, run_dir in state_specs
        }

    phonons: dict[int, Any] = {}
    relax_reports: dict[str, Any] = {}
    for sign, state_name, eta, initial_state, run_dir in state_specs:
        relaxed, relax_report = relaxed_states[state_name]
        if relaxed.get_chemical_symbols() != reference_atoms.get_chemical_symbols():
            raise RuntimeError(f"atom_order_changed:{state_name}")
        branch_report = relaxation_branch_report(
            initial_state,
            relaxed,
            max_displacement_A=float(args.max_internal_relax_displacement),
        )
        relax_report["branch_consistency"] = branch_report
        write_json(run_dir / "internal_relax_report.json", relax_report)
        relax_reports[state_name] = relax_report
        phonons[sign] = calculate_force_constants(
            atoms=relaxed,
            calculator=calculator,
            run_dir=run_dir,
            supercell_matrix=context["supercell_matrix"],
            displacement_A=parameters.displacement_A,
            state_fingerprint={
                **common_state,
                "scalar_strain": sign * float(parameters.strain),
                "engineering_strain_voigt": eta.tolist(),
                "relaxed_structure_sha256": sha256_file(run_dir / "CONTCAR"),
            },
            resume=args.resume,
            force=args.force,
        )

    print("[alpha-split] direct compliance-weighted GruneisenMesh", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
            module="phonopy.gruneisen.core",
        )
        mesh = DiagnosticGruneisenMesh(
            phonon_zero.dynamical_matrix,
            phonons[1].dynamical_matrix,
            phonons[-1].dynamical_matrix,
            mesh=parameters.mesh,
            delta_strain=2.0 * parameters.strain,
            is_time_reversal=True,
            is_gamma_center=True,
            is_mesh_symmetry=False,
            imaginary_cutoff_THz=parameters.frequency_cutoff_THz,
        )
    qpoints = np.asarray(mesh.get_qpoints(), dtype=float)
    weights = np.asarray(mesh.get_weights(), dtype=float)
    frequencies = np.asarray(mesh.get_frequencies(), dtype=float)
    gamma_path = np.asarray(mesh.get_gruneisen(), dtype=float)

    response, quality = compute_alpha_volume_split(
        temperatures_K=parameters.temperatures(),
        frequencies_THz=frequencies,
        gamma_path=gamma_path,
        weights=weights,
        path_scale_1_per_GPa=float(path["path_scale_1_per_GPa"]),
        volume_A3=float(phonon_zero.primitive.volume),
        frequency_cutoff_THz=parameters.frequency_cutoff_THz,
        effective_gamma_zero_tolerance_1_per_GPa=float(
            args.effective_gamma_zero_tolerance
        ),
    )
    quality.update(
        {
            "method": "direct_compliance_weighted_volumetric_path",
            "path_normalization": path["normalization"],
            "path_scale_1_per_GPa": float(path["path_scale_1_per_GPa"]),
            "volumetric_compliance_beta_1_per_GPa": float(
                path["volumetric_compliance_beta_1_per_GPa"]
            ),
            "phase_consistency_status": context["phase_consistency"]["status"],
            "axis_mapping_status": context["axis_mapping"]["status"],
            "reference_force_stress": reference_report,
            "internal_relaxation": relax_reports,
            "strain_convergence_status": "not_checked_run_fallback_strain_separately",
            "strained_imaginary_diagnostics": {
                "minus_imaginary_mode_count": int(
                    np.dot(weights, np.asarray(mesh.minus_imaginary_counts_by_q, dtype=float))
                ),
                "plus_imaginary_mode_count": int(
                    np.dot(weights, np.asarray(mesh.plus_imaginary_counts_by_q, dtype=float))
                ),
                "minus_min_eigenvalue": float(np.min(mesh.minus_min_eigenvalues)),
                "plus_min_eigenvalue": float(np.min(mesh.plus_min_eigenvalues)),
            },
        }
    )
    quality["production_readiness"] = assess_split_readiness(quality, args)
    write_json(result_dir / "quality_report.json", quality)

    np.savez_compressed(
        result_dir / "effective_gruneisen_mesh.npz",
        qpoints=qpoints,
        weights=weights,
        frequencies_0_thz=frequencies,
        gamma_path=gamma_path,
        effective_gamma_1_per_GPa=response["effective_gamma_1_per_GPa"],
        valid_mask=response["valid_mask"],
        positive_mask=response["positive_mask"],
        negative_mask=response["negative_mask"],
        exact_zero_mask=response["exact_zero_mask"],
        unresolved_mask=response["unresolved_mask"],
        normalized_direction=direction,
        path_scale_1_per_GPa=float(path["path_scale_1_per_GPa"]),
    )

    temperatures = response["temperatures_K"]
    rows = np.column_stack(
        [
            temperatures,
            response["alpha_volume_positive_per_K"] * 1.0e6,
            response["alpha_volume_negative_per_K"] * 1.0e6,
            np.abs(response["alpha_volume_negative_per_K"]) * 1.0e6,
            response["alpha_volume_unresolved_signed_per_K"] * 1.0e6,
            response["alpha_volume_unresolved_absolute_bound_per_K"] * 1.0e6,
            response["alpha_volume_total_per_K"] * 1.0e6,
            response["ratio_abs_negative_to_positive"],
            response["ratio_lower_bound"],
            response["ratio_upper_bound"],
            response["excluded_heat_capacity_fraction"],
            response["unresolved_heat_capacity_fraction"],
            response["unresolved_alpha_fraction"],
        ]
    )
    (result_dir / "alpha_volume_split.dat").write_text(
        rows_to_text_table(
            [
                "T_K",
                "alphaV_positive_micro_per_K",
                "alphaV_negative_micro_per_K",
                "alphaV_negative_abs_micro_per_K",
                "alphaV_unresolved_signed_micro_per_K",
                "alphaV_unresolved_abs_bound_micro_per_K",
                "alphaV_total_micro_per_K",
                "ratio_abs_negative_to_positive",
                "ratio_lower_bound",
                "ratio_upper_bound",
                "excluded_Cv_fraction",
                "unresolved_Cv_fraction",
                "unresolved_alpha_fraction",
            ],
            rows,
        ),
        encoding="utf-8",
    )
    target = summarize_at_temperature(response, args.target_temperature)
    readiness = quality["production_readiness"]
    contribution_resolved = (
        target["alphaV_positive_micro_per_K"]
        >= float(args.minimum_reportable_contribution_micro)
        and target["alphaV_negative_abs_micro_per_K"]
        >= float(args.minimum_reportable_contribution_micro)
    )
    target["contribution_above_reporting_floor"] = bool(contribution_resolved)
    target["single_run_quality_ready"] = readiness["status"] == "ready"
    target["ratio_reportable"] = False
    target["ratio_status_reason"] = (
        "contribution_below_reporting_floor"
        if not contribution_resolved
        else (
            readiness["status"]
            if readiness["status"] != "ready"
            else "strain_convergence_not_checked"
        )
    )
    target["minimum_reportable_contribution_micro_per_K"] = float(
        args.minimum_reportable_contribution_micro
    )
    target_path = result_dir / "alpha_volume_split_target.json"
    write_json(target_path, target)
    if abs(float(args.target_temperature) - 300.0) < 1.0e-8:
        write_json(result_dir / "alpha_volume_split_300K.json", target)

    plot_report = update_result_plots(args, context)

    metadata = json.loads((result_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "calculation_status": "complete",
            "device": device,
            "runtime_versions": runtime_versions(),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "plots": plot_report,
        }
    )
    write_json(result_dir / "run_metadata.json", metadata)
    write_json(
        result_dir / "calculation_complete.json",
        {
            "status": "complete",
            "fingerprint_sha256": context["fingerprint"]["fingerprint_sha256"],
            "quality_report": str(result_dir / "quality_report.json"),
            "alpha_split_file": str(result_dir / "alpha_volume_split.dat"),
            "target_summary": str(target_path),
            "plot_metadata": (
                str(result_dir / PLOT_METADATA_JSON) if plot_report is not None else None
            ),
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report, context = preflight(args)
    result_dir: Path = context["result_dir"]
    fingerprint = context["fingerprint"]["fingerprint_sha256"]
    result_matches = completed_result_matches(result_dir, fingerprint)
    complete_path = result_dir / "calculation_complete.json"
    if (
        args.preflight_only
        and complete_path.is_file()
        and not result_matches
        and not args.force
    ):
        raise RuntimeError(
            "completed_result_fingerprint_mismatch_use_new_result_subdir_or_force"
        )
    reuse_completed = bool(args.resume and not args.force and result_matches)
    archive_completed = bool(
        complete_path.is_file()
        and (
            (not args.preflight_only and not reuse_completed)
            or (args.preflight_only and args.force and not result_matches)
        )
    )
    if archive_completed:
        for name in (
            "calculation_complete.json",
            *COMPLETE_ARTIFACTS,
            "alpha_volume_split_300K.json",
            "effective_gruneisen_mesh.npz",
            ALPHA_SPLIT_PNG,
            QHA_COMPARISON_PNG,
            PLOT_METADATA_JSON,
        ):
            current = result_dir / name
            if not current.exists():
                continue
            previous = result_dir / f"{name}.previous"
            if previous.exists():
                previous.unlink()
            current.replace(previous)
    write_preflight_outputs(report, context)
    if args.preflight_only:
        print(
            f"[alpha-split] preflight {report['status']}: {result_dir / 'preflight_report.json'}"
        )
        return
    if reuse_completed:
        update_result_plots(args, context)
        print(f"[alpha-split] completed result matches fingerprint: {result_dir}")
        return
    metadata_path = result_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["calculation_status"] = "running"
    metadata["fingerprint"] = context["fingerprint"]
    write_json(metadata_path, metadata)
    run_calculation(args, report, context)
    print(f"[alpha-split] complete: {result_dir}")
    quality = json.loads((result_dir / "quality_report.json").read_text(encoding="utf-8"))
    if quality.get("production_readiness", {}).get("status") == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[alpha-split] failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
