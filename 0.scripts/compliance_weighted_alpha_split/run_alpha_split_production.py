#!/usr/bin/env python
"""Run h and h/2 alpha-split stages and accept only a converged result."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from alpha_split_core import compare_split_responses
from compare_alpha_split_runs import load_alpha_split_table
from plot_alpha_split_results import (
    ALPHA_SPLIT_PNG,
    PLOT_METADATA_JSON,
    QHA_COMPARISON_PNG,
    generate_result_plots,
)
from v2_runtime_adapter import write_json


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_compliance_weighted_alpha_split.py"


def strain_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-dir", type=Path, required=True)
    parser.add_argument("--result-subdir", default="gruneisen_alpha_split_1M_v1")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-size", choices=("1M", "5M"), default="1M")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--primary-strain", type=float, default=0.005)
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
    parser.add_argument("--convergence-temperature", type=float, default=100.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    parser.add_argument("--absolute-tolerance-micro", type=float, default=0.5)
    parser.add_argument("--ratio-absolute-tolerance", type=float, default=0.10)
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
    result = Path(args.result_subdir)
    if result.is_absolute() or ".." in result.parts:
        raise SystemExit("--result-subdir must be relative to the material directory")
    if args.primary_strain <= 0.0 or args.fallback_strain <= 0.0:
        raise SystemExit("strain amplitudes must be positive")
    if args.fallback_strain >= args.primary_strain:
        raise SystemExit("--fallback-strain must be smaller than --primary-strain")
    if args.max_internal_relax_displacement <= 0.0:
        raise SystemExit("--max-internal-relax-displacement must be positive")
    if args.plot_dpi <= 0:
        raise SystemExit("--plot-dpi must be positive")
    if args.minimum_reportable_contribution_micro < 0.0:
        raise SystemExit("--minimum-reportable-contribution-micro must be non-negative")
    for value, label in (
        (args.max_excluded_cv_fraction, "--max-excluded-cv-fraction"),
        (args.max_unresolved_cv_fraction, "--max-unresolved-cv-fraction"),
        (args.max_unresolved_alpha_fraction, "--max-unresolved-alpha-fraction"),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{label} must be between zero and one")
    for temperature, label in (
        (args.target_temperature, "--target-temperature"),
        (args.convergence_temperature, "--convergence-temperature"),
    ):
        position = (temperature - args.tmin) / args.tstep
        if (
            temperature < args.tmin
            or temperature > args.tmax
            or abs(position - round(position)) > 1.0e-8
        ):
            raise SystemExit(f"{label} must lie exactly on the temperature grid")
    return args


def runner_command(
    args: argparse.Namespace, strain: float, stage_subdir: str
) -> list[str]:
    command = [
        str(args.python),
        str(RUNNER),
        "--material-dir",
        str(args.material_dir),
        "--result-subdir",
        stage_subdir,
        "--model-size",
        args.model_size,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--strain",
        str(strain),
        "--fallback-strain",
        str(args.fallback_strain),
        "--displacement",
        str(args.displacement),
        "--mesh",
        *[str(value) for value in args.mesh],
        "--min-supercell-length",
        str(args.min_supercell_length),
        "--fmax",
        str(args.fmax),
        "--max-steps",
        str(args.max_steps),
        "--max-internal-relax-displacement",
        str(args.max_internal_relax_displacement),
        "--batch-relax-atom-cap",
        str(args.batch_relax_atom_cap),
        "--frequency-cutoff",
        str(args.frequency_cutoff),
        "--effective-gamma-zero-tolerance",
        str(args.effective_gamma_zero_tolerance),
        "--tmin",
        str(args.tmin),
        "--tmax",
        str(args.tmax),
        "--tstep",
        str(args.tstep),
        "--target-temperature",
        str(args.target_temperature),
        "--minimum-reportable-contribution-micro",
        str(args.minimum_reportable_contribution_micro),
        "--max-excluded-cv-fraction",
        str(args.max_excluded_cv_fraction),
        "--max-unresolved-cv-fraction",
        str(args.max_unresolved_cv_fraction),
        "--max-unresolved-alpha-fraction",
        str(args.max_unresolved_alpha_fraction),
    ]
    if args.model is not None:
        command.extend(["--model", str(args.model)])
    if args.supercell is not None:
        command.extend(["--supercell", *[str(value) for value in args.supercell]])
    if args.batch_relax:
        command.append("--batch-relax")
    if args.resume:
        command.append("--resume")
    if args.force:
        command.append("--force")
    if args.skip_internal_relax:
        command.append("--skip-internal-relax")
    if args.skip_plots:
        command.append("--skip-plots")
    if args.qha_thermal_expansion is not None:
        command.extend(
            ["--qha-thermal-expansion", str(args.qha_thermal_expansion)]
        )
    command.extend(["--plot-dpi", str(args.plot_dpi)])
    return command


def run_stage(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"stage_failed:{process.returncode}:{log_path}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_production_plots(
    args: argparse.Namespace,
    material_dir: Path,
    result_dir: Path,
) -> dict[str, Any] | None:
    if args.skip_plots:
        return None
    try:
        return generate_result_plots(
            result_dir,
            material_dir=material_dir,
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
        print(f"[alpha-split-production] plotting warning: {report['failure']}")
        return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    material_dir = args.material_dir.expanduser().resolve()
    base_dir = material_dir / args.result_subdir
    primary_rel = str(
        Path(args.result_subdir) / f"primary_h{strain_tag(args.primary_strain)}"
    )
    fallback_rel = str(
        Path(args.result_subdir) / f"fallback_h{strain_tag(args.fallback_strain)}"
    )
    primary_dir = material_dir / primary_rel
    fallback_dir = material_dir / fallback_rel
    decision_path = base_dir / "production_decision.json"
    base_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "production_complete.json",
        "alpha_volume_split.dat",
        "alpha_volume_split_target.json",
        "alpha_volume_split_300K.json",
        "quality_report.json",
        "effective_strain_path.json",
        "preflight_report.json",
        "run_metadata.json",
        "strain_convergence.json",
        ALPHA_SPLIT_PNG,
        QHA_COMPARISON_PNG,
        PLOT_METADATA_JSON,
    ):
        published = base_dir / name
        if published.exists():
            previous = base_dir / f"{name}.previous"
            if previous.exists():
                previous.unlink()
            published.replace(previous)
    decision: dict[str, Any] = {
        "schema_version": 1,
        "status": "running_primary",
        "primary_result": str(primary_dir),
        "fallback_result": str(fallback_dir),
    }
    write_json(decision_path, decision)
    try:
        run_stage(
            runner_command(args, args.primary_strain, primary_rel),
            base_dir / "primary.log",
        )
        primary_reference = primary_dir / "work" / "strain_0"
        fallback_reference = fallback_dir / "work" / "strain_0"
        if primary_reference.is_dir() and not fallback_reference.exists() and not args.force:
            fallback_reference.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(primary_reference, fallback_reference)
            decision["reference_force_constants_reused"] = True
        else:
            decision["reference_force_constants_reused"] = False
        decision["status"] = "running_fallback"
        write_json(decision_path, decision)
        run_stage(
            runner_command(args, args.fallback_strain, fallback_rel),
            base_dir / "fallback.log",
        )
        primary_quality = read_json(primary_dir / "quality_report.json")
        fallback_quality = read_json(fallback_dir / "quality_report.json")
        convergence_check = compare_split_responses(
            load_alpha_split_table(primary_dir),
            load_alpha_split_table(fallback_dir),
            target_temperature_K=args.convergence_temperature,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance_micro_per_K=args.absolute_tolerance_micro,
            ratio_absolute_tolerance=args.ratio_absolute_tolerance,
        )
        if abs(args.target_temperature - args.convergence_temperature) < 1.0e-12:
            target_check = convergence_check
        else:
            target_check = compare_split_responses(
                load_alpha_split_table(primary_dir),
                load_alpha_split_table(fallback_dir),
                target_temperature_K=args.target_temperature,
                relative_tolerance=args.relative_tolerance,
                absolute_tolerance_micro_per_K=args.absolute_tolerance_micro,
                ratio_absolute_tolerance=args.ratio_absolute_tolerance,
            )
        comparison = {
            "status": (
                "converged"
                if convergence_check["status"] == "converged"
                and target_check["status"] == "converged"
                else "strain_derivative_unresolved"
            ),
            "convergence_temperature_check": convergence_check,
            "target_temperature_check": target_check,
        }
        write_json(base_dir / "strain_convergence.json", comparison)
        decision.update(
            {
                "primary_readiness": primary_quality["production_readiness"],
                "fallback_readiness": fallback_quality["production_readiness"],
                "strain_convergence": comparison,
            }
        )
        accepted = (
            comparison["status"] == "converged"
            and not fallback_quality["production_readiness"]["hard_failures"]
            and not fallback_quality["production_readiness"]["fallback_reasons"]
        )
        if accepted:
            decision["status"] = "ready"
            decision["selected_result"] = str(fallback_dir)
            for name in (
                "alpha_volume_split.dat",
                "alpha_volume_split_target.json",
                "quality_report.json",
                "effective_strain_path.json",
                "preflight_report.json",
                "run_metadata.json",
            ):
                shutil.copy2(fallback_dir / name, base_dir / name)
            target = read_json(base_dir / "alpha_volume_split_target.json")
            target["ratio_reportable"] = bool(
                target.get("ratio_abs_negative_to_positive") is not None
                and target.get("contribution_above_reporting_floor")
            )
            target["ratio_status_reason"] = (
                "strain_converged"
                if target["ratio_reportable"]
                else target.get("ratio_status_reason", "not_reportable")
            )
            write_json(base_dir / "alpha_volume_split_target.json", target)
            if abs(float(args.target_temperature) - 300.0) < 1.0e-8:
                write_json(base_dir / "alpha_volume_split_300K.json", target)
            plot_report = update_production_plots(args, material_dir, base_dir)
            fallback_complete = read_json(fallback_dir / "calculation_complete.json")
            write_json(
                base_dir / "production_complete.json",
                {
                    "status": "complete",
                    "selected_result": str(fallback_dir),
                    "selected_fingerprint_sha256": fallback_complete.get(
                        "fingerprint_sha256"
                    ),
                    "strain_convergence": str(base_dir / "strain_convergence.json"),
                    "plot_metadata": (
                        str(base_dir / PLOT_METADATA_JSON)
                        if plot_report is not None
                        else None
                    ),
                },
            )
        else:
            decision["status"] = "strain_derivative_unresolved"
            decision["selected_result"] = None
        write_json(decision_path, decision)
        if not accepted:
            raise SystemExit(2)
    except Exception as error:
        decision["status"] = "failed"
        decision["failure"] = f"{type(error).__name__}:{error}"
        write_json(decision_path, decision)
        raise


if __name__ == "__main__":
    main()
