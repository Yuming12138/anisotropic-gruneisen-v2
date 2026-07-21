#!/usr/bin/env python
"""Batch dispatcher and CSV collector for compliance-weighted alpha splitting."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RUNNER = SCRIPT_DIR / "run_compliance_weighted_alpha_split.py"
PRODUCTION_RUNNER = SCRIPT_DIR / "run_alpha_split_production.py"
DEFAULT_ROOTS = (PROJECT_ROOT / "NTE_materials", PROJECT_ROOT / "PTE_materials")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument("--materials", nargs="+", default=None)
    parser.add_argument("--materials-file", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--result-subdir", default="gruneisen_alpha_split_1M_v1")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--production", action="store_true")
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
    parser.add_argument("--chunk-count", type=int, default=None)
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-root", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=SCRIPT_DIR / "batch_logs")
    args = parser.parse_args(argv)
    if args.chunk_count is not None:
        if args.chunk_count <= 0 or args.chunk_index is None:
            raise SystemExit("--chunk-count requires a positive value and --chunk-index")
        if args.chunk_index < 0 or args.chunk_index >= args.chunk_count:
            raise SystemExit("--chunk-index must satisfy 0 <= index < chunk-count")
    elif args.chunk_index is not None:
        raise SystemExit("--chunk-index requires --chunk-count")
    return args


def selected_names(args: argparse.Namespace) -> list[str]:
    names = list(args.materials or [])
    if args.materials_file is not None:
        for line in args.materials_file.read_text(encoding="utf-8-sig").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                names.append(value)
    return names


def discover_materials(args: argparse.Namespace) -> list[Path]:
    patterns = [value.lower() for value in selected_names(args)]
    by_root: list[list[Path]] = []
    for root in args.roots:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            print(f"warning: missing root skipped: {resolved}", file=sys.stderr)
            continue
        materials = [
            path
            for path in sorted(resolved.iterdir(), key=lambda value: value.name.lower())
            if path.is_dir()
            and (
                not patterns
                or any(pattern == path.name.lower() or pattern in path.name.lower() for pattern in patterns)
            )
        ]
        if args.limit_per_root is not None:
            materials = materials[: args.limit_per_root]
        by_root.append(materials)
    interleaved: list[Path] = []
    for group in zip_longest(*by_root):
        interleaved.extend(path for path in group if path is not None)
    if args.chunk_count is not None:
        interleaved = [
            path
            for index, path in enumerate(interleaved)
            if index % args.chunk_count == args.chunk_index
        ]
    if args.limit is not None:
        interleaved = interleaved[: args.limit]
    return interleaved


def runner_command(args: argparse.Namespace, material_dir: Path) -> list[str]:
    production = args.production and not args.preflight_only
    runner = PRODUCTION_RUNNER if production else RUNNER
    command = [
        str(args.python),
        str(runner),
        "--material-dir",
        str(material_dir),
        "--result-subdir",
        args.result_subdir,
        "--model-size",
        args.model_size,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
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
    if production:
        command.extend(
            [
                "--python",
                str(args.python),
                "--primary-strain",
                str(args.strain),
                "--fallback-strain",
                str(args.fallback_strain),
                "--convergence-temperature",
                str(args.convergence_temperature),
                "--relative-tolerance",
                str(args.relative_tolerance),
                "--absolute-tolerance-micro",
                str(args.absolute_tolerance_micro),
                "--ratio-absolute-tolerance",
                str(args.ratio_absolute_tolerance),
            ]
        )
    else:
        command.extend(
            [
                "--strain",
                str(args.strain),
                "--fallback-strain",
                str(args.fallback_strain),
            ]
        )
        if args.preflight_only:
            command.append("--preflight-only")
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
    return command


def read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def result_row(
    material_dir: Path, result_subdir: str, returncode: int, log_path: Path
) -> dict[str, Any]:
    result_dir = material_dir / result_subdir
    preflight = read_json_if_present(result_dir / "preflight_report.json")
    quality = read_json_if_present(result_dir / "quality_report.json")
    target = read_json_if_present(result_dir / "alpha_volume_split_target.json")
    decision = read_json_if_present(result_dir / "production_decision.json")
    path = read_json_if_present(result_dir / "effective_strain_path.json")
    readiness = quality.get("production_readiness", {})
    return {
        "root": str(material_dir.parent),
        "material": material_dir.name,
        "returncode": returncode,
        "status": decision.get("status") or readiness.get("status") or preflight.get("status"),
        "result_dir": str(result_dir),
        "log": str(log_path),
        "preflight_issues": ";".join(preflight.get("issues", [])),
        "phase_consistency": preflight.get("phase_consistency", {}).get("status"),
        "path_scale_1_per_GPa": path.get("path_scale_1_per_GPa"),
        "beta_1_per_GPa": path.get("volumetric_compliance_beta_1_per_GPa"),
        "alphaV_plus_300K_micro": target.get("alphaV_positive_micro_per_K"),
        "alphaV_minus_300K_micro": target.get("alphaV_negative_micro_per_K"),
        "alphaV_total_300K_micro": target.get("alphaV_total_micro_per_K"),
        "ratio_300K": target.get("ratio_abs_negative_to_positive"),
        "ratio_reportable": target.get("ratio_reportable"),
        "unresolved_Cv_fraction_300K": target.get("unresolved_heat_capacity_fraction"),
        "max_excluded_Cv_fraction": quality.get("max_excluded_heat_capacity_fraction"),
        "max_unresolved_Cv_fraction": quality.get("max_unresolved_heat_capacity_fraction"),
        "max_unresolved_alpha_fraction": quality.get("max_unresolved_alpha_fraction"),
        "split_identity_error_per_K": quality.get("split_identity_max_abs_error_per_K"),
        "fingerprint": preflight.get("fingerprint", {}).get("fingerprint_sha256"),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    materials = discover_materials(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = args.log_dir.expanduser().resolve() / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for material in materials:
        safe = f"{material.parent.name}__{material.name}".replace(" ", "_")
        log_path = log_dir / f"{safe}.log"
        command = runner_command(args, material)
        print(" ".join(command), flush=True)
        if args.dry_run:
            returncode = 0
            log_path.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
        else:
            with log_path.open("w", encoding="utf-8") as handle:
                process = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, text=True
                )
            returncode = process.returncode
        rows.append(result_row(material, args.result_subdir, returncode, log_path))
        write_summary(log_dir / "batch_summary.csv", rows)
        if returncode != 0 and args.stop_on_error:
            break
    write_summary(log_dir / "batch_summary.csv", rows)
    failures = sum(1 for row in rows if row["returncode"] != 0)
    print(f"batch complete: total={len(rows)} failures={failures} summary={log_dir / 'batch_summary.csv'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
