#!/usr/bin/env python
"""Batch dispatcher and audit collector for anisotropic Gruneisen v2."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RUNNER = SCRIPT_DIR / "run_gruneisen_thermal_expansion_v2.py"
DEFAULT_ROOTS = (PROJECT_ROOT / "NTE_materials", PROJECT_ROOT / "PTE_materials")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument("--materials", nargs="+", default=None)
    parser.add_argument("--materials-file", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--result-subdir", default="gruneisen_aniso_1M_v2")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-size", choices=("1M", "5M"), default="1M")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--strain", type=float, default=0.005)
    parser.add_argument("--fallback-strain", type=float, default=0.0025)
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--mesh", type=int, nargs=3, default=(30, 30, 30))
    parser.add_argument("--min-supercell-length", type=float, default=12.0)
    parser.add_argument("--supercell", type=int, nargs=3, default=None)
    parser.add_argument("--fmax", type=float, default=1.0e-3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--frequency-cutoff", type=float, default=1.0e-4)
    parser.add_argument("--tmin", type=float, default=10.0)
    parser.add_argument("--tmax", type=float, default=1000.0)
    parser.add_argument("--tstep", type=float, default=10.0)
    parser.add_argument("--fani-threshold", type=float, default=0.20)
    parser.add_argument("--sign-tolerance-micro", type=float, default=1.0e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-internal-relax", action="store_true")
    parser.add_argument("--chunk-count", type=int, default=None)
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-root", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=SCRIPT_DIR / "batch_logs_v2")
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


def material_matches(path: Path, patterns: list[str]) -> bool:
    if not patterns:
        return True
    name = path.name.lower()
    return any(pattern.lower() == name or pattern.lower() in name for pattern in patterns)


def discover_materials(args: argparse.Namespace) -> list[Path]:
    patterns = selected_names(args)
    by_root: list[list[Path]] = []
    for root in args.roots:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            print(f"warning: missing root skipped: {resolved}", file=sys.stderr)
            continue
        materials = [
            path
            for path in sorted(resolved.iterdir(), key=lambda value: value.name.lower())
            if path.is_dir() and material_matches(path, patterns)
        ]
        if args.limit_per_root is not None:
            materials = materials[: args.limit_per_root]
        by_root.append(materials)

    # Interleave roots so a small representative run does not silently contain
    # only the first root (historically NTE).
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


def safe_name(material_dir: Path) -> str:
    return f"{material_dir.parent.name}__{material_dir.name}".replace(" ", "_")


def runner_command(args: argparse.Namespace, material_dir: Path) -> list[str | Path]:
    command: list[str | Path] = [
        args.python,
        RUNNER,
        "--material-dir",
        material_dir,
        "--result-subdir",
        args.result_subdir,
        "--model-size",
        args.model_size,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--strain",
        str(args.strain),
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
        "--frequency-cutoff",
        str(args.frequency_cutoff),
        "--tmin",
        str(args.tmin),
        "--tmax",
        str(args.tmax),
        "--tstep",
        str(args.tstep),
        "--fani-threshold",
        str(args.fani_threshold),
        "--sign-tolerance-micro",
        str(args.sign_tolerance_micro),
    ]
    if args.preflight_only:
        command.append("--preflight-only")
    if args.model is not None:
        command.extend(["--model", args.model])
    if args.supercell is not None:
        command.extend(["--supercell", *[str(value) for value in args.supercell]])
    if args.resume:
        command.append("--resume")
    if args.force:
        command.append("--force")
    if args.skip_internal_relax:
        command.append("--skip-internal-relax")
    return command


def run_command(command: list[str | Path], cwd: Path, log_path: Path, dry_run: bool) -> tuple[int, float]:
    if dry_run:
        print("DRY-RUN:", " ".join(str(value) for value in command))
        return 0, 0.0
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(value) for value in command) + "\n\n")
        process = subprocess.run(
            [str(value) for value in command],
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process.returncode, time.perf_counter() - start


def run_preflight_direct(
    args: argparse.Namespace,
    material_dir: Path,
    log_path: Path,
) -> tuple[int, float]:
    """Run preflight in-process to avoid hundreds of Python import startups."""

    from run_gruneisen_thermal_expansion_v2 import preflight, write_preflight_outputs

    start = time.perf_counter()
    runner_args = argparse.Namespace(**vars(args))
    runner_args.material_dir = material_dir
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report, context = preflight(runner_args)
        write_preflight_outputs(report, context)
        log_path.write_text(
            json.dumps(
                {
                    "material": material_dir.name,
                    "status": report["status"],
                    "issues": report["issues"],
                    "blocking_issues": report["blocking_issues"],
                    "result_dir": str(context["result_dir"]),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0, time.perf_counter() - start
    except Exception:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        return 1, time.perf_counter() - start


def read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def result_row(material_dir: Path, result_subdir: str) -> dict[str, Any]:
    result_dir = material_dir / result_subdir
    preflight = read_json_if_present(result_dir / "preflight_report.json")
    quality = read_json_if_present(result_dir / "quality_report.json")
    complete = read_json_if_present(result_dir / "calculation_complete.json")
    elastic = preflight.get("elastic", {})
    return {
        "root": material_dir.parent.name,
        "material": material_dir.name,
        "material_dir": str(material_dir),
        "result_dir": str(result_dir),
        "preflight_status": preflight.get("status", "missing"),
        "calc_status": complete.get("status", "not_complete"),
        "quality_status": "available" if quality else "not_available",
        "elastic_positive_definite": elastic.get("positive_definite", ""),
        "elastic_min_eigenvalue_GPa": elastic.get("min_eigenvalue_GPa", ""),
        "elastic_condition_number": elastic.get("condition_number", ""),
        "axis_mapping_status": preflight.get("axis_mapping", {}).get("status", ""),
        "reference_imaginary_count": quality.get("reference_imaginary_or_zero_count", ""),
        "strained_imaginary_count": (
            sum(
                int(item.get("minus_imaginary_mode_count", 0))
                + int(item.get("plus_imaginary_mode_count", 0))
                for item in quality.get("strained_imaginary_diagnostics", [])
            )
            if quality.get("strained_imaginary_diagnostics")
            else ""
        ),
        "strain_converged": quality.get("strain_convergence_status", ""),
        "effective_isotropy_status": quality.get("effective_isotropy_screen", {}).get("status", ""),
        "Fani_max": quality.get("effective_isotropy_screen", {}).get("fani_max", ""),
        "fingerprint_sha256": preflight.get("fingerprint", {}).get("fingerprint_sha256", ""),
        "issues": ";".join(preflight.get("issues", [])),
        "metadata_path": str(result_dir / "run_metadata.json"),
        "quality_report_path": str(result_dir / "quality_report.json"),
        "log_path": "",
        "seconds": "",
        "runner_returncode": "",
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    materials = discover_materials(args)
    if not materials:
        raise SystemExit("No material folders selected")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = args.log_dir.expanduser().resolve() / stamp
    summary_path = log_dir / "batch_summary.csv"
    rows: list[dict[str, Any]] = []
    print(f"selected materials: {len(materials)}")
    print(f"log_dir: {log_dir}")
    for index, material_dir in enumerate(materials, start=1):
        print(f"[{index}/{len(materials)}] {material_dir.parent.name}/{material_dir.name}")
        log_path = log_dir / f"{safe_name(material_dir)}.log"
        if args.preflight_only and not args.dry_run:
            code, seconds = run_preflight_direct(args, material_dir, log_path)
        else:
            code, seconds = run_command(
                runner_command(args, material_dir),
                PROJECT_ROOT,
                log_path,
                args.dry_run,
            )
        row = result_row(material_dir, args.result_subdir)
        row["log_path"] = str(log_path)
        row["seconds"] = f"{seconds:.2f}"
        row["runner_returncode"] = code
        if args.dry_run:
            row["preflight_status"] = "dry_run"
            row["calc_status"] = "dry_run"
        rows.append(row)
        write_summary(summary_path, rows)
        print(
            f"  returncode={code} preflight={row['preflight_status']} calc={row['calc_status']}"
        )
        if code != 0 and args.stop_on_error:
            raise SystemExit(f"Stopped after failure; summary: {summary_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
