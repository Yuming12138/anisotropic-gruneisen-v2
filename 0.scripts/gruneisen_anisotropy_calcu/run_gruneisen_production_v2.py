#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gruneisen_v2_core import (
    compare_mesh_responses,
    compare_strain_derivatives,
    runtime_versions,
    sha256_file,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_gruneisen_thermal_expansion_v2.py"
MESH_RUNNER = SCRIPT_DIR / "benchmark_mesh_convergence_v2.py"
COMPLETE_ARTIFACTS = (
    "quality_report.json",
    "run_metadata.json",
    "thermal_expansion_cartesian.dat",
    "thermal_expansion_directional.dat",
    "gruneisen_integrals.dat",
)
MESH_COMPLETE_ARTIFACTS = (
    "summary.json",
    "mesh_arrays.npz",
    "thermal_expansion_cartesian.dat",
    "thermal_expansion_directional.dat",
    "gruneisen_integrals.dat",
)


class StageFailure(RuntimeError):
    def __init__(self, command: list[str], returncode: int, result_subdir: Path) -> None:
        super().__init__(f"stage_failed:{result_subdir}:returncode={returncode}")
        self.command = command
        self.returncode = returncode
        self.result_subdir = result_subdir


def strain_tag(strain: float) -> str:
    value = np.format_float_positional(float(strain), trim="-")
    return value.replace("-", "m").replace(".", "p")


def runner_mesh(runner_args: list[str]) -> tuple[int, int, int]:
    if "--mesh" not in runner_args:
        return (20, 20, 20)
    index = runner_args.index("--mesh")
    try:
        return tuple(int(value) for value in runner_args[index + 1 : index + 4])
    except (TypeError, ValueError):
        raise SystemExit("invalid --mesh values") from None


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Production orchestrator with adaptive strain fallback"
    )
    parser.add_argument("--material-dir", type=Path, required=True)
    parser.add_argument("--result-subdir", default="gruneisen_aniso_1M_v2_prod")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--primary-strain", type=float, default=0.005)
    parser.add_argument("--fallback-strain", type=float, default=0.0025)
    parser.add_argument("--dense-mesh", type=int, nargs=3, default=(24, 24, 24))
    parser.add_argument("--batch-relax-atom-cap", type=int, default=1024)
    parser.add_argument("--disable-batch-relax", action="store_true")
    args, runner_args = parser.parse_known_args(argv)
    result_subdir = Path(args.result_subdir)
    if result_subdir.is_absolute() or ".." in result_subdir.parts:
        raise SystemExit("--result-subdir must be relative to the material directory")
    forbidden = {"--material-dir", "--result-subdir", "--strain", "--fallback-strain"}
    if any(value.split("=", 1)[0] in forbidden for value in runner_args):
        raise SystemExit("material, result, and strain options must use orchestrator arguments")
    if args.primary_strain <= 0.0 or args.fallback_strain <= 0.0:
        raise SystemExit("strain amplitudes must be positive")
    if args.fallback_strain >= args.primary_strain:
        raise SystemExit("--fallback-strain must be smaller than --primary-strain")
    if args.batch_relax_atom_cap <= 0:
        raise SystemExit("--batch-relax-atom-cap must be positive")
    base_mesh = runner_mesh(runner_args)
    if len(base_mesh) != 3 or any(value <= 0 for value in base_mesh):
        raise SystemExit("invalid screening mesh")
    if any(dense <= screening for dense, screening in zip(args.dense_mesh, base_mesh)):
        raise SystemExit("--dense-mesh must exceed the screening --mesh on every axis")
    return args, runner_args


def build_stage_command(
    python: Path,
    material_dir: Path,
    result_subdir: Path,
    strain: float,
    fallback_strain: float,
    batch_relax_atom_cap: int,
    disable_batch_relax: bool,
    runner_args: list[str],
) -> list[str]:
    command = [
        str(python.expanduser().resolve()),
        str(RUNNER),
        "--material-dir",
        str(material_dir),
        "--result-subdir",
        str(result_subdir),
        "--strain",
        str(strain),
        "--fallback-strain",
        str(fallback_strain),
        *runner_args,
    ]
    if not disable_batch_relax:
        command.extend(
            ["--batch-relax", "--batch-relax-atom-cap", str(batch_relax_atom_cap)]
        )
    return command


def build_mesh_command(
    python: Path,
    source_result: Path,
    dense_mesh: tuple[int, int, int],
    output_dir: Path,
) -> list[str]:
    return [
        str(python.expanduser().resolve()),
        str(MESH_RUNNER),
        "--source-result",
        str(source_result),
        "--mesh",
        *[str(value) for value in dense_mesh],
        "--output-dir",
        str(output_dir),
    ]


def run_stage(command: list[str], result_subdir: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        process = subprocess.run(
            command,
            cwd=str(SCRIPT_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise StageFailure(command, process.returncode, result_subdir)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_is_complete(
    result_dir: Path,
    current_command: list[str],
    previous_command: list[str] | None,
) -> bool:
    if previous_command != current_command:
        return False
    try:
        complete = load_json(result_dir / "calculation_complete.json")
        metadata = load_json(result_dir / "run_metadata.json")
    except (OSError, json.JSONDecodeError):
        return False
    fingerprint = metadata.get("fingerprint", {})
    fingerprint_sha256 = fingerprint.get("fingerprint_sha256")
    if complete.get("status") != "complete":
        return False
    if metadata.get("calculation_status") != "complete":
        return False
    if complete.get("fingerprint_sha256") != fingerprint_sha256:
        return False
    if fingerprint.get("execution", {}).get("runtime_versions") != runtime_versions():
        return False
    if not all((result_dir / name).is_file() for name in COMPLETE_ARTIFACTS):
        return False
    for item in fingerprint.get("files", {}).values():
        path_text = item.get("path")
        expected_sha256 = item.get("sha256")
        if not path_text or not expected_sha256:
            return False
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != expected_sha256:
            return False
    return True


def mesh_stage_is_complete(
    output_dir: Path,
    source_result: Path,
    current_command: list[str],
    previous_command: list[str] | None,
) -> bool:
    if previous_command != current_command:
        return False
    try:
        complete = load_json(output_dir / "calculation_complete.json")
        summary = load_json(output_dir / "summary.json")
        source_complete = load_json(source_result / "calculation_complete.json")
    except (OSError, json.JSONDecodeError):
        return False
    fingerprint = summary.get("fingerprint", {})
    if complete.get("status") != "complete":
        return False
    if complete.get("fingerprint_sha256") != fingerprint.get("fingerprint_sha256"):
        return False
    if fingerprint.get("source_result") != str(source_result):
        return False
    if (
        fingerprint.get("source_fingerprint_sha256")
        != source_complete.get("fingerprint_sha256")
    ):
        return False
    if fingerprint.get("runtime_versions") != runtime_versions():
        return False
    if not all((output_dir / name).is_file() for name in MESH_COMPLETE_ARTIFACTS):
        return False
    for item in fingerprint.get("files", {}).values():
        path_text = item.get("path")
        expected_sha256 = item.get("sha256")
        if not path_text or not expected_sha256:
            return False
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != expected_sha256:
            return False
    return True


def execute_or_resume_stage(
    command: list[str],
    result_subdir: Path,
    result_dir: Path,
    previous_command: list[str] | None,
    log_path: Path,
) -> None:
    resume = "--resume" in command and "--force" not in command
    if resume and stage_is_complete(result_dir, command, previous_command):
        log_path.write_text(
            "$ " + " ".join(command) + "\n\nresume_complete_without_subprocess\n",
            encoding="utf-8",
        )
        return
    run_stage(command, result_subdir, log_path)


def execute_or_resume_mesh_stage(
    command: list[str],
    result_subdir: Path,
    output_dir: Path,
    source_result: Path,
    previous_command: list[str] | None,
    resume: bool,
    log_path: Path,
) -> None:
    if resume and mesh_stage_is_complete(
        output_dir, source_result, command, previous_command
    ):
        log_path.write_text(
            "$ " + " ".join(command) + "\n\nresume_complete_without_subprocess\n",
            encoding="utf-8",
        )
        return
    run_stage(command, result_subdir, log_path)


def load_response_tables(result_dir: Path) -> dict[str, np.ndarray]:
    integrals = np.loadtxt(result_dir / "gruneisen_integrals.dat")
    directional = np.loadtxt(result_dir / "thermal_expansion_directional.dat")
    cartesian = np.loadtxt(result_dir / "thermal_expansion_cartesian.dat")
    if integrals.ndim != 2 or integrals.shape[1] != 7:
        raise RuntimeError(f"invalid_integral_table:{result_dir}")
    if directional.ndim != 2 or directional.shape != (len(integrals), 6):
        raise RuntimeError(f"invalid_directional_table:{result_dir}")
    if cartesian.ndim != 2 or cartesian.shape != (len(integrals), 8):
        raise RuntimeError(f"invalid_cartesian_table:{result_dir}")
    if not np.array_equal(integrals[:, 0], directional[:, 0]):
        raise RuntimeError(f"temperature_grid_mismatch:{result_dir}")
    if not np.array_equal(integrals[:, 0], cartesian[:, 0]):
        raise RuntimeError(f"temperature_grid_mismatch:{result_dir}")
    if not all(np.all(np.isfinite(table)) for table in (integrals, directional, cartesian)):
        raise RuntimeError(f"nonfinite_response_table:{result_dir}")
    return {
        "temperatures": integrals[:, 0],
        "integrals": integrals[:, 1:7],
        "alpha_directional": directional[:, 1:4] * 1.0e-6,
        "alpha_volume": directional[:, 4] * 1.0e-6,
        "fani": directional[:, 5],
    }


def write_decision(production_dir: Path, decision: dict[str, Any], echo: bool = True) -> None:
    write_json(production_dir / "production_decision.json", decision)
    if echo:
        print(json.dumps(decision, indent=2))


def failure_details(error: Exception) -> dict[str, Any]:
    details: dict[str, Any] = {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, StageFailure):
        details.update(
            {
                "command": error.command,
                "returncode": error.returncode,
                "result_subdir": str(error.result_subdir),
            }
        )
    return details


def main(argv: list[str] | None = None) -> None:
    args, runner_args = parse_args(argv)
    material_dir = args.material_dir.expanduser().resolve()
    production_dir = material_dir / args.result_subdir
    production_dir.mkdir(parents=True, exist_ok=True)
    decision_path = production_dir / "production_decision.json"
    try:
        previous_decision = load_json(decision_path)
    except (OSError, json.JSONDecodeError):
        previous_decision = {}

    primary_subdir = Path(args.result_subdir) / f"primary_h{strain_tag(args.primary_strain)}"
    fallback_subdir = Path(args.result_subdir) / f"fallback_h{strain_tag(args.fallback_strain)}"
    primary_dir = material_dir / primary_subdir
    fallback_dir = material_dir / fallback_subdir
    dense_mesh = tuple(int(value) for value in args.dense_mesh)
    mesh_tag = "x".join(str(value) for value in dense_mesh)
    mesh_name = f"mesh_check_h{strain_tag(args.fallback_strain)}_m{mesh_tag}"
    mesh_subdir = Path(args.result_subdir) / mesh_name
    mesh_dir = material_dir / mesh_subdir
    primary_command = build_stage_command(
        args.python,
        material_dir,
        primary_subdir,
        args.primary_strain,
        args.fallback_strain,
        args.batch_relax_atom_cap,
        args.disable_batch_relax,
        runner_args,
    )
    fallback_command = build_stage_command(
        args.python,
        material_dir,
        fallback_subdir,
        args.fallback_strain,
        args.fallback_strain,
        args.batch_relax_atom_cap,
        args.disable_batch_relax,
        runner_args,
    )
    mesh_command = build_mesh_command(args.python, fallback_dir, dense_mesh, mesh_dir)
    decision: dict[str, Any] = {
        "schema_version": 1,
        "status": "running_primary",
        "material_dir": str(material_dir),
        "primary_result": str(primary_dir),
        "fallback_result": None,
        "selected_result": None,
        "primary_command": primary_command,
        "fallback_command": None,
        "primary_readiness": None,
        "fallback_readiness": None,
        "strain_convergence": None,
        "mesh_check_result": None,
        "mesh_check_command": None,
        "mesh_convergence": None,
        "failure": None,
    }
    write_decision(production_dir, decision, echo=False)

    try:
        execute_or_resume_stage(
            primary_command,
            primary_subdir,
            primary_dir,
            previous_decision.get("primary_command"),
            production_dir / "primary.log",
        )
        primary_quality = load_json(primary_dir / "quality_report.json")
        primary_readiness = primary_quality["production_readiness"]
        primary_tables = load_response_tables(primary_dir)
        decision["primary_readiness"] = primary_readiness
    except Exception as error:
        decision["status"] = "failed_primary_stage"
        decision["failure"] = failure_details(error)
        write_decision(production_dir, decision)
        raise SystemExit(2) from None

    if primary_readiness["status"] == "failed":
        decision["status"] = "failed_primary_quality"
    elif not primary_readiness["fallback_required"]:
        decision["status"] = "ready"
        decision["selected_result"] = str(primary_dir)
    else:
        decision["status"] = "running_fallback"
        decision["fallback_result"] = str(fallback_dir)
        decision["fallback_command"] = fallback_command
        write_decision(production_dir, decision, echo=False)
        try:
            execute_or_resume_stage(
                fallback_command,
                fallback_subdir,
                fallback_dir,
                previous_decision.get("fallback_command"),
                production_dir / "fallback.log",
            )
            fallback_quality = load_json(fallback_dir / "quality_report.json")
            fallback_readiness = fallback_quality["production_readiness"]
            fallback_tables = load_response_tables(fallback_dir)
            comparison = compare_strain_derivatives(
                primary_tables["temperatures"],
                primary_tables["integrals"],
                fallback_tables["integrals"],
                primary_tables["alpha_volume"],
                fallback_tables["alpha_volume"],
                primary_tables["alpha_directional"],
                fallback_tables["alpha_directional"],
            )
            decision.update(
                {
                    "fallback_readiness": fallback_readiness,
                    "strain_convergence": comparison,
                }
            )
        except Exception as error:
            decision["status"] = "failed_fallback_stage"
            decision["failure"] = failure_details(error)
            write_decision(production_dir, decision)
            raise SystemExit(2) from None
        if fallback_readiness["hard_failures"]:
            decision["status"] = "failed_fallback_quality"
        elif comparison["status"] != "converged":
            decision["status"] = "strain_derivative_unresolved"
        else:
            decision["status"] = "running_mesh_check"
            decision["mesh_check_result"] = str(mesh_dir)
            decision["mesh_check_command"] = mesh_command
            write_decision(production_dir, decision, echo=False)
            try:
                execute_or_resume_mesh_stage(
                    mesh_command,
                    mesh_subdir,
                    mesh_dir,
                    fallback_dir,
                    previous_decision.get("mesh_check_command"),
                    "--resume" in runner_args and "--force" not in runner_args,
                    production_dir / "mesh_check.log",
                )
                dense_tables = load_response_tables(mesh_dir)
                mesh_comparison = compare_mesh_responses(
                    fallback_tables["temperatures"],
                    fallback_tables["integrals"],
                    dense_tables["integrals"],
                    fallback_tables["alpha_volume"],
                    dense_tables["alpha_volume"],
                    fallback_tables["alpha_directional"],
                    dense_tables["alpha_directional"],
                    fallback_tables["fani"],
                    dense_tables["fani"],
                )
                decision["mesh_convergence"] = mesh_comparison
            except Exception as error:
                decision["status"] = "failed_mesh_check_stage"
                decision["failure"] = failure_details(error)
                write_decision(production_dir, decision)
                raise SystemExit(2) from None
            if mesh_comparison["status"] != "converged":
                decision["status"] = "mesh_convergence_unresolved"
            else:
                decision["status"] = "ready_with_fallback"
                decision["selected_result"] = str(fallback_dir)

    write_decision(production_dir, decision)
    if not str(decision["status"]).startswith("ready"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
