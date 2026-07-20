#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read


STATE_NAMES = tuple(
    f"eta{component}_{sign}"
    for component in range(1, 7)
    for sign in ("minus", "plus")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare serial and BatchRelaxer anisotropic-v2 production results"
    )
    parser.add_argument("--batch-decision", type=Path, required=True)
    parser.add_argument("--serial-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--integral-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--alpha-cartesian-tolerance-micro", type=float, default=1.0)
    parser.add_argument("--alpha-directional-tolerance-micro", type=float, default=1.0)
    parser.add_argument("--alpha-volume-tolerance-micro", type=float, default=0.5)
    parser.add_argument("--fani-tolerance", type=float, default=0.01)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def maximum_position_difference(left_path: Path, right_path: Path) -> float:
    left = read(left_path)
    right = read(right_path)
    if left.get_chemical_symbols() != right.get_chemical_symbols():
        raise ValueError(f"symbol_order_mismatch:{left_path}:{right_path}")
    if not np.allclose(left.cell.array, right.cell.array, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"cell_mismatch:{left_path}:{right_path}")
    fractional = right.get_scaled_positions(wrap=False) - left.get_scaled_positions(wrap=False)
    fractional -= np.rint(fractional)
    cartesian = fractional @ left.cell.array
    return float(np.max(np.linalg.norm(cartesian, axis=1)))


def compare_relaxations(batch_dir: Path, serial_dir: Path) -> dict[str, Any]:
    positions = []
    energy_differences = []
    force_differences = []
    for state_name in STATE_NAMES:
        batch_state = batch_dir / "work" / state_name
        serial_state = serial_dir / "work" / state_name
        positions.append(
            maximum_position_difference(batch_state / "CONTCAR", serial_state / "CONTCAR")
        )
        batch_report = load_json(batch_state / "internal_relax_report.json")
        serial_report = load_json(serial_state / "internal_relax_report.json")
        energy_differences.append(
            abs(float(batch_report["energy_eV"]) - float(serial_report["energy_eV"]))
        )
        force_differences.append(
            abs(
                float(batch_report["max_force_eV_A"])
                - float(serial_report["max_force_eV_A"])
            )
        )
    return {
        "max_position_difference_A": max(positions),
        "max_energy_difference_eV": max(energy_differences),
        "max_final_force_difference_eV_A": max(force_differences),
    }


def compare_stage(batch_dir: Path, serial_dir: Path) -> dict[str, Any]:
    batch_cartesian = np.loadtxt(batch_dir / "thermal_expansion_cartesian.dat")
    serial_cartesian = np.loadtxt(serial_dir / "thermal_expansion_cartesian.dat")
    batch_directional = np.loadtxt(batch_dir / "thermal_expansion_directional.dat")
    serial_directional = np.loadtxt(serial_dir / "thermal_expansion_directional.dat")
    batch_integrals = np.loadtxt(batch_dir / "gruneisen_integrals.dat")
    serial_integrals = np.loadtxt(serial_dir / "gruneisen_integrals.dat")
    for left, right, label in (
        (batch_cartesian, serial_cartesian, "cartesian"),
        (batch_directional, serial_directional, "directional"),
        (batch_integrals, serial_integrals, "integrals"),
    ):
        if left.shape != right.shape or not np.array_equal(left[:, 0], right[:, 0]):
            raise ValueError(f"temperature_grid_mismatch:{label}")
    integral_scale = max(
        float(np.max(np.abs(batch_integrals[:, 1:]))),
        float(np.max(np.abs(serial_integrals[:, 1:]))),
        1.0e-25,
    )
    return {
        "batch_result": str(batch_dir),
        "serial_result": str(serial_dir),
        "relaxation": compare_relaxations(batch_dir, serial_dir),
        "integral_relative_difference": float(
            np.max(np.abs(batch_integrals[:, 1:] - serial_integrals[:, 1:]))
            / integral_scale
        ),
        "alpha_cartesian_max_difference_micro_per_K": float(
            np.max(np.abs(batch_cartesian[:, 1:7] - serial_cartesian[:, 1:7]))
        ),
        "alpha_directional_max_difference_micro_per_K": float(
            np.max(np.abs(batch_directional[:, 1:4] - serial_directional[:, 1:4]))
        ),
        "alpha_volume_max_difference_micro_per_K": float(
            np.max(np.abs(batch_directional[:, 4] - serial_directional[:, 4]))
        ),
        "fani_max_difference": float(
            np.max(np.abs(batch_directional[:, 5] - serial_directional[:, 5]))
        ),
    }


def main() -> None:
    args = parse_args()
    batch_decision = load_json(args.batch_decision)
    serial_decision = load_json(args.serial_decision)
    stages = {}
    for stage in ("primary", "fallback"):
        batch_path = batch_decision.get(f"{stage}_result")
        serial_path = serial_decision.get(f"{stage}_result")
        if batch_path and serial_path and Path(batch_path).is_dir() and Path(serial_path).is_dir():
            stages[stage] = compare_stage(Path(batch_path), Path(serial_path))

    checks = {
        "decision_status_match": batch_decision.get("status") == serial_decision.get("status"),
        "stage_set_match": bool(stages)
        and set(stages)
        == {
            stage
            for stage in ("primary", "fallback")
            if batch_decision.get(f"{stage}_result") and serial_decision.get(f"{stage}_result")
        },
        "position_difference": all(
            result["relaxation"]["max_position_difference_A"] <= args.position_tolerance
            for result in stages.values()
        ),
        "integral_difference": all(
            result["integral_relative_difference"] <= args.integral_relative_tolerance
            for result in stages.values()
        ),
        "alpha_cartesian_difference": all(
            result["alpha_cartesian_max_difference_micro_per_K"]
            <= args.alpha_cartesian_tolerance_micro
            for result in stages.values()
        ),
        "alpha_directional_difference": all(
            result["alpha_directional_max_difference_micro_per_K"]
            <= args.alpha_directional_tolerance_micro
            for result in stages.values()
        ),
        "alpha_volume_difference": all(
            result["alpha_volume_max_difference_micro_per_K"]
            <= args.alpha_volume_tolerance_micro
            for result in stages.values()
        ),
        "fani_difference": all(
            result["fani_max_difference"] <= args.fani_tolerance for result in stages.values()
        ),
    }
    report = {
        "schema_version": 1,
        "batch_decision": str(args.batch_decision.expanduser().resolve()),
        "serial_decision": str(args.serial_decision.expanduser().resolve()),
        "batch_status": batch_decision.get("status"),
        "serial_status": serial_decision.get("status"),
        "thresholds": {
            "position_tolerance_A": args.position_tolerance,
            "integral_relative_tolerance": args.integral_relative_tolerance,
            "alpha_cartesian_tolerance_micro_per_K": args.alpha_cartesian_tolerance_micro,
            "alpha_directional_tolerance_micro_per_K": args.alpha_directional_tolerance_micro,
            "alpha_volume_tolerance_micro_per_K": args.alpha_volume_tolerance_micro,
            "fani_tolerance": args.fani_tolerance,
        },
        "stages": stages,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
