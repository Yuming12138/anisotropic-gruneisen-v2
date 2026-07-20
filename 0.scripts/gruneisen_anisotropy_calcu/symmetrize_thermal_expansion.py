#!/usr/bin/env python3
"""Project calculated thermal-expansion tensors onto crystal point-group symmetry.

This is a non-destructive post-processing step.  The original Cartesian table is
preserved as ``thermal_expansion_cartesian_raw.dat`` and the projected tensor is
written to separate ``*_symmetrized`` files.  A residual report makes numerical
symmetry breaking visible instead of silently hiding it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import spglib
from pymatgen.core import Structure


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
VOIGT_OUTPUT_HEADERS = (
    "alpha_xx",
    "alpha_yy",
    "alpha_zz",
    "alpha_yz_eng",
    "alpha_xz_eng",
    "alpha_xy_eng",
)


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
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rows_to_text_table(headers: Iterable[str], rows: np.ndarray) -> str:
    lines = ["# " + " ".join(headers)]
    lines.extend(" ".join(f"{float(value):.10e}" for value in row) for row in rows)
    return "\n".join(lines) + "\n"


def engineering_voigt_to_tensor(values: np.ndarray) -> np.ndarray:
    """Convert ``xx yy zz yz xz xy`` engineering components to 3x3 tensors."""

    voigt = np.asarray(values, dtype=float)
    if voigt.shape[-1] != 6:
        raise ValueError("engineering_voigt_last_dimension_must_be_6")
    tensor = np.zeros(voigt.shape[:-1] + (3, 3), dtype=float)
    tensor[..., 0, 0] = voigt[..., 0]
    tensor[..., 1, 1] = voigt[..., 1]
    tensor[..., 2, 2] = voigt[..., 2]
    tensor[..., 1, 2] = tensor[..., 2, 1] = voigt[..., 3] / 2.0
    tensor[..., 0, 2] = tensor[..., 2, 0] = voigt[..., 4] / 2.0
    tensor[..., 0, 1] = tensor[..., 1, 0] = voigt[..., 5] / 2.0
    return tensor


def tensor_to_engineering_voigt(tensors: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensors, dtype=float)
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("tensor_last_dimensions_must_be_3x3")
    return np.stack(
        [
            tensor[..., 0, 0],
            tensor[..., 1, 1],
            tensor[..., 2, 2],
            2.0 * tensor[..., 1, 2],
            2.0 * tensor[..., 0, 2],
            2.0 * tensor[..., 0, 1],
        ],
        axis=-1,
    )


def _dataset_value(dataset: Any, name: str) -> Any:
    if hasattr(dataset, name):
        return getattr(dataset, name)
    return dataset[name]


def _spglib_cell(structure: Structure) -> tuple[np.ndarray, np.ndarray, list[int]]:
    try:
        numbers = [int(site.specie.Z) for site in structure]
    except AttributeError as exc:
        raise ValueError("disordered_structures_are_not_supported") from exc
    return (
        np.asarray(structure.lattice.matrix, dtype=float),
        np.asarray(structure.frac_coords, dtype=float),
        numbers,
    )


def _nearest_orthogonal(matrix: np.ndarray) -> np.ndarray:
    left, _singular_values, right = np.linalg.svd(np.asarray(matrix, dtype=float))
    return left @ right


def cartesian_point_group(
    structure: Structure,
    symprec_A: float,
    angle_tolerance_deg: float = -1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return unique orthogonal Cartesian point operations recognized by spglib."""

    cell = _spglib_cell(structure)
    dataset = spglib.get_symmetry_dataset(
        cell,
        symprec=float(symprec_A),
        angle_tolerance=float(angle_tolerance_deg),
    )
    symmetry = spglib.get_symmetry(
        cell,
        symprec=float(symprec_A),
        angle_tolerance=float(angle_tolerance_deg),
    )
    if dataset is None or symmetry is None:
        raise RuntimeError("spglib_symmetry_identification_failed")

    lattice_rows = np.asarray(structure.lattice.matrix, dtype=float)
    lattice_columns = lattice_rows.T
    lattice_columns_inverse = np.linalg.inv(lattice_columns)
    operations: list[np.ndarray] = []
    preprojection_orthogonality = []
    for fractional_rotation in np.asarray(symmetry["rotations"], dtype=float):
        cartesian = lattice_columns @ fractional_rotation @ lattice_columns_inverse
        preprojection_orthogonality.append(
            float(np.linalg.norm(cartesian.T @ cartesian - np.eye(3)))
        )
        cartesian = _nearest_orthogonal(cartesian)
        if not any(np.allclose(cartesian, existing, atol=1.0e-7) for existing in operations):
            operations.append(cartesian)

    if not operations:
        raise RuntimeError("empty_cartesian_point_group")
    operation_array = np.asarray(operations, dtype=float)
    report = {
        "symprec_A": float(symprec_A),
        "angle_tolerance_deg": float(angle_tolerance_deg),
        "spacegroup_number": int(_dataset_value(dataset, "number")),
        "spacegroup_symbol": str(_dataset_value(dataset, "international")),
        "hall_number": int(_dataset_value(dataset, "hall_number")),
        "spacegroup_operation_count": int(len(symmetry["rotations"])),
        "point_group_operation_count": int(len(operation_array)),
        "max_preprojection_orthogonality_error": float(max(preprojection_orthogonality)),
        "max_final_orthogonality_error": float(
            max(np.linalg.norm(op.T @ op - np.eye(3)) for op in operation_array)
        ),
    }
    return operation_array, report


def select_point_group(
    structure: Structure,
    symprec_A: float,
    strict_symprec_A: float,
    angle_tolerance_deg: float = -1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    regular_operations, regular_report = cartesian_point_group(
        structure, symprec_A, angle_tolerance_deg
    )
    strict_operations, strict_report = cartesian_point_group(
        structure, strict_symprec_A, angle_tolerance_deg
    )
    stable = (
        regular_report["spacegroup_number"] == strict_report["spacegroup_number"]
        and regular_report["point_group_operation_count"]
        == strict_report["point_group_operation_count"]
    )
    # If symmetry recognition is tolerance-sensitive, use the stricter and
    # therefore less aggressive group so that the postprocessor cannot invent
    # forbidden tensor constraints from a loose tolerance.
    selected = regular_operations if stable else strict_operations
    return selected, {
        "status": "stable" if stable else "tolerance_sensitive_using_strict_group",
        "requested": regular_report,
        "strict_audit": strict_report,
        "selected_symprec_A": float(symprec_A if stable else strict_symprec_A),
        "selected_point_group_operation_count": int(len(selected)),
    }


def symmetrize_tensors(tensors: np.ndarray, operations: np.ndarray) -> np.ndarray:
    raw = np.asarray(tensors, dtype=float)
    point_group = np.asarray(operations, dtype=float)
    if raw.shape[-2:] != (3, 3):
        raise ValueError("tensor_last_dimensions_must_be_3x3")
    if point_group.ndim != 3 or point_group.shape[1:] != (3, 3):
        raise ValueError("point_group_shape_must_be_nx3x3")
    projected = np.zeros_like(raw)
    for rotation in point_group:
        projected += np.einsum("ij,...jk,lk->...il", rotation, raw, rotation)
    projected /= float(len(point_group))
    return 0.5 * (projected + np.swapaxes(projected, -1, -2))


def _read_table(path: Path, expected_columns: int) -> np.ndarray:
    table = np.loadtxt(path, comments="#", ndmin=2)
    if table.ndim != 2 or table.shape[1] != expected_columns:
        raise ValueError(f"invalid_table_shape:{path}:{table.shape}")
    if not np.all(np.isfinite(table)):
        raise ValueError(f"nonfinite_table:{path}")
    return np.asarray(table, dtype=float)


def _directional_projection(
    tensors: np.ndarray, mapping_path: Path
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not mapping_path.is_file():
        return None, {"status": "structure_mapping_missing"}
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    vectors = mapping.get("axis_unit_vectors_in_elastic_frame")
    if mapping.get("status") != "ok" or not isinstance(vectors, dict):
        return None, {"status": mapping.get("status", "axis_mapping_unavailable")}
    directions = []
    for axis in ("a", "b", "c"):
        direction = np.asarray(vectors[axis], dtype=float)
        direction /= np.linalg.norm(direction)
        directions.append(direction)
    values = np.stack(
        [np.einsum("i,...ij,j->...", direction, tensors, direction) for direction in directions],
        axis=-1,
    )
    return values, {"status": "ok", "mapping_path": str(mapping_path)}


def process_result_dir(
    result_dir: Path,
    symprec_A: float = 5.0e-3,
    strict_symprec_A: float = 1.0e-3,
    angle_tolerance_deg: float = -1.0,
    absolute_tolerance_micro_per_K: float = 5.0e-2,
    relative_tolerance: float = 2.0e-2,
) -> dict[str, Any]:
    result_dir = Path(result_dir).expanduser().resolve()
    cartesian_path = result_dir / "thermal_expansion_cartesian.dat"
    raw_path = result_dir / "thermal_expansion_cartesian_raw.dat"
    structure_path = result_dir / "reference" / "POSCAR"
    mapping_path = result_dir / "reference" / "structure_mapping.json"
    if not cartesian_path.is_file():
        raise FileNotFoundError(f"thermal_expansion_cartesian_missing:{cartesian_path}")
    if not structure_path.is_file():
        raise FileNotFoundError(f"reference_poscar_missing:{structure_path}")

    source_path = raw_path if raw_path.is_file() else cartesian_path
    table = _read_table(source_path, expected_columns=8)
    temperatures = table[:, 0]
    raw_voigt = table[:, 1:7]
    raw_tensors = engineering_voigt_to_tensor(raw_voigt)
    structure = Structure.from_file(structure_path)
    operations, symmetry_report = select_point_group(
        structure,
        symprec_A=symprec_A,
        strict_symprec_A=strict_symprec_A,
        angle_tolerance_deg=angle_tolerance_deg,
    )
    projected_tensors = symmetrize_tensors(raw_tensors, operations)
    projected_voigt = tensor_to_engineering_voigt(projected_tensors)
    projected_volume = np.trace(projected_tensors, axis1=-2, axis2=-1)

    delta_tensors = raw_tensors - projected_tensors
    delta_voigt = raw_voigt - projected_voigt
    residual_frobenius = np.linalg.norm(delta_tensors, axis=(-2, -1))
    residual_max_component = np.max(np.abs(delta_voigt), axis=1)
    raw_max_component = np.max(np.abs(raw_voigt), axis=1)
    residual_relative = residual_max_component / np.maximum(raw_max_component, 1.0e-15)
    allowed_residual = np.maximum(
        float(absolute_tolerance_micro_per_K),
        float(relative_tolerance) * raw_max_component,
    )
    violations = residual_max_component > allowed_residual

    if not raw_path.is_file():
        shutil.copy2(cartesian_path, raw_path)
    symmetrized_cartesian_path = result_dir / "thermal_expansion_cartesian_symmetrized.dat"
    symmetrized_cartesian_path.write_text(
        rows_to_text_table(
            ["T_K", *VOIGT_OUTPUT_HEADERS, "alpha_volume"],
            np.column_stack([temperatures, projected_voigt, projected_volume]),
        ),
        encoding="utf-8",
    )

    directional, directional_report = _directional_projection(projected_tensors, mapping_path)
    directional_path = result_dir / "thermal_expansion_directional_symmetrized.dat"
    if directional is not None:
        original_directional_path = result_dir / "thermal_expansion_directional.dat"
        fani = None
        if original_directional_path.is_file():
            original_directional = _read_table(original_directional_path, expected_columns=6)
            if np.allclose(original_directional[:, 0], temperatures, rtol=0.0, atol=1.0e-8):
                fani = original_directional[:, 5]
            else:
                directional_report = {
                    **directional_report,
                    "fani_status": "temperature_grid_mismatch_not_copied",
                }
        directional_columns = [temperatures, directional, projected_volume]
        directional_headers = ["T_K", "alpha_a", "alpha_b", "alpha_c", "alpha_volume"]
        if fani is not None:
            directional_columns.append(fani)
            directional_headers.append("F_ani")
            directional_report = {**directional_report, "fani_status": "copied_unchanged"}
        directional_path.write_text(
            rows_to_text_table(
                directional_headers,
                np.column_stack(directional_columns),
            ),
            encoding="utf-8",
        )

    residual_path = result_dir / "thermal_expansion_symmetry_residual.dat"
    residual_path.write_text(
        rows_to_text_table(
            [
                "T_K",
                "residual_frobenius_micro_per_K",
                "residual_max_engineering_component_micro_per_K",
                "residual_relative_to_max_component",
                "allowed_residual_micro_per_K",
                "threshold_exceeded",
            ],
            np.column_stack(
                [
                    temperatures,
                    residual_frobenius,
                    residual_max_component,
                    residual_relative,
                    allowed_residual,
                    violations.astype(float),
                ]
            ),
        ),
        encoding="utf-8",
    )

    original_trace = np.trace(raw_tensors, axis1=-2, axis2=-1)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "warning_large_symmetry_residual" if np.any(violations) else "ok",
        "method": "point_group_average_alpha_sym_equals_mean_R_alpha_raw_RT",
        "units": "micro_per_K",
        "engineering_voigt_order": list(VOIGT_LABELS),
        "result_dir": str(result_dir),
        "reference_structure": str(structure_path),
        "raw_source": str(source_path),
        "symmetry": symmetry_report,
        "directional_projection": directional_report,
        "residual_thresholds": {
            "absolute_micro_per_K": float(absolute_tolerance_micro_per_K),
            "relative_to_largest_raw_engineering_component": float(relative_tolerance),
            "criterion": "max_component_residual <= max(absolute, relative * raw_scale)",
        },
        "residual_summary": {
            "temperature_point_count": int(len(temperatures)),
            "threshold_exceeded_count": int(np.sum(violations)),
            "max_frobenius_micro_per_K": float(np.max(residual_frobenius)),
            "max_engineering_component_micro_per_K": float(np.max(residual_max_component)),
            "max_relative_to_largest_raw_component": float(np.max(residual_relative)),
            "max_volume_trace_change_micro_per_K": float(
                np.max(np.abs(projected_volume - original_trace))
            ),
        },
        "outputs": {
            "raw_cartesian": str(raw_path),
            "symmetrized_cartesian": str(symmetrized_cartesian_path),
            "symmetrized_directional": str(directional_path) if directional is not None else None,
            "residual_table": str(residual_path),
        },
    }
    write_json(result_dir / "thermal_expansion_symmetry_report.json", report)
    return report


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def selected_result_for_material(material_dir: Path, result_subdir: str) -> Path | None:
    root = Path(material_dir) / result_subdir
    decision = _load_json(root / "production_decision.json")
    selected = decision.get("selected_result")
    if selected:
        selected_path = Path(selected)
        return selected_path if selected_path.is_absolute() else root / selected_path
    if (root / "thermal_expansion_cartesian.dat").is_file():
        return root
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--result-dir", type=Path)
    target.add_argument("--material-dir", type=Path)
    target.add_argument("--results-root", type=Path)
    parser.add_argument("--result-subdir", default="anisotropic_thermal_expansion")
    parser.add_argument("--symprec", type=float, default=5.0e-3)
    parser.add_argument("--strict-symprec", type=float, default=1.0e-3)
    parser.add_argument("--angle-tolerance", type=float, default=-1.0)
    parser.add_argument("--absolute-residual-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--relative-residual-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--batch-report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.result_dir is not None:
        result_dirs = [args.result_dir]
    elif args.material_dir is not None:
        selected = selected_result_for_material(args.material_dir, args.result_subdir)
        if selected is None:
            raise SystemExit(f"no_selected_complete_result:{args.material_dir}")
        result_dirs = [selected]
    else:
        root = args.results_root.expanduser().resolve()
        result_dirs = []
        for material_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            selected = selected_result_for_material(material_dir, args.result_subdir)
            if selected is not None:
                result_dirs.append(selected)

    reports = []
    failures = []
    for result_dir in result_dirs:
        try:
            report = process_result_dir(
                result_dir,
                symprec_A=args.symprec,
                strict_symprec_A=args.strict_symprec,
                angle_tolerance_deg=args.angle_tolerance,
                absolute_tolerance_micro_per_K=args.absolute_residual_tolerance,
                relative_tolerance=args.relative_residual_tolerance,
            )
            reports.append(report)
            print(f"{report['status']}: {Path(result_dir).resolve()}", flush=True)
        except Exception as exc:
            failures.append({"result_dir": str(Path(result_dir).resolve()), "error": str(exc)})
            print(f"failed: {result_dir}: {exc}", flush=True)

    batch_report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_result_count": len(result_dirs),
        "processed_count": len(reports),
        "ok_count": sum(report["status"] == "ok" for report in reports),
        "warning_count": sum(report["status"] != "ok" for report in reports),
        "failure_count": len(failures),
        "failures": failures,
    }
    if args.batch_report is not None:
        write_json(args.batch_report.expanduser().resolve(), batch_report)
    print(json.dumps(batch_report, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
