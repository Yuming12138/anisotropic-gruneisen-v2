#!/usr/bin/env python
"""Create PNG plots for compliance-weighted alpha-split results.

The module is intentionally usable both from the calculation runners and as a
standalone post-processor for already completed result directories.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ALPHA_SPLIT_PNG = "alpha_volume_split.png"
QHA_COMPARISON_PNG = "qha_vs_gruneisen_thermal_expansion.png"
PLOT_METADATA_JSON = "plot_metadata.json"


def read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_alpha_split(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 13:
        raise ValueError(f"invalid_alpha_split_shape:{path}:{data.shape}")
    if not np.all(np.isfinite(data[:, :7])):
        raise ValueError(f"nonfinite_alpha_split_plot_columns:{path}")
    return {
        "temperature_K": data[:, 0],
        "positive_micro_per_K": data[:, 1],
        "negative_micro_per_K": data[:, 2],
        "total_micro_per_K": data[:, 6],
    }


def load_qha_thermal_expansion(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"invalid_qha_thermal_expansion_shape:{path}:{data.shape}")
    temperature = np.asarray(data[:, 0], dtype=float)
    alpha_per_K = np.asarray(data[:, 1], dtype=float)
    finite = np.isfinite(temperature) & np.isfinite(alpha_per_K)
    if not np.any(finite):
        raise ValueError(f"no_finite_qha_thermal_expansion_rows:{path}")
    return {
        "temperature_K": temperature[finite],
        "alpha_micro_per_K": alpha_per_K[finite] * 1.0e6,
    }


def locate_qha_thermal_expansion(
    material_dir: Path | None,
    explicit_path: Path | None = None,
) -> Path | None:
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        return candidate if candidate.is_file() else None
    if material_dir is None:
        return None
    candidates = (
        material_dir / "thermal_expansion/thermal_properties/thermal_expansion.dat",
        material_dir / "qha/thermal_properties/thermal_expansion.dat",
    )
    return next((path for path in candidates if path.is_file()), None)


def infer_material_dir(result_dir: Path) -> Path | None:
    for parent in (result_dir, *result_dir.parents):
        if (
            len(parent.name) == 4
            and parent.name.isdigit()
            and (parent / "thermal_expansion").is_dir()
        ):
            return parent
    for parent in (result_dir, *result_dir.parents):
        if (parent / "thermal_expansion").is_dir() or (parent / "elastic").is_dir():
            return parent
    return None


def infer_material_label(material_dir: Path | None, result_dir: Path) -> str:
    if material_dir is not None:
        return material_dir.name
    for parent in (result_dir, *result_dir.parents):
        if len(parent.name) == 4 and parent.name.isdigit():
            return parent.name
    return result_dir.name


def target_scope(target: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    return str(target.get("validity_scope") or target.get("scope") or "computed_curve")


def add_scope_annotation(
    axis: Any,
    scope: str,
    target_temperature_K: float,
) -> None:
    normalized = scope.lower().replace("-", "_")
    if normalized in {"300k_only", "only_300k", "ready_300k"}:
        axis.axvline(target_temperature_K, color="0.35", linestyle="--", linewidth=1.2)
        axis.text(
            0.98,
            0.96,
            f"Only {target_temperature_K:g} K validated; full curve is diagnostic",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    elif normalized in {"not_converged", "unresolved", "diagnostic_only"}:
        axis.text(
            0.98,
            0.96,
            "Diagnostic only: strain derivative not converged",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="darkred",
            bbox={"facecolor": "white", "edgecolor": "darkred", "alpha": 0.9},
        )


def save_png(figure: Any, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    figure.savefig(temporary, format="png", dpi=dpi, bbox_inches="tight")
    temporary.replace(output)


def generate_result_plots(
    result_dir: Path,
    *,
    material_dir: Path | None = None,
    qha_thermal_expansion: Path | None = None,
    target_temperature_K: float = 300.0,
    validity_scope: str | None = None,
    dpi: int = 200,
) -> dict[str, Any]:
    result_dir = result_dir.expanduser().resolve()
    alpha_path = result_dir / "alpha_volume_split.dat"
    if not alpha_path.is_file():
        raise FileNotFoundError(f"missing_alpha_volume_split:{alpha_path}")
    if material_dir is None:
        material_dir = infer_material_dir(result_dir)
    elif material_dir is not None:
        material_dir = material_dir.expanduser().resolve()
    qha_path = locate_qha_thermal_expansion(material_dir, qha_thermal_expansion)
    target = read_json_if_present(result_dir / "alpha_volume_split_target.json")
    scope = target_scope(target, validity_scope)
    label = infer_material_label(material_dir, result_dir)
    alpha = load_alpha_split(alpha_path)

    # Delay the heavy plotting import so command-line help and calculation
    # preflight remain responsive.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    generated: list[str] = []
    warnings: list[str] = []

    figure, axis = plt.subplots(figsize=(9.0, 6.2))
    axis.plot(
        alpha["temperature_K"],
        alpha["positive_micro_per_K"],
        color="green",
        linewidth=2.2,
        label=r"$\alpha_{pos}$ ($\alpha_V$, $\chi>0$)",
    )
    axis.plot(
        alpha["temperature_K"],
        alpha["negative_micro_per_K"],
        color="#bf00bf",
        linewidth=2.2,
        label=r"$\alpha_{neg}$ ($\alpha_V$, $\chi<0$)",
    )
    axis.plot(
        alpha["temperature_K"],
        alpha["total_micro_per_K"],
        color="red",
        linewidth=2.2,
        label=r"$\alpha_{tot}$ ($\alpha_V$)",
    )
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)")
    axis.set_title(f"{label}: compliance-weighted thermal expansion")
    axis.legend(loc="best")
    axis.grid(alpha=0.18)
    add_scope_annotation(axis, scope, target_temperature_K)
    split_png = result_dir / ALPHA_SPLIT_PNG
    save_png(figure, split_png, dpi)
    plt.close(figure)
    generated.append(str(split_png))

    comparison_png: Path | None = None
    if qha_path is None:
        warnings.append("qha_thermal_expansion_not_found")
    else:
        try:
            qha = load_qha_thermal_expansion(qha_path)
            figure, axis = plt.subplots(figsize=(9.0, 6.2))
            axis.plot(
                alpha["temperature_K"],
                alpha["total_micro_per_K"],
                color="red",
                linewidth=2.2,
                label=r"Gruneisen ($\alpha_V$)",
            )
            markevery = max(1, len(qha["temperature_K"]) // 45)
            axis.plot(
                qha["temperature_K"],
                qha["alpha_micro_per_K"],
                color="blue",
                linewidth=1.8,
                marker="p",
                markersize=4.5,
                markevery=markevery,
                label="QHA",
            )
            axis.axhline(0.0, color="0.35", linewidth=0.8)
            axis.set_xlabel("Temperature (K)")
            axis.set_ylabel(r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)")
            axis.set_title(f"{label}: Gruneisen and QHA thermal expansion")
            axis.legend(loc="best")
            axis.grid(alpha=0.18)
            add_scope_annotation(axis, scope, target_temperature_K)
            comparison_png = result_dir / QHA_COMPARISON_PNG
            save_png(figure, comparison_png, dpi)
            plt.close(figure)
            generated.append(str(comparison_png))
        except Exception as error:
            plt.close("all")
            warnings.append(f"qha_comparison_failed:{type(error).__name__}:{error}")

    report = {
        "schema_version": 1,
        "status": "complete" if comparison_png is not None else "partial",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "material": label,
        "validity_scope": scope,
        "target_temperature_K": float(target_temperature_K),
        "plotted_quantity": "volumetric_thermal_expansion_coefficient_alpha_V",
        "coordinate_convention": "orthonormal_Cartesian_engineering_Voigt_xx_yy_zz_yz_xz_xy",
        "volumetric_identity": "alpha_V=trace(alpha_Cartesian)=alpha_xx+alpha_yy+alpha_zz",
        "gruneisen_volume_projection": "chi=e^T*S*gamma_with_e=(1,1,1,0,0,0)",
        "qha_input_units": "1/K",
        "plot_units": "1e-6/K",
        "alpha_split_source": str(alpha_path),
        "qha_thermal_expansion_source": str(qha_path) if qha_path else None,
        "alpha_split_png": str(split_png),
        "qha_comparison_png": str(comparison_png) if comparison_png else None,
        "generated_pngs": generated,
        "warnings": warnings,
    }
    (result_dir / PLOT_METADATA_JSON).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def result_dirs_from_summary(path: Path) -> Iterable[tuple[Path, str | None]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            selected = (row.get("selected_result") or "").strip()
            if not selected:
                continue
            yield Path(selected), (row.get("validity_scope") or None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result-dir", type=Path, action="append")
    group.add_argument("--summary-csv", type=Path)
    parser.add_argument("--material-dir", type=Path, default=None)
    parser.add_argument("--qha-thermal-expansion", type=Path, default=None)
    parser.add_argument("--target-temperature", type=float, default=300.0)
    parser.add_argument("--validity-scope", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.summary_csv is not None:
        targets = list(result_dirs_from_summary(args.summary_csv.expanduser().resolve()))
    else:
        targets = [(path, args.validity_scope) for path in args.result_dir]
    failures: list[str] = []
    reports: list[dict[str, Any]] = []
    complete = 0
    partial = 0
    for result_dir, row_scope in targets:
        try:
            report = generate_result_plots(
                result_dir,
                material_dir=args.material_dir,
                qha_thermal_expansion=args.qha_thermal_expansion,
                target_temperature_K=args.target_temperature,
                validity_scope=args.validity_scope or row_scope,
                dpi=args.dpi,
            )
        except Exception as error:
            failures.append(f"{result_dir}:{type(error).__name__}:{error}")
            continue
        if report["status"] == "complete":
            complete += 1
        else:
            partial += 1
        reports.append(report)
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_count": len(targets),
        "complete_count": complete,
        "partial_count": partial,
        "failed_count": len(failures),
        "failures": failures,
        "results": reports,
    }
    if args.summary_csv is not None:
        summary_path = args.summary_csv.expanduser().resolve().with_name(
            "plot_generation_summary.json"
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"plotting complete: targets={len(targets)} complete={complete} "
        f"partial={partial} failed={len(failures)}"
    )
    for failure in failures:
        print(f"plotting failed: {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
