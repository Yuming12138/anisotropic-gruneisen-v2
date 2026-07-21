#!/usr/bin/env python
"""Compare two compliance-weighted alpha-split runs, normally h and h/2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alpha_split_core import compare_split_responses
from v2_runtime_adapter import write_json


def load_alpha_split_table(result_dir: Path) -> dict[str, np.ndarray]:
    path = Path(result_dir) / "alpha_volume_split.dat"
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 13:
        raise ValueError(f"alpha_split_table_column_mismatch:{data.shape[1]}")
    return {
        "temperatures_K": data[:, 0],
        "alpha_volume_positive_per_K": data[:, 1] * 1.0e-6,
        "alpha_volume_negative_per_K": data[:, 2] * 1.0e-6,
        "alpha_volume_unresolved_signed_per_K": data[:, 4] * 1.0e-6,
        "alpha_volume_unresolved_absolute_bound_per_K": data[:, 5] * 1.0e-6,
        "alpha_volume_total_per_K": data[:, 6] * 1.0e-6,
        "ratio_abs_negative_to_positive": data[:, 7],
        "ratio_lower_bound": data[:, 8],
        "ratio_upper_bound": data[:, 9],
        "excluded_heat_capacity_fraction": data[:, 10],
        "unresolved_heat_capacity_fraction": data[:, 11],
        "unresolved_alpha_fraction": data[:, 12],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--fallback-result", type=Path, required=True)
    parser.add_argument("--target-temperature", type=float, default=100.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    parser.add_argument("--absolute-tolerance-micro", type=float, default=0.5)
    parser.add_argument("--ratio-absolute-tolerance", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary = load_alpha_split_table(args.primary_result.expanduser().resolve())
    fallback = load_alpha_split_table(args.fallback_result.expanduser().resolve())
    report = compare_split_responses(
        primary,
        fallback,
        target_temperature_K=args.target_temperature,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance_micro_per_K=args.absolute_tolerance_micro,
        ratio_absolute_tolerance=args.ratio_absolute_tolerance,
    )
    output = args.output
    if output is None:
        output = args.fallback_result / "strain_convergence.json"
    write_json(output.expanduser().resolve(), report)
    print(f"{report['status']}: {output}")


if __name__ == "__main__":
    main()
