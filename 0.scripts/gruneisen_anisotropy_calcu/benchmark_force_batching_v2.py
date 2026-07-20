#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read

from gruneisen_v2_core import apply_engineering_strain, choose_supercell_matrix
from run_gruneisen_thermal_expansion_v2 import make_phonon, phonopy_to_ase_atoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark anisotropic-v2 phonon force batching")
    parser.add_argument("--elastic-poscar", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--component", type=int, default=6)
    parser.add_argument("--strain", type=float, default=0.005)
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--min-supercell-length", type=float, default=12.0)
    parser.add_argument("--max-displacements", type=int, default=32)
    parser.add_argument("--max-sampled-atoms", type=int, default=10240)
    parser.add_argument("--atom-caps", type=int, nargs="+", default=(512, 1024, 2048))
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def reset_memory(device: str) -> int:
    if device != "cuda":
        return 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.memory_allocated())


def memory_metrics(device: str, baseline: int) -> dict[str, float | None]:
    if device != "cuda":
        return {"peak_allocated_gib": None, "incremental_peak_gib": None}
    peak = int(torch.cuda.max_memory_allocated())
    gib = 1024**3
    return {
        "peak_allocated_gib": peak / gib,
        "incremental_peak_gib": max(0, peak - baseline) / gib,
    }


def run_serial(calculator: Any, structures: list, device: str) -> tuple[list[np.ndarray], dict]:
    calculator.reset()
    baseline = reset_memory(device)
    synchronize(device)
    start = time.perf_counter()
    forces = []
    for structure in structures:
        atoms = structure.copy()
        atoms.calc = calculator
        forces.append(np.asarray(atoms.get_forces(), dtype=float))
    synchronize(device)
    elapsed = time.perf_counter() - start
    return forces, {"seconds": elapsed, **memory_metrics(device, baseline)}


def run_batch(
    potential: Any,
    structures: list,
    batch_size: int,
    device: str,
    repeats: int,
    batch_converter: bool,
) -> tuple[list[np.ndarray], dict]:
    from mattersim.datasets.utils.build import build_dataloader

    cutoff = float(potential.model.model_args["cutoff"])
    threebody_cutoff = float(potential.model.model_args["threebody_cutoff"])
    start_build = time.perf_counter()
    dataloader = build_dataloader(
        atoms=structures,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
        batch_size=batch_size,
        model_type=potential.model_name,
        shuffle=False,
        only_inference=True,
        num_workers=0,
        batch_converter=batch_converter,
    )
    build_seconds = time.perf_counter() - start_build

    baseline = reset_memory(device)
    synchronize(device)
    start_cold = time.perf_counter()
    _, forces, _ = potential.predict_properties(dataloader, include_forces=True)
    synchronize(device)
    cold_seconds = time.perf_counter() - start_cold

    warm_seconds = []
    for _ in range(repeats):
        synchronize(device)
        start_warm = time.perf_counter()
        _, forces, _ = potential.predict_properties(dataloader, include_forces=True)
        synchronize(device)
        warm_seconds.append(time.perf_counter() - start_warm)
    return forces, {
        "graph_build_seconds": build_seconds,
        "cold_inference_seconds": cold_seconds,
        "warm_inference_seconds": warm_seconds,
        "warm_inference_median_seconds": statistics.median(warm_seconds),
        **memory_metrics(device, baseline),
    }


def max_force_difference(reference: list[np.ndarray], candidate: list[np.ndarray]) -> float:
    return max(
        float(np.max(np.abs(reference_force - candidate_force)))
        for reference_force, candidate_force in zip(reference, candidate)
    )


def main() -> None:
    args = parse_args()
    from mattersim.forcefield.potential import MatterSimCalculator

    atoms = read(args.elastic_poscar.expanduser().resolve())
    strained = apply_engineering_strain(atoms, args.component, args.strain)
    supercell_matrix = choose_supercell_matrix(
        atoms.cell.array,
        minimum_length_A=args.min_supercell_length,
    )
    phonon = make_phonon(strained, supercell_matrix)
    phonon.generate_displacements(distance=args.displacement)
    all_supercells = phonon.supercells_with_displacements
    if all_supercells is None or not all_supercells:
        raise RuntimeError("phonopy_displacement_generation_failed")

    supercell_atoms = len(all_supercells[0])
    atom_limited_count = max(1, args.max_sampled_atoms // supercell_atoms)
    sample_count = min(len(all_supercells), args.max_displacements, atom_limited_count)
    sample_indices = np.linspace(0, len(all_supercells) - 1, sample_count, dtype=int)
    structures = [phonopy_to_ase_atoms(all_supercells[index]) for index in sample_indices]

    calculator_stress = MatterSimCalculator(
        load_path=str(args.model.expanduser().resolve()),
        device=args.device,
        dtype=args.dtype,
        compute_stress=True,
    )
    calculator_forces = MatterSimCalculator.from_potential(
        calculator_stress.potential,
        device=args.device,
        compute_stress=False,
    )
    warmup = structures[0].copy()
    warmup.calc = calculator_forces
    warmup.get_forces()
    synchronize(args.device)

    stress_forces, stress_metrics = run_serial(calculator_stress, structures, args.device)
    force_forces, force_metrics = run_serial(calculator_forces, structures, args.device)

    batch_results = []
    for atom_cap in args.atom_caps:
        batch_size = max(1, min(len(structures), atom_cap // supercell_atoms))
        converter_modes = (False, True) if args.device == "cuda" else (False,)
        for batch_converter in converter_modes:
            forces, metrics = run_batch(
                calculator_stress.potential,
                structures,
                batch_size,
                args.device,
                args.repeats,
                batch_converter,
            )
            steady_seconds = (
                metrics["graph_build_seconds"] + metrics["warm_inference_median_seconds"]
            )
            batch_results.append(
                {
                    "converter": "gpu_batch" if batch_converter else "legacy_cpu",
                    "atom_cap": atom_cap,
                    "batch_size_structures": batch_size,
                    "batch_atoms": batch_size * supercell_atoms,
                    **metrics,
                    "steady_total_seconds": steady_seconds,
                    "speedup_vs_stress_serial": stress_metrics["seconds"] / steady_seconds,
                    "speedup_vs_force_serial": force_metrics["seconds"] / steady_seconds,
                    "max_force_abs_diff_ev_per_a": max_force_difference(force_forces, forces),
                }
            )

    first_parameter = next(calculator_stress.potential.model.parameters())
    result = {
        "elastic_poscar": str(args.elastic_poscar.expanduser().resolve()),
        "model": str(args.model.expanduser().resolve()),
        "device": args.device,
        "requested_dtype": args.dtype,
        "model_parameter_dtype": str(first_parameter.dtype),
        "unit_cell_atoms": len(atoms),
        "supercell_matrix": np.asarray(supercell_matrix, dtype=int).tolist(),
        "supercell_atoms": supercell_atoms,
        "total_displacements": len(all_supercells),
        "sampled_displacements": len(structures),
        "sample_indices": sample_indices.tolist(),
        "stress_serial": stress_metrics,
        "force_only_serial": {
            **force_metrics,
            "speedup_vs_stress_serial": stress_metrics["seconds"] / force_metrics["seconds"],
            "max_force_abs_diff_ev_per_a": max_force_difference(stress_forces, force_forces),
        },
        "batch_results": batch_results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
