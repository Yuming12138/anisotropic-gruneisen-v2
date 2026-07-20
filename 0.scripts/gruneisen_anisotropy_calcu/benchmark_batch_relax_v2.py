#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read
from ase.optimize import BFGS
from mattersim.applications.batch_relax import BatchRelaxer, DummyBatchCalculator
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.potential import MatterSimCalculator

from gruneisen_v2_core import apply_engineering_strain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark twelve fixed-cell strained relaxations")
    parser.add_argument("--elastic-poscar", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--strain", type=float, default=0.005)
    parser.add_argument("--fmax", type=float, default=1.0e-3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--maxstep", type=float, default=0.1)
    parser.add_argument("--max-natoms-per-batch", type=int, default=2048)
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


def strained_states(reference_atoms, strain: float) -> list:
    return [
        apply_engineering_strain(reference_atoms, component, sign * strain)
        for component in range(1, 7)
        for sign in (-1, 1)
    ]


def state_report(atoms, steps: int, converged: bool) -> dict[str, Any]:
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return {
        "steps": steps,
        "converged": converged,
        "energy_eV": float(atoms.get_potential_energy()),
        "max_force_eV_A": float(np.max(np.linalg.norm(forces, axis=1))),
        "scaled_positions": atoms.get_scaled_positions(wrap=False).tolist(),
        "cell": atoms.cell.array.tolist(),
    }


def run_sequential(
    states: list,
    calculator: Any,
    device: str,
    fmax: float,
    max_steps: int,
    maxstep: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calculator.reset()
    baseline = reset_memory(device)
    synchronize(device)
    start = time.perf_counter()
    reports = []
    for state in states:
        atoms = state.copy()
        atoms.calc = calculator
        optimizer = BFGS(atoms, logfile=None, maxstep=maxstep)
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
        reports.append(state_report(atoms, optimizer.nsteps, converged))
    synchronize(device)
    elapsed = time.perf_counter() - start
    return reports, {"seconds": elapsed, **memory_metrics(device, baseline)}


class FixedCellBatchRelaxer(BatchRelaxer):
    def __init__(self, *args, maxstep: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxstep = maxstep
        self.max_steps_reached: list[int] = []

    def insert(self, atoms):
        atoms.calc = DummyBatchCalculator()
        optimizer = self.optimizer(atoms, logfile=None, maxstep=self.maxstep)
        optimizer.fmax = self.fmax
        optimizer.nsteps = 0
        self.optimizer_instances.append(optimizer)
        self.is_active_instance.append(True)

    def step_batch(self):
        active_optimizers = [
            optimizer
            for optimizer, active in zip(self.optimizer_instances, self.is_active_instance)
            if active
        ]
        if not active_optimizers:
            self.finished = True
            return
        dataloader = build_dataloader(
            [optimizer.atoms for optimizer in active_optimizers],
            batch_size=len(active_optimizers),
            only_inference=True,
            batch_converter=False,
        )
        energies, forces, _ = self.potential.predict_properties(
            dataloader,
            include_forces=True,
            include_stresses=False,
        )

        counter = 0
        self.finished = True
        for index, optimizer in enumerate(self.optimizer_instances):
            if not self.is_active_instance[index]:
                continue
            optimizer.atoms.info["total_energy"] = energies[counter]
            optimizer.atoms.arrays["forces"] = forces[counter]
            structure_index = int(optimizer.atoms.info["structure_index"])
            self.trajectories.setdefault(structure_index, []).append(optimizer.atoms.copy())
            gradient = optimizer.optimizable.get_gradient()
            if optimizer.converged(gradient):
                self.is_active_instance[index] = False
                self.total_converged += 1
            elif optimizer.nsteps >= self.max_n_steps:
                self.is_active_instance[index] = False
                self.max_steps_reached.append(structure_index)
            else:
                optimizer.step()
                optimizer.nsteps += 1
                self.finished = False
            counter += 1

        self.optimizer_instances = [
            optimizer
            for optimizer, active in zip(self.optimizer_instances, self.is_active_instance)
            if active
        ]
        self.is_active_instance = [True] * len(self.optimizer_instances)


def run_batch(
    states: list,
    potential: Any,
    device: str,
    fmax: float,
    max_steps: int,
    maxstep: float,
    max_natoms_per_batch: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = reset_memory(device)
    synchronize(device)
    start = time.perf_counter()
    relaxer = FixedCellBatchRelaxer(
        potential,
        optimizer="BFGS",
        filter=None,
        fmax=fmax,
        max_natoms_per_batch=max_natoms_per_batch,
        max_n_steps=max_steps,
        maxstep=maxstep,
    )
    trajectories = relaxer.relax(states)
    synchronize(device)
    elapsed = time.perf_counter() - start
    reports = []
    for index in range(len(states)):
        trajectory = trajectories[index]
        final = trajectory[-1]
        final.calc = DummyBatchCalculator()
        reports.append(
            state_report(
                final,
                max(0, len(trajectory) - 1),
                index not in relaxer.max_steps_reached,
            )
        )
    return reports, {
        "seconds": elapsed,
        "max_steps_reached": sorted(relaxer.max_steps_reached),
        **memory_metrics(device, baseline),
    }


def maximum_position_difference(left: dict, right: dict) -> float:
    left_scaled = np.asarray(left["scaled_positions"], dtype=float)
    right_scaled = np.asarray(right["scaled_positions"], dtype=float)
    fractional = right_scaled - left_scaled
    fractional -= np.rint(fractional)
    cartesian = fractional @ np.asarray(left["cell"], dtype=float)
    return float(np.max(np.linalg.norm(cartesian, axis=1)))


def compare_reports(reference: list[dict], candidate: list[dict]) -> dict[str, float]:
    return {
        "max_position_difference_A": max(
            maximum_position_difference(left, right)
            for left, right in zip(reference, candidate)
        ),
        "max_energy_difference_eV": max(
            abs(left["energy_eV"] - right["energy_eV"])
            for left, right in zip(reference, candidate)
        ),
        "max_final_force_difference_eV_A": max(
            abs(left["max_force_eV_A"] - right["max_force_eV_A"])
            for left, right in zip(reference, candidate)
        ),
    }


def main() -> None:
    args = parse_args()
    reference = read(args.elastic_poscar.expanduser().resolve())
    states = strained_states(reference, args.strain)
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
    warmup = states[0].copy()
    warmup.calc = calculator_forces
    warmup.get_forces()
    calculator_forces.reset()

    stress_reports, stress_metrics = run_sequential(
        states,
        calculator_stress,
        args.device,
        args.fmax,
        args.max_steps,
        args.maxstep,
    )
    force_reports, force_metrics = run_sequential(
        states,
        calculator_forces,
        args.device,
        args.fmax,
        args.max_steps,
        args.maxstep,
    )
    batch_reports, batch_metrics = run_batch(
        states,
        calculator_stress.potential,
        args.device,
        args.fmax,
        args.max_steps,
        args.maxstep,
        args.max_natoms_per_batch,
    )

    result = {
        "elastic_poscar": str(args.elastic_poscar.expanduser().resolve()),
        "unit_cell_atoms": len(reference),
        "state_count": len(states),
        "max_natoms_per_batch": args.max_natoms_per_batch,
        "stress_sequential": {
            **stress_metrics,
            "states": stress_reports,
        },
        "force_sequential": {
            **force_metrics,
            "speedup_vs_stress": stress_metrics["seconds"] / force_metrics["seconds"],
            "comparison_vs_stress": compare_reports(stress_reports, force_reports),
            "states": force_reports,
        },
        "force_batch": {
            **batch_metrics,
            "speedup_vs_stress": stress_metrics["seconds"] / batch_metrics["seconds"],
            "speedup_vs_force_sequential": force_metrics["seconds"] / batch_metrics["seconds"],
            "comparison_vs_stress": compare_reports(stress_reports, batch_reports),
            "comparison_vs_force_sequential": compare_reports(force_reports, batch_reports),
            "states": batch_reports,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
