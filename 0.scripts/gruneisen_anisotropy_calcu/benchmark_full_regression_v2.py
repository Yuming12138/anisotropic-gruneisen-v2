#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.potential import MatterSimCalculator
from phonopy import Phonopy
from pymatgen.core import Structure

from gruneisen_v2_core import (
    compute_thermal_response,
    read_elastic_tensor,
    structure_axis_mapping,
    summarize_effective_isotropy,
    validate_elastic_tensor,
)
from run_gruneisen_thermal_expansion_v2 import (
    DiagnosticGruneisenMesh,
    make_phonon,
    phonopy_to_ase_atoms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end anisotropic-v2 acceleration regression"
    )
    parser.add_argument("--material-id", required=True)
    parser.add_argument("--root-poscar", type=Path, required=True)
    parser.add_argument("--elastic-poscar", type=Path, required=True)
    parser.add_argument("--elastic-tensor", type=Path, required=True)
    parser.add_argument("--relax-benchmark-json", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--supercell", type=int, nargs=3, required=True)
    parser.add_argument("--mesh", type=int, nargs=3, default=(6, 6, 6))
    parser.add_argument("--strain", type=float, default=0.005)
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--batch-atom-cap", type=int, default=1024)
    parser.add_argument("--isolate", action="store_true")
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def atoms_from_report(symbols: list[str], report: dict[str, Any]) -> Atoms:
    return Atoms(
        symbols=symbols,
        cell=np.asarray(report["cell"], dtype=float),
        scaled_positions=np.asarray(report["scaled_positions"], dtype=float),
        pbc=True,
    )


def calculate_phonon(
    atoms: Atoms,
    calculator: Any,
    supercell_matrix: np.ndarray,
    displacement: float,
    batch_atom_cap: int | None,
    batch_include_stresses: bool,
    device: str,
) -> tuple[Phonopy, dict[str, Any]]:
    phonon = make_phonon(atoms, supercell_matrix)
    phonon.generate_displacements(distance=displacement)
    supercells = phonon.supercells_with_displacements
    if supercells is None or not supercells:
        raise RuntimeError("phonopy_displacement_generation_failed")
    structures = [phonopy_to_ase_atoms(supercell) for supercell in supercells]
    supercell_atoms = len(structures[0])
    start = time.perf_counter()
    if batch_atom_cap is None or batch_atom_cap // supercell_atoms <= 1:
        forces = []
        calculator.reset()
        for structure in structures:
            structure.calc = calculator
            forces.append(np.asarray(structure.get_forces(), dtype=float))
        batch_size = 1
        method = "calculator_serial"
    else:
        potential = calculator.potential
        batch_size = min(len(structures), batch_atom_cap // supercell_atoms)
        dataloader = build_dataloader(
            atoms=structures,
            cutoff=float(potential.model.model_args["cutoff"]),
            threebody_cutoff=float(potential.model.model_args["threebody_cutoff"]),
            batch_size=batch_size,
            model_type=potential.model_name,
            shuffle=False,
            only_inference=True,
            num_workers=0,
            batch_converter=False,
        )
        _, forces, _ = potential.predict_properties(
            dataloader,
            include_forces=True,
            include_stresses=batch_include_stresses,
        )
        method = (
            "legacy_patched_batch_with_stress"
            if batch_include_stresses
            else "legacy_patched_batch_force_only"
        )
    synchronize(device)
    elapsed = time.perf_counter() - start
    phonon.forces = np.asarray(forces)
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    return phonon, {
        "method": method,
        "displacements": len(structures),
        "supercell_atoms": supercell_atoms,
        "batch_size": batch_size,
        "seconds": elapsed,
    }


def build_mode(
    label: str,
    reference_atoms: Atoms,
    state_reports: list[dict[str, Any]],
    calculator: Any,
    supercell_matrix: np.ndarray,
    displacement: float,
    batch_atom_cap: int | None,
    mesh_numbers: tuple[int, int, int],
    strain: float,
    compliance: np.ndarray,
    axis_mapping: dict[str, Any],
    device: str,
    batch_include_stresses: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    phonons: dict[tuple[int, int], Phonopy] = {}
    timings = []
    start_force_constants = time.perf_counter()
    phonon_zero, timing = calculate_phonon(
        reference_atoms,
        calculator,
        supercell_matrix,
        displacement,
        batch_atom_cap,
        batch_include_stresses,
        device,
    )
    timings.append({"state": "reference", **timing})
    symbols = reference_atoms.get_chemical_symbols()
    for index, report in enumerate(state_reports):
        component = index // 2 + 1
        sign = -1 if index % 2 == 0 else 1
        atoms = atoms_from_report(symbols, report)
        phonon, timing = calculate_phonon(
            atoms,
            calculator,
            supercell_matrix,
            displacement,
            batch_atom_cap,
            batch_include_stresses,
            device,
        )
        phonons[(component, sign)] = phonon
        timings.append({"state": f"eta{component}_{'minus' if sign < 0 else 'plus'}", **timing})
    force_constant_seconds = time.perf_counter() - start_force_constants

    gammas = []
    qpoints_ref = None
    weights_ref = None
    frequencies_ref = None
    start_mesh = time.perf_counter()
    for component in range(1, 7):
        mesh = DiagnosticGruneisenMesh(
            phonon_zero.dynamical_matrix,
            phonons[(component, 1)].dynamical_matrix,
            phonons[(component, -1)].dynamical_matrix,
            mesh=mesh_numbers,
            delta_strain=2.0 * strain,
            is_time_reversal=True,
            is_gamma_center=True,
            is_mesh_symmetry=False,
            imaginary_cutoff_THz=1.0e-4,
        )
        qpoints = np.asarray(mesh.get_qpoints(), dtype=float)
        weights = np.asarray(mesh.get_weights(), dtype=float)
        frequencies = np.asarray(mesh.get_frequencies(), dtype=float)
        if qpoints_ref is None:
            qpoints_ref = qpoints
            weights_ref = weights
            frequencies_ref = frequencies
        gammas.append(np.asarray(mesh.get_gruneisen(), dtype=float))
    mesh_seconds = time.perf_counter() - start_mesh
    gamma_array = np.asarray(gammas)
    temperatures = np.arange(10.0, 1000.0 + 5.0, 10.0)
    response, quality = compute_thermal_response(
        temperatures_K=temperatures,
        frequencies_THz=frequencies_ref,
        gammas=gamma_array,
        weights=weights_ref,
        compliance_1_per_GPa=compliance,
        volume_A3=float(phonon_zero.primitive.volume),
        frequency_cutoff_THz=1.0e-4,
        axis_mapping=axis_mapping,
    )
    screen = summarize_effective_isotropy(response)
    arrays = {
        "qpoints": qpoints_ref,
        "weights": weights_ref,
        "frequencies": frequencies_ref,
        "gammas": gamma_array,
        "temperatures": response["temperatures_K"],
        "alpha_voigt": response["alpha_voigt_per_K"],
        "alpha_volume": response["alpha_volume_per_K"],
        "alpha_directional": response["alpha_directional_per_K"],
        "F_ani": response["F_ani"],
    }
    metadata = {
        "label": label,
        "force_constant_seconds": force_constant_seconds,
        "mesh_seconds": mesh_seconds,
        "state_timings": timings,
        "quality": quality,
        "effective_screen": screen,
    }
    return arrays, metadata


def array_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    finite = np.isfinite(left) & np.isfinite(right)
    differences = np.abs(left[finite] - right[finite])
    return {
        "shape_equal": left.shape == right.shape,
        "finite_mismatch_count": int(np.sum(np.isfinite(left) != np.isfinite(right))),
        "max_abs": float(np.max(differences)) if differences.size else None,
        "mean_abs": float(np.mean(differences)) if differences.size else None,
        "p99_abs": float(np.quantile(differences, 0.99)) if differences.size else None,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_atoms = read(args.elastic_poscar.expanduser().resolve())
    relax_data = json.loads(args.relax_benchmark_json.read_text(encoding="utf-8"))
    stiffness, _ = validate_elastic_tensor(read_elastic_tensor(args.elastic_tensor))
    compliance = np.linalg.inv(stiffness)
    axis_mapping = structure_axis_mapping(
        Structure.from_file(args.root_poscar),
        Structure.from_file(args.elastic_poscar),
    )
    supercell_matrix = np.diag(np.asarray(args.supercell, dtype=int))

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
    warmup = reference_atoms.copy()
    warmup.calc = calculator_forces
    warmup.get_forces()
    calculator_forces.reset()

    baseline_arrays, baseline_metadata = build_mode(
        "baseline_stress_serial",
        reference_atoms,
        relax_data["stress_sequential"]["states"],
        calculator_stress,
        supercell_matrix,
        args.displacement,
        None,
        tuple(args.mesh),
        args.strain,
        compliance,
        axis_mapping,
        args.device,
    )
    optimized_arrays, optimized_metadata = build_mode(
        "optimized_batch_relax_force_only",
        reference_atoms,
        relax_data["force_batch"]["states"],
        calculator_forces,
        supercell_matrix,
        args.displacement,
        args.batch_atom_cap,
        tuple(args.mesh),
        args.strain,
        compliance,
        axis_mapping,
        args.device,
    )
    modes = {
        "baseline": (baseline_arrays, baseline_metadata),
        "optimized": (optimized_arrays, optimized_metadata),
    }
    if args.isolate:
        modes["force_only_on_baseline_relax"] = build_mode(
            "force_only_on_baseline_relax",
            reference_atoms,
            relax_data["stress_sequential"]["states"],
            calculator_forces,
            supercell_matrix,
            args.displacement,
            args.batch_atom_cap,
            tuple(args.mesh),
            args.strain,
            compliance,
            axis_mapping,
            args.device,
        )
        modes["stress_serial_on_batch_relax"] = build_mode(
            "stress_serial_on_batch_relax",
            reference_atoms,
            relax_data["force_batch"]["states"],
            calculator_stress,
            supercell_matrix,
            args.displacement,
            None,
            tuple(args.mesh),
            args.strain,
            compliance,
            axis_mapping,
            args.device,
        )
        modes["stress_batch_on_baseline_relax"] = build_mode(
            "stress_batch_on_baseline_relax",
            reference_atoms,
            relax_data["stress_sequential"]["states"],
            calculator_stress,
            supercell_matrix,
            args.displacement,
            args.batch_atom_cap,
            tuple(args.mesh),
            args.strain,
            compliance,
            axis_mapping,
            args.device,
            batch_include_stresses=True,
        )
    for mode, (arrays, _) in modes.items():
        np.savez_compressed(output_dir / f"{mode}_arrays.npz", **arrays)
    comparisons = {
        mode: {
            key: array_difference(baseline_arrays[key], arrays[key])
            for key in baseline_arrays
        }
        for mode, (arrays, _) in modes.items()
        if mode != "baseline"
    }
    differences = comparisons["optimized"]
    temperature_index = int(np.argmin(np.abs(baseline_arrays["temperatures"] - 300.0)))
    summary = {
        "material_id": args.material_id,
        "mesh": list(args.mesh),
        "supercell": list(args.supercell),
        "batch_atom_cap": args.batch_atom_cap,
        "baseline": baseline_metadata,
        "optimized": optimized_metadata,
        "isolation_modes": {
            mode: metadata
            for mode, (_, metadata) in modes.items()
            if mode not in {"baseline", "optimized"}
        },
        "differences": differences,
        "comparisons_vs_baseline": comparisons,
        "at_300K": {
            "baseline_alpha_voigt_per_K": baseline_arrays["alpha_voigt"][
                temperature_index
            ].tolist(),
            "optimized_alpha_voigt_per_K": optimized_arrays["alpha_voigt"][
                temperature_index
            ].tolist(),
            "baseline_alpha_volume_per_K": float(
                baseline_arrays["alpha_volume"][temperature_index]
            ),
            "optimized_alpha_volume_per_K": float(
                optimized_arrays["alpha_volume"][temperature_index]
            ),
            "baseline_F_ani": float(baseline_arrays["F_ani"][temperature_index]),
            "optimized_F_ani": float(optimized_arrays["F_ani"][temperature_index]),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
