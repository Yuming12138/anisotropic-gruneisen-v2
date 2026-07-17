#!/usr/bin/env python
"""Coordinate-consistent six-component anisotropic Gruneisen v2 runner.

The default mode is a real calculation.  Use ``--preflight-only`` to validate
inputs and write an auditable report without importing MatterSim or calculating
forces.  Existing legacy outputs and authoritative elastic tensors are never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from phonopy.gruneisen.mesh import GruneisenMesh
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure

from gruneisen_v2_core import (
    V2Parameters,
    apply_engineering_strain,
    choose_supercell_matrix,
    compute_thermal_response,
    input_fingerprint,
    read_elastic_tensor,
    rows_to_text_table,
    runtime_versions,
    sha256_file,
    stable_json_hash,
    structure_axis_mapping,
    summarize_effective_isotropy,
    validate_elastic_tensor,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
CORE_PATH = SCRIPT_DIR / "gruneisen_v2_core.py"
DEFAULT_RESULT_SUBDIR = "gruneisen_aniso_1M_v2"
MODEL_FILENAMES = {
    "1M": "mattersim-v1.0.0-1M.pth",
    "5M": "mattersim-v1.0.0-5M.pth",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-dir", type=Path, required=True)
    parser.add_argument("--result-subdir", default=DEFAULT_RESULT_SUBDIR)
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
    parser.add_argument(
        "--skip-internal-relax",
        action="store_true",
        help="Diagnostic only: skip fixed-cell internal-coordinate relaxation.",
    )
    args = parser.parse_args(argv)
    result_subdir = Path(args.result_subdir)
    if result_subdir.is_absolute() or ".." in result_subdir.parts:
        raise SystemExit("--result-subdir must be a relative path inside the material directory")
    if args.strain <= 0.0 or args.fallback_strain <= 0.0:
        raise SystemExit("strain amplitudes must be positive")
    if any(value <= 0 for value in args.mesh):
        raise SystemExit("mesh values must be positive")
    if args.tmax < args.tmin or args.tstep <= 0.0:
        raise SystemExit("invalid temperature range")
    if not 0.0 <= args.fani_threshold <= 1.0:
        raise SystemExit("--fani-threshold must be between 0 and 1")
    if args.sign_tolerance_micro < 0.0:
        raise SystemExit("--sign-tolerance-micro must be non-negative")
    return args


def model_candidates(model_size: str) -> list[Path]:
    filename = MODEL_FILENAMES[model_size]
    candidates: list[Path] = []
    if value := os.environ.get("MATTERSIM_MODEL"):
        candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            Path.home() / ".local" / "mattersim" / "pretrained_models" / filename,
            SCRIPT_DIR / filename,
        ]
    )
    return candidates


def resolve_model_path(explicit: Path | None, model_size: str) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = model_candidates(model_size)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].expanduser()


def choose_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda_requested_but_unavailable")
    return requested


class DiagnosticGruneisenMesh(GruneisenMesh):
    """GruneisenMesh that also records true strained imaginary-mode counts."""

    def __init__(self, *args: Any, imaginary_cutoff_THz: float, **kwargs: Any) -> None:
        factor = float(get_physical_units().DefaultToTHz)
        self._negative_eigenvalue_threshold = -((float(imaginary_cutoff_THz) / factor) ** 2)
        self.minus_imaginary_counts_by_q: list[int] = []
        self.plus_imaginary_counts_by_q: list[int] = []
        self.minus_min_eigenvalues: list[float] = []
        self.plus_min_eigenvalues: list[float] = []
        super().__init__(*args, **kwargs)

    def _get_dD(self, q: np.ndarray, d_a: Any, d_b: Any) -> np.ndarray:
        # GruneisenBase passes d_a=minus and d_b=plus.
        difference = super()._get_dD(q, d_a, d_b)
        eigen_minus = np.linalg.eigvalsh(d_a.dynamical_matrix)
        eigen_plus = np.linalg.eigvalsh(d_b.dynamical_matrix)
        self.minus_imaginary_counts_by_q.append(
            int(np.sum(eigen_minus < self._negative_eigenvalue_threshold))
        )
        self.plus_imaginary_counts_by_q.append(
            int(np.sum(eigen_plus < self._negative_eigenvalue_threshold))
        )
        self.minus_min_eigenvalues.append(float(np.min(eigen_minus)))
        self.plus_min_eigenvalues.append(float(np.min(eigen_plus)))
        return difference


def ase_to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell.array),
        scaled_positions=atoms.get_scaled_positions(wrap=False),
        masses=atoms.get_masses(),
    )


def phonopy_to_ase_atoms(atoms: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=list(atoms.symbols),
        cell=np.asarray(atoms.cell),
        scaled_positions=np.asarray(atoms.scaled_positions),
        pbc=True,
    )


def make_phonon(atoms: Atoms, supercell_matrix: np.ndarray) -> Phonopy:
    return Phonopy(
        ase_to_phonopy_atoms(atoms),
        supercell_matrix=np.asarray(supercell_matrix, dtype=int),
        primitive_matrix=np.eye(3),
        symprec=1.0e-5,
        log_level=1,
    )


def fixed_cell_relax(
    atoms: Atoms,
    calculator: Any,
    run_dir: Path,
    fmax: float,
    max_steps: int,
    skip: bool,
) -> tuple[Atoms, dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    unrelaxed_path = run_dir / "POSCAR_unrelaxed"
    relaxed_path = run_dir / "CONTCAR"
    write(unrelaxed_path, atoms, format="vasp", direct=True, vasp5=True)
    if skip:
        relaxed = atoms.copy()
        status = "skipped_by_user"
        steps = 0
    else:
        relaxed = atoms.copy()
        relaxed.calc = calculator
        optimizer = BFGS(relaxed, logfile=str(run_dir / "internal_relax.log"))
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
        steps = int(optimizer.nsteps)
        status = "converged" if converged else "max_steps_reached"
    relaxed.calc = calculator
    forces = np.asarray(relaxed.get_forces(), dtype=float)
    energy = float(relaxed.get_potential_energy())
    max_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0
    write(relaxed_path, relaxed, format="vasp", direct=True, vasp5=True)
    report = {
        "status": status,
        "steps": steps,
        "energy_eV": energy,
        "max_force_eV_A": max_force,
        "atom_count": len(relaxed),
        "symbols": relaxed.get_chemical_symbols(),
        "volume_A3": float(relaxed.get_volume()),
    }
    write_json(run_dir / "internal_relax_report.json", report)
    return relaxed, report


def calculate_force_constants(
    atoms: Atoms,
    calculator: Any,
    run_dir: Path,
    supercell_matrix: np.ndarray,
    displacement_A: float,
    state_fingerprint: dict[str, Any],
    resume: bool,
    force: bool,
) -> Phonopy:
    run_dir.mkdir(parents=True, exist_ok=True)
    fc_path = run_dir / "FORCE_CONSTANTS"
    metadata_path = run_dir / "force_constants_metadata.json"
    state_hash = stable_json_hash(state_fingerprint)
    if fc_path.is_file() and metadata_path.is_file() and not force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("state_fingerprint_sha256") == state_hash:
            phonon = make_phonon(atoms, supercell_matrix)
            phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(fc_path))
            return phonon
        if resume:
            raise RuntimeError(f"cache_fingerprint_mismatch:{run_dir}")
    elif fc_path.is_file() and not force:
        if resume:
            raise RuntimeError(f"cache_metadata_missing:{run_dir}")

    write(run_dir / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)
    phonon = make_phonon(atoms, supercell_matrix)
    phonon.generate_displacements(distance=float(displacement_A))
    displaced_supercells = phonon.supercells_with_displacements
    if displaced_supercells is None:
        raise RuntimeError("phonopy_displacement_generation_failed")
    forces = []
    for index, supercell in enumerate(displaced_supercells, start=1):
        displaced = phonopy_to_ase_atoms(supercell)
        displaced.calc = calculator
        forces.append(np.asarray(displaced.get_forces(), dtype=float))
        if index % 25 == 0 or index == len(displaced_supercells):
            print(f"    displaced supercells: {index}/{len(displaced_supercells)}", flush=True)
    phonon.forces = np.asarray(forces)
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    write_FORCE_CONSTANTS(phonon.force_constants, filename=str(fc_path))
    metadata = {
        "state_fingerprint_sha256": state_hash,
        "state_fingerprint": state_fingerprint,
        "displacement_count": len(displaced_supercells),
        "force_constants_shape": list(np.asarray(phonon.force_constants).shape),
    }
    write_json(metadata_path, metadata)
    return phonon


def reference_force_stress_report(atoms: Atoms, calculator: Any) -> dict[str, Any]:
    checked = atoms.copy()
    checked.calc = calculator
    forces = np.asarray(checked.get_forces(), dtype=float)
    energy = float(checked.get_potential_energy())
    try:
        stress = np.asarray(checked.get_stress(voigt=True), dtype=float)
        stress_GPa = stress / 0.006241509125883258
        stress_status = "ok"
    except Exception as exc:
        stress_GPa = np.full(6, np.nan)
        stress_status = f"unavailable:{type(exc).__name__}"
    return {
        "energy_eV": energy,
        "max_force_eV_A": float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0,
        "stress_voigt_GPa": stress_GPa.tolist(),
        "max_abs_stress_GPa": float(np.nanmax(np.abs(stress_GPa))),
        "stress_status": stress_status,
    }


def build_parameters(args: argparse.Namespace) -> V2Parameters:
    return V2Parameters(
        model_size=args.model_size,
        dtype=args.dtype,
        strain=float(args.strain),
        fallback_strain=float(args.fallback_strain),
        displacement_A=float(args.displacement),
        mesh=tuple(int(value) for value in args.mesh),
        min_supercell_length_A=float(args.min_supercell_length),
        internal_relax_fmax_eV_A=float(args.fmax),
        internal_relax_max_steps=int(args.max_steps),
        frequency_cutoff_THz=float(args.frequency_cutoff),
        temperature_min_K=float(args.tmin),
        temperature_max_K=float(args.tmax),
        temperature_step_K=float(args.tstep),
        fani_threshold=float(args.fani_threshold),
        sign_tolerance_micro_per_K=float(args.sign_tolerance_micro),
    )


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    material_dir = args.material_dir.expanduser().resolve()
    result_dir = material_dir / args.result_subdir
    root_poscar = material_dir / "POSCAR"
    elastic_dir = material_dir / "elastic"
    elastic_poscar = elastic_dir / "POSCAR"
    elastic_tensor = elastic_dir / "ELASTIC_TENSOR"
    required = (root_poscar, elastic_poscar, elastic_tensor)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing_required_inputs:" + ";".join(missing))

    root_structure = Structure.from_file(root_poscar)
    elastic_structure = Structure.from_file(elastic_poscar)
    C_raw = read_elastic_tensor(elastic_tensor)
    C, elastic_report = validate_elastic_tensor(C_raw)
    axis_mapping = structure_axis_mapping(root_structure, elastic_structure)
    if args.supercell is None:
        supercell = choose_supercell_matrix(
            elastic_structure.lattice.matrix,
            minimum_length_A=args.min_supercell_length,
        )
        supercell_source = "automatic_minimum_length"
    else:
        supercell = np.diag(np.asarray(args.supercell, dtype=int))
        supercell_source = "explicit_cli"

    model_path = resolve_model_path(args.model, args.model_size)
    parameters = build_parameters(args)
    fingerprint = input_fingerprint(
        root_poscar=root_poscar,
        elastic_poscar=elastic_poscar,
        elastic_tensor=elastic_tensor,
        model_path=model_path if model_path.is_file() else None,
        parameters=parameters,
        runner_path=SCRIPT_PATH,
        core_path=CORE_PATH,
    )
    formula_match = root_structure.composition.reduced_composition == elastic_structure.composition.reduced_composition
    atom_count_match = len(root_structure) == len(elastic_structure)
    issues = []
    if not formula_match:
        issues.append("root_elastic_formula_mismatch")
    if not atom_count_match:
        issues.append("root_elastic_atom_count_mismatch")
    if not elastic_report["positive_definite"]:
        issues.append("elastic_not_positive_definite")
    if elastic_report["ill_conditioned"]:
        issues.append("elastic_ill_conditioned")
    if axis_mapping["status"] != "ok":
        issues.append(axis_mapping["status"])
    if not model_path.is_file():
        issues.append("mattersim_model_missing_in_current_environment")
    if not (elastic_dir / "calculation_metadata.json").is_file():
        issues.append("elastic_calculation_metadata_missing")

    blocking = [
        issue
        for issue in issues
        if issue
        in {
            "root_elastic_formula_mismatch",
            "root_elastic_atom_count_mismatch",
            "elastic_not_positive_definite",
            "elastic_ill_conditioned",
        }
    ]
    report = {
        "schema_version": 1,
        "material": material_dir.name,
        "material_dir": str(material_dir),
        "result_dir": str(result_dir),
        "status": "failed" if blocking else ("warning" if issues else "ok"),
        "blocking_issues": blocking,
        "issues": issues,
        "root_poscar": str(root_poscar),
        "elastic_poscar": str(elastic_poscar),
        "elastic_tensor": str(elastic_tensor),
        "root_formula": root_structure.composition.reduced_formula,
        "elastic_formula": elastic_structure.composition.reduced_formula,
        "root_atom_count": len(root_structure),
        "elastic_atom_count": len(elastic_structure),
        "elastic": elastic_report,
        "axis_mapping": axis_mapping,
        "supercell_matrix": supercell.tolist(),
        "supercell_source": supercell_source,
        "supercell_lengths_A": (
            np.linalg.norm(np.asarray(elastic_structure.lattice.matrix) * np.diag(supercell)[:, None], axis=1)
        ).tolist(),
        "model_path": str(model_path),
        "model_present": model_path.is_file(),
        "parameters": asdict(parameters),
        "fingerprint": fingerprint,
        "runtime_versions": runtime_versions(),
    }
    context = {
        "material_dir": material_dir,
        "result_dir": result_dir,
        "root_poscar": root_poscar,
        "elastic_poscar": elastic_poscar,
        "elastic_tensor": elastic_tensor,
        "root_structure": root_structure,
        "elastic_structure": elastic_structure,
        "stiffness_GPa": C,
        "compliance_1_per_GPa": np.linalg.inv(C) if elastic_report["positive_definite"] else None,
        "axis_mapping": axis_mapping,
        "supercell_matrix": supercell,
        "model_path": model_path,
        "parameters": parameters,
        "fingerprint": fingerprint,
    }
    return report, context


def write_preflight_outputs(report: dict[str, Any], context: dict[str, Any]) -> None:
    result_dir: Path = context["result_dir"]
    reference_dir = result_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(context["elastic_poscar"], reference_dir / "POSCAR")
    shutil.copy2(context["root_poscar"], reference_dir / "ROOT_POSCAR")
    write_json(reference_dir / "structure_mapping.json", context["axis_mapping"])
    write_json(result_dir / "preflight_report.json", report)
    run_metadata_path = result_dir / "run_metadata.json"
    preserve_complete = False
    if run_metadata_path.is_file():
        try:
            preserve_complete = (
                json.loads(run_metadata_path.read_text(encoding="utf-8")).get("calculation_status")
                == "complete"
            )
        except Exception:
            preserve_complete = False
    if not preserve_complete:
        write_json(
            run_metadata_path,
            {
                "schema_version": 1,
                "calculation_status": "preflight_complete",
                "fingerprint": context["fingerprint"],
                "runtime_versions": report["runtime_versions"],
                "parameters": report["parameters"],
            },
        )
    np.savetxt(
        result_dir / "elastic_tensor_used.dat",
        context["stiffness_GPa"],
        header="Stiffness tensor C_ij in GPa; Voigt order xx yy zz yz xz xy",
        fmt="%16.8f",
    )


def run_gruneisen(args: argparse.Namespace, report: dict[str, Any], context: dict[str, Any]) -> None:
    if report["blocking_issues"]:
        raise RuntimeError("preflight_blocked:" + ";".join(report["blocking_issues"]))
    model_path: Path = context["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"MatterSim model not found: {model_path}")

    from mattersim.forcefield.potential import MatterSimCalculator

    device = choose_device(args.device)
    calculator = MatterSimCalculator(
        load_path=str(model_path),
        device=device,
        dtype=args.dtype,
        compute_stress=True,
    )
    result_dir: Path = context["result_dir"]
    work_dir = result_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    reference_atoms = read(context["elastic_poscar"])
    reference_report = reference_force_stress_report(reference_atoms, calculator)
    write_json(result_dir / "reference" / "residual_force_stress.json", reference_report)

    parameters: V2Parameters = context["parameters"]
    common_state = {
        "input_fingerprint_sha256": context["fingerprint"]["fingerprint_sha256"],
        "supercell_matrix": context["supercell_matrix"].tolist(),
        "displacement_A": parameters.displacement_A,
        "primitive_matrix": np.eye(3).tolist(),
        "model_sha256": sha256_file(model_path),
        "dtype": args.dtype,
    }

    print("[v2] reference force constants", flush=True)
    phonon_zero = calculate_force_constants(
        atoms=reference_atoms,
        calculator=calculator,
        run_dir=work_dir / "strain_0",
        supercell_matrix=context["supercell_matrix"],
        displacement_A=parameters.displacement_A,
        state_fingerprint={**common_state, "component": 0, "strain": 0.0},
        resume=args.resume,
        force=args.force,
    )

    strained_phonons: dict[tuple[int, int], Phonopy] = {}
    relax_reports: dict[str, Any] = {}
    for component in range(1, 7):
        for sign, tag in ((-1, "minus"), (1, "plus")):
            amplitude = sign * parameters.strain
            state_name = f"eta{component}_{tag}"
            print(f"[v2] {state_name}: engineering strain {amplitude:+.6f}", flush=True)
            strained = apply_engineering_strain(reference_atoms, component, amplitude)
            relaxed, relax_report = fixed_cell_relax(
                strained,
                calculator,
                work_dir / state_name,
                fmax=parameters.internal_relax_fmax_eV_A,
                max_steps=parameters.internal_relax_max_steps,
                skip=args.skip_internal_relax,
            )
            if relaxed.get_chemical_symbols() != reference_atoms.get_chemical_symbols():
                raise RuntimeError(f"atom_order_changed:{state_name}")
            relax_reports[state_name] = relax_report
            strained_phonons[(component, sign)] = calculate_force_constants(
                atoms=relaxed,
                calculator=calculator,
                run_dir=work_dir / state_name,
                supercell_matrix=context["supercell_matrix"],
                displacement_A=parameters.displacement_A,
                state_fingerprint={
                    **common_state,
                    "component": component,
                    "strain": amplitude,
                    "relaxed_structure_sha256": sha256_file(work_dir / state_name / "CONTCAR"),
                },
                resume=args.resume,
                force=args.force,
            )

    gammas = []
    qpoints_ref = None
    weights_ref = None
    frequencies_ref = None
    strained_imaginary_diagnostics: list[dict[str, Any]] = []
    for component in range(1, 7):
        print(f"[v2] GruneisenMesh eta{component}", flush=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
                module="phonopy.gruneisen.core",
            )
            mesh = DiagnosticGruneisenMesh(
                phonon_zero.dynamical_matrix,
                strained_phonons[(component, 1)].dynamical_matrix,
                strained_phonons[(component, -1)].dynamical_matrix,
                mesh=parameters.mesh,
                delta_strain=2.0 * parameters.strain,
                is_time_reversal=True,
                is_gamma_center=True,
                is_mesh_symmetry=False,
                imaginary_cutoff_THz=parameters.frequency_cutoff_THz,
            )
        qpoints = np.asarray(mesh.get_qpoints(), dtype=float)
        weights = np.asarray(mesh.get_weights(), dtype=float)
        frequencies = np.asarray(mesh.get_frequencies(), dtype=float)
        gamma = np.asarray(mesh.get_gruneisen(), dtype=float)
        if qpoints_ref is None:
            qpoints_ref = qpoints
            weights_ref = weights
            frequencies_ref = frequencies
        else:
            if not np.allclose(qpoints, qpoints_ref) or not np.allclose(weights, weights_ref):
                raise RuntimeError(f"gruneisen_mesh_mismatch:eta{component}")
            if not np.allclose(frequencies, frequencies_ref, rtol=1.0e-8, atol=1.0e-10):
                raise RuntimeError(f"reference_frequency_mismatch:eta{component}")
        gammas.append(gamma)
        strained_imaginary_diagnostics.append(
            {
                "component": component,
                "minus_imaginary_mode_count": int(
                    np.dot(weights, np.asarray(mesh.minus_imaginary_counts_by_q, dtype=float))
                ),
                "plus_imaginary_mode_count": int(
                    np.dot(weights, np.asarray(mesh.plus_imaginary_counts_by_q, dtype=float))
                ),
                "minus_min_eigenvalue": float(np.min(mesh.minus_min_eigenvalues)),
                "plus_min_eigenvalue": float(np.min(mesh.plus_min_eigenvalues)),
            }
        )

    assert qpoints_ref is not None and weights_ref is not None and frequencies_ref is not None
    gamma_array = np.asarray(gammas)
    np.savez_compressed(
        result_dir / "gruneisen_mesh.npz",
        qpoints=qpoints_ref,
        weights=weights_ref,
        frequencies_0_thz=frequencies_ref,
        gamma_voigt=gamma_array,
        voigt_labels=np.asarray(["xx", "yy", "zz", "yz", "xz", "xy"]),
    )

    response, quality = compute_thermal_response(
        temperatures_K=parameters.temperatures(),
        frequencies_THz=frequencies_ref,
        gammas=gamma_array,
        weights=weights_ref,
        compliance_1_per_GPa=context["compliance_1_per_GPa"],
        volume_A3=float(phonon_zero.primitive.volume),
        frequency_cutoff_THz=parameters.frequency_cutoff_THz,
        axis_mapping=context["axis_mapping"],
    )
    effective_screen = summarize_effective_isotropy(
        response,
        fani_threshold=parameters.fani_threshold,
        sign_tolerance_micro_per_K=parameters.sign_tolerance_micro_per_K,
    )
    quality.update(
        {
            "reference_force_stress": reference_report,
            "internal_relaxation": relax_reports,
            "strain_convergence_status": "not_checked_fallback_pending",
            "strained_imaginary_diagnostics": strained_imaginary_diagnostics,
            "effective_isotropy_screen": effective_screen,
        }
    )
    write_json(result_dir / "quality_report.json", quality)

    T = response["temperatures_K"]
    alpha_micro = response["alpha_voigt_per_K"] * 1.0e6
    cartesian_rows = np.column_stack([T, alpha_micro, response["alpha_volume_per_K"] * 1.0e6])
    (result_dir / "thermal_expansion_cartesian.dat").write_text(
        rows_to_text_table(
            ["T_K", "alpha_xx", "alpha_yy", "alpha_zz", "alpha_yz_eng", "alpha_xz_eng", "alpha_xy_eng", "alpha_volume"],
            cartesian_rows,
        ),
        encoding="utf-8",
    )
    directional_rows = np.column_stack(
        [
            T,
            response["alpha_directional_per_K"] * 1.0e6,
            response["alpha_volume_per_K"] * 1.0e6,
            response["F_ani"],
        ]
    )
    (result_dir / "thermal_expansion_directional.dat").write_text(
        rows_to_text_table(
            ["T_K", "alpha_a", "alpha_b", "alpha_c", "alpha_volume", "F_ani"],
            directional_rows,
        ),
        encoding="utf-8",
    )
    integral_rows = np.column_stack([T, response["gruneisen_integrals_J_per_K"]])
    (result_dir / "gruneisen_integrals.dat").write_text(
        rows_to_text_table(["T_K", "I_xx", "I_yy", "I_zz", "I_yz", "I_xz", "I_xy"], integral_rows),
        encoding="utf-8",
    )
    macro_rows = np.column_stack([T, response["macro_strain_gruneisen"]])
    (result_dir / "gruneisen_temperature_voigt.dat").write_text(
        rows_to_text_table(["T_K", "G_xx", "G_yy", "G_zz", "G_yz", "G_xz", "G_xy"], macro_rows),
        encoding="utf-8",
    )
    fani_rows = np.column_stack(
        [
            T,
            response["alpha_volume_hyd_per_K"] * 1.0e6,
            response["alpha_volume_dev_per_K"] * 1.0e6,
            response["alpha_volume_per_K"] * 1.0e6,
            response["F_ani"],
            response["excluded_heat_capacity_fraction"],
        ]
    )
    (result_dir / "fani_temperature.dat").write_text(
        rows_to_text_table(
            ["T_K", "alphaV_hyd", "alphaV_dev", "alphaV_total", "F_ani", "excluded_Cv_fraction"],
            fani_rows,
        ),
        encoding="utf-8",
    )

    metadata = json.loads((result_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "calculation_status": "complete",
            "device": device,
            "runtime_versions": runtime_versions(),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
        }
    )
    write_json(result_dir / "run_metadata.json", metadata)
    write_json(
        result_dir / "calculation_complete.json",
        {
            "status": "complete",
            "fingerprint_sha256": context["fingerprint"]["fingerprint_sha256"],
            "quality_report": str(result_dir / "quality_report.json"),
            "fani_file": str(result_dir / "fani_temperature.dat"),
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report, context = preflight(args)
    write_preflight_outputs(report, context)
    print(f"preflight_status: {report['status']}")
    print(f"result_dir: {context['result_dir']}")
    if report["issues"]:
        print("issues: " + ";".join(report["issues"]))
    if args.preflight_only:
        return
    run_gruneisen(args, report, context)
    print(f"complete: {context['result_dir']}")


if __name__ == "__main__":
    main()
