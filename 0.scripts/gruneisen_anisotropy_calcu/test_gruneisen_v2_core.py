#!/usr/bin/env python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms

from gruneisen_v2_core import (
    V2Parameters,
    assess_production_readiness,
    apply_engineering_strain,
    choose_supercell_matrix,
    compute_thermal_response,
    compare_mesh_responses,
    compare_strain_derivatives,
    engineering_strain_tensor,
    input_fingerprint,
    runtime_versions,
    sha256_file,
    summarize_effective_isotropy,
    validate_elastic_tensor,
)
from run_gruneisen_production_v2 import main as production_main
from run_gruneisen_production_v2 import StageFailure
from run_gruneisen_production_v2 import load_response_tables
from run_gruneisen_production_v2 import stage_is_complete
from run_gruneisen_production_v2 import strain_tag
from run_gruneisen_thermal_expansion_v2 import COMPLETE_ARTIFACTS, completed_result_matches


class GruneisenV2CoreTests(unittest.TestCase):
    def test_engineering_shear_convention_and_row_vector_cell(self):
        amplitude = 0.02
        strain = engineering_strain_tensor(4, amplitude)
        self.assertAlmostEqual(strain[1, 2], amplitude / 2.0)
        self.assertAlmostEqual(strain[2, 1], amplitude / 2.0)
        atoms = Atoms("He", cell=np.eye(3), scaled_positions=[[0, 0, 0]], pbc=True)
        strained = apply_engineering_strain(atoms, 4, amplitude)
        expected = np.eye(3) + strain
        np.testing.assert_allclose(strained.cell.array, expected)

    def test_supercell_minimum_length(self):
        cell = np.diag([3.0, 7.0, 20.0])
        matrix = choose_supercell_matrix(cell, minimum_length_A=12.0)
        np.testing.assert_array_equal(matrix, np.diag([4, 2, 2]))

    def test_isotropic_elastic_tensor_is_positive_definite(self):
        lam = 40.0
        mu = 30.0
        C = np.zeros((6, 6))
        C[:3, :3] = lam
        np.fill_diagonal(C[:3, :3], lam + 2.0 * mu)
        np.fill_diagonal(C[3:, 3:], mu)
        sym, report = validate_elastic_tensor(C)
        np.testing.assert_allclose(sym, C)
        self.assertTrue(report["positive_definite"])
        self.assertFalse(report["ill_conditioned"])

    def test_hydrostatic_gamma_gives_zero_Fani(self):
        frequencies = np.asarray([[1.0, 2.0, 3.0]])
        gammas = np.zeros((6, 1, 3))
        gammas[0:3, :, :] = 1.5
        C = np.zeros((6, 6))
        C[:3, :3] = 40.0
        np.fill_diagonal(C[:3, :3], 100.0)
        np.fill_diagonal(C[3:, 3:], 30.0)
        compliance = np.linalg.inv(C)
        mapping = {
            "status": "ok",
            "axis_unit_vectors_in_elastic_frame": {
                "a": [1.0, 0.0, 0.0],
                "b": [0.0, 1.0, 0.0],
                "c": [0.0, 0.0, 1.0],
            },
        }
        result, quality = compute_thermal_response(
            temperatures_K=np.asarray([300.0]),
            frequencies_THz=frequencies,
            gammas=gammas,
            weights=np.asarray([1.0]),
            compliance_1_per_GPa=compliance,
            volume_A3=100.0,
            frequency_cutoff_THz=1.0e-4,
            axis_mapping=mapping,
        )
        self.assertAlmostEqual(float(result["F_ani"][0]), 0.0, places=12)
        directional = result["alpha_directional_per_K"][0]
        np.testing.assert_allclose(directional, directional[0], rtol=1.0e-12, atol=1.0e-20)
        self.assertEqual(quality["valid_mode_count"], 3)
        screen = summarize_effective_isotropy(result, fani_threshold=0.20)
        self.assertEqual(screen["status"], "effective_isotropic_candidate")
        self.assertTrue(screen["passed"])

    def test_production_readiness_requests_fallback_for_soft_modes(self):
        quality = {
            "axis_mapping_status": "ok",
            "reference_force_stress": {
                "max_force_eV_A": 1.0e-4,
                "max_abs_stress_GPa": 1.0e-3,
            },
            "internal_relaxation": {"eta1_minus": {"status": "converged"}},
            "max_excluded_heat_capacity_fraction": 0.01,
            "reference_imaginary_or_zero_count": 4,
            "strained_imaginary_diagnostics": [
                {"minus_imaginary_mode_count": 1, "plus_imaginary_mode_count": 0}
            ],
            "gamma_statistics": [{"abs_max": 800.0}],
        }
        report = assess_production_readiness(quality)
        self.assertEqual(report["status"], "requires_strain_check")
        self.assertTrue(report["fallback_required"])
        self.assertEqual(len(report["fallback_reasons"]), 3)

    def test_production_readiness_accepts_stable_result(self):
        quality = {
            "axis_mapping_status": "ok",
            "reference_force_stress": {
                "max_force_eV_A": 1.0e-4,
                "max_abs_stress_GPa": 1.0e-3,
            },
            "internal_relaxation": {"eta1_minus": {"status": "converged"}},
            "max_excluded_heat_capacity_fraction": 0.01,
            "reference_imaginary_or_zero_count": 3,
            "strained_imaginary_diagnostics": [
                {"minus_imaginary_mode_count": 0, "plus_imaginary_mode_count": 0}
            ],
            "gamma_statistics": [{"abs_max": 100.0}],
        }
        report = assess_production_readiness(quality)
        self.assertEqual(report["status"], "ready")
        self.assertFalse(report["fallback_required"])

    def test_strain_derivative_comparison(self):
        temperatures = np.asarray([100.0, 300.0])
        primary_integrals = np.ones((2, 6)) * 1.0e-22
        fallback_integrals = primary_integrals * 1.02
        primary_alpha_volume = np.asarray([20.0e-6, 30.0e-6])
        fallback_alpha_volume = np.asarray([20.2e-6, 30.2e-6])
        primary_directional = np.ones((2, 3)) * 10.0e-6
        fallback_directional = primary_directional + 0.1e-6
        converged = compare_strain_derivatives(
            temperatures,
            primary_integrals,
            fallback_integrals,
            primary_alpha_volume,
            fallback_alpha_volume,
            primary_directional,
            fallback_directional,
        )
        self.assertEqual(converged["status"], "converged")

        fallback_integrals[:, 0] = 2.0e-22
        fallback_alpha_volume[0] = 22.0e-6
        fallback_directional[0, 0] = 13.0e-6
        unresolved = compare_strain_derivatives(
            temperatures,
            primary_integrals,
            fallback_integrals,
            primary_alpha_volume,
            fallback_alpha_volume,
            primary_directional,
            fallback_directional,
        )
        self.assertEqual(unresolved["status"], "strain_derivative_unresolved")
        self.assertEqual(len(unresolved["reasons"]), 3)

    def test_mesh_response_comparison(self):
        temperatures = np.asarray([100.0, 300.0])
        screening_integrals = np.ones((2, 6)) * 1.0e-22
        dense_integrals = screening_integrals * 1.005
        screening_alpha_volume = np.asarray([20.0e-6, 30.0e-6])
        dense_alpha_volume = screening_alpha_volume + 0.1e-6
        screening_directional = np.ones((2, 3)) * 10.0e-6
        dense_directional = screening_directional + 0.1e-6
        screening_fani = np.asarray([0.1, 0.2])
        dense_fani = screening_fani + 0.001
        converged = compare_mesh_responses(
            temperatures,
            screening_integrals,
            dense_integrals,
            screening_alpha_volume,
            dense_alpha_volume,
            screening_directional,
            dense_directional,
            screening_fani,
            dense_fani,
        )
        self.assertEqual(converged["status"], "converged")

        dense_integrals[:, 0] = 2.0e-22
        dense_alpha_volume[0] = 22.0e-6
        dense_directional[0, 0] = 13.0e-6
        dense_fani[0] = 0.2
        unresolved = compare_mesh_responses(
            temperatures,
            screening_integrals,
            dense_integrals,
            screening_alpha_volume,
            dense_alpha_volume,
            screening_directional,
            dense_directional,
            screening_fani,
            dense_fani,
        )
        self.assertEqual(unresolved["status"], "mesh_convergence_unresolved")
        self.assertEqual(len(unresolved["reasons"]), 4)

    def test_strain_tag_tracks_nondefault_values(self):
        self.assertEqual(strain_tag(0.005), "0p005")
        self.assertEqual(strain_tag(0.0025), "0p0025")
        self.assertEqual(strain_tag(0.00125), "0p00125")

    def test_completed_result_requires_matching_fingerprint_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            for name in COMPLETE_ARTIFACTS:
                (result_dir / name).write_text("test\n", encoding="utf-8")
            (result_dir / "calculation_complete.json").write_text(
                json.dumps({"status": "complete", "fingerprint_sha256": "abc"}),
                encoding="utf-8",
            )
            self.assertTrue(completed_result_matches(result_dir, "abc"))
            self.assertFalse(completed_result_matches(result_dir, "different"))
            (result_dir / COMPLETE_ARTIFACTS[0]).unlink()
            self.assertFalse(completed_result_matches(result_dir, "abc"))

    def test_production_stage_failure_persists_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            material_dir = Path(temporary) / "material"
            material_dir.mkdir()
            with self.assertRaisesRegex(SystemExit, "2"):
                production_main(
                    [
                        "--material-dir",
                        str(material_dir),
                        "--result-subdir",
                        "failure_probe",
                        "--python",
                        "/bin/false",
                    ]
                )
            decision_path = material_dir / "failure_probe" / "production_decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(decision["status"], "failed_primary_stage")
            self.assertEqual(decision["failure"]["returncode"], 1)

    def test_production_fast_resume_revalidates_input_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_dir = root / "result"
            result_dir.mkdir()
            source = root / "source.dat"
            source.write_text("original\n", encoding="utf-8")
            for name in COMPLETE_ARTIFACTS:
                (result_dir / name).write_text("test\n", encoding="utf-8")
            fingerprint = {
                "fingerprint_sha256": "abc",
                "execution": {"runtime_versions": runtime_versions()},
                "files": {
                    "source": {"path": str(source), "sha256": sha256_file(source)}
                },
            }
            (result_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "calculation_status": "complete",
                        "fingerprint": fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            (result_dir / "calculation_complete.json").write_text(
                json.dumps({"status": "complete", "fingerprint_sha256": "abc"}),
                encoding="utf-8",
            )
            command = ["python", "runner", "--resume"]
            self.assertTrue(stage_is_complete(result_dir, command, command.copy()))
            source.write_text("changed\n", encoding="utf-8")
            self.assertFalse(stage_is_complete(result_dir, command, command.copy()))

    def test_input_fingerprint_includes_execution_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / name for name in ("root", "elastic", "tensor", "runner", "core")]
            for path in paths:
                path.write_text(path.name + "\n", encoding="utf-8")
            common = {
                "root_poscar": paths[0],
                "elastic_poscar": paths[1],
                "elastic_tensor": paths[2],
                "model_path": None,
                "parameters": V2Parameters(),
                "runner_path": paths[3],
                "core_path": paths[4],
            }
            batch = input_fingerprint(
                **common,
                execution={"supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]},
            )
            serial = input_fingerprint(
                **common,
                execution={"supercell_matrix": [[3, 0, 0], [0, 2, 0], [0, 0, 2]]},
            )
            self.assertNotEqual(batch["fingerprint_sha256"], serial["fingerprint_sha256"])

    def test_production_fallback_stage_failure_persists_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            material_dir = Path(temporary) / "material"
            material_dir.mkdir()

            def fake_stage(command, result_subdir, result_dir, previous_command, log_path):
                if "primary_" in str(result_subdir):
                    result_dir.mkdir(parents=True)
                    (result_dir / "quality_report.json").write_text(
                        json.dumps(
                            {
                                "production_readiness": {
                                    "status": "requires_strain_check",
                                    "hard_failures": [],
                                    "fallback_required": True,
                                    "fallback_reasons": ["test"],
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                    temperatures = np.asarray([100.0, 300.0])
                    np.savetxt(
                        result_dir / "gruneisen_integrals.dat",
                        np.column_stack([temperatures, np.ones((2, 6))]),
                    )
                    np.savetxt(
                        result_dir / "thermal_expansion_directional.dat",
                        np.column_stack([temperatures, np.ones((2, 5))]),
                    )
                    np.savetxt(
                        result_dir / "thermal_expansion_cartesian.dat",
                        np.column_stack([temperatures, np.ones((2, 7))]),
                    )
                    return
                raise StageFailure(command, 7, result_subdir)

            with patch(
                "run_gruneisen_production_v2.execute_or_resume_stage",
                side_effect=fake_stage,
            ):
                with self.assertRaisesRegex(SystemExit, "2"):
                    production_main(
                        [
                            "--material-dir",
                            str(material_dir),
                            "--result-subdir",
                            "fallback_failure_probe",
                            "--python",
                            "/bin/true",
                        ]
                    )
            decision = json.loads(
                (
                    material_dir
                    / "fallback_failure_probe"
                    / "production_decision.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(decision["status"], "failed_fallback_stage")
            self.assertEqual(decision["failure"]["returncode"], 7)

    def test_response_table_validation_rejects_truncated_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            temperatures = np.asarray([100.0, 300.0])
            np.savetxt(
                result_dir / "gruneisen_integrals.dat",
                np.column_stack([temperatures, np.ones((2, 6))]),
            )
            np.savetxt(
                result_dir / "thermal_expansion_directional.dat",
                np.column_stack([temperatures, np.ones((2, 5))]),
            )
            np.savetxt(
                result_dir / "thermal_expansion_cartesian.dat",
                np.column_stack([temperatures, np.ones((2, 7))]),
            )
            tables = load_response_tables(result_dir)
            np.testing.assert_array_equal(tables["temperatures"], temperatures)
            (result_dir / "thermal_expansion_directional.dat").write_text(
                "100 1 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "invalid_directional_table"):
                load_response_tables(result_dir)


if __name__ == "__main__":
    unittest.main()
