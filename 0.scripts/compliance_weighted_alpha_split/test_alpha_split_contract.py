#!/usr/bin/env python

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.io import write

from compare_alpha_split_runs import load_alpha_split_table
from run_compliance_weighted_alpha_split import (
    assess_split_readiness,
    completed_result_matches,
    parse_args,
    preflight,
    relaxation_branch_report,
    write_preflight_outputs,
)
from v2_runtime_adapter import sha256_file, write_json


class AlphaSplitContractTests(unittest.TestCase):
    def make_material(self, root: Path) -> tuple[Path, Path]:
        material = root / "material"
        elastic = material / "elastic"
        elastic.mkdir(parents=True)
        atoms = Atoms(
            "NaCl",
            cell=np.eye(3) * 5.6,
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            pbc=True,
        )
        write(material / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)
        write(elastic / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)
        stiffness = np.zeros((6, 6))
        stiffness[:3, :3] = 40.0
        np.fill_diagonal(stiffness[:3, :3], 100.0)
        np.fill_diagonal(stiffness[3:, 3:], 30.0)
        np.savetxt(elastic / "ELASTIC_TENSOR", stiffness)
        write_json(elastic / "calculation_metadata.json", {"test": True})
        model = root / "model.pth"
        model.write_bytes(b"test model fingerprint only")
        return material, model

    def test_preflight_is_non_destructive_and_writes_auditable_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            material, model = self.make_material(Path(temporary))
            tensor = material / "elastic" / "ELASTIC_TENSOR"
            before = sha256_file(tensor)
            args = parse_args(
                [
                    "--material-dir",
                    str(material),
                    "--result-subdir",
                    "probe",
                    "--model",
                    str(model),
                    "--preflight-only",
                ]
            )
            report, context = preflight(args)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["phase_consistency"]["status"], "ok")
            write_preflight_outputs(report, context)
            self.assertEqual(before, sha256_file(tensor))
            self.assertTrue((material / "probe" / "effective_strain_path.json").is_file())
            self.assertTrue((material / "probe" / "reference" / "POSCAR").is_file())

    def test_completed_result_requires_matching_fingerprint_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary)
            fingerprint = "abc123"
            write_json(
                result / "calculation_complete.json",
                {"status": "complete", "fingerprint_sha256": fingerprint},
            )
            self.assertFalse(completed_result_matches(result, fingerprint))
            for name in (
                "quality_report.json",
                "run_metadata.json",
                "effective_strain_path.json",
                "alpha_volume_split.dat",
                "alpha_volume_split_target.json",
            ):
                (result / name).write_text("{}\n", encoding="utf-8")
            self.assertTrue(completed_result_matches(result, fingerprint))
            self.assertFalse(completed_result_matches(result, "different"))

    def test_relaxation_branch_report_rejects_large_mapped_displacement(self):
        initial = Atoms(
            "NaCl",
            cell=np.eye(3) * 5.6,
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            pbc=True,
        )
        small = initial.copy()
        small.positions[1, 0] += 0.1
        self.assertEqual(
            relaxation_branch_report(initial, small, 1.0)["status"],
            "ok",
        )
        large = initial.copy()
        large.positions[1, 0] += 1.5
        self.assertEqual(
            relaxation_branch_report(initial, large, 1.0)["status"],
            "relaxation_structure_branch_changed",
        )

    def test_nonfinite_reference_stress_is_a_hard_failure(self):
        quality = {
            "reference_force_stress": {
                "max_force_eV_A": 0.0,
                "max_abs_stress_GPa": float("nan"),
                "stress_status": "unavailable:RuntimeError",
            },
            "phase_consistency_status": "ok",
            "internal_relaxation": {
                "cw_minus": {
                    "status": "converged",
                    "branch_consistency": {"status": "ok"},
                },
                "cw_plus": {
                    "status": "converged",
                    "branch_consistency": {"status": "ok"},
                },
            },
            "max_excluded_heat_capacity_fraction": 0.0,
            "max_unresolved_heat_capacity_fraction": 0.0,
            "max_unresolved_alpha_fraction": 0.0,
            "reference_imaginary_or_zero_count": 3,
            "strained_imaginary_diagnostics": {
                "minus_imaginary_mode_count": 0,
                "plus_imaginary_mode_count": 0,
            },
            "effective_gamma_statistics_1_per_GPa": {"abs_max": 1.0},
            "path_scale_1_per_GPa": 0.01,
        }
        args = SimpleNamespace(
            skip_internal_relax=False,
            max_excluded_cv_fraction=0.05,
            max_unresolved_cv_fraction=0.01,
            max_unresolved_alpha_fraction=0.05,
        )
        report = assess_split_readiness(quality, args)
        self.assertEqual(report["status"], "failed")
        self.assertIn("reference_residual_stress_too_large", report["hard_failures"])

    def test_alpha_split_table_contract_has_thirteen_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary)
            row = np.asarray(
                [[300.0, 10.0, -5.0, 5.0, 0.0, 0.1, 5.0, 0.5, 0.48, 0.52, 0.0, 0.0, 0.01]]
            )
            np.savetxt(result / "alpha_volume_split.dat", row)
            loaded = load_alpha_split_table(result)
            self.assertAlmostEqual(float(loaded["alpha_volume_positive_per_K"][0]), 10.0e-6)
            self.assertAlmostEqual(float(loaded["ratio_upper_bound"][0]), 0.52)
            self.assertAlmostEqual(float(loaded["unresolved_alpha_fraction"][0]), 0.01)


if __name__ == "__main__":
    unittest.main()
