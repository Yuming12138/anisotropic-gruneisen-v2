#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from symmetrize_thermal_expansion import (
    cartesian_point_group,
    engineering_voigt_to_tensor,
    process_result_dir,
    symmetrize_tensors,
    tensor_to_engineering_voigt,
)


class ThermalExpansionSymmetryTests(unittest.TestCase):
    def test_engineering_voigt_round_trip_preserves_shear_convention(self):
        values = np.asarray([[1.0, 2.0, 3.0, 0.4, -0.6, 0.8]])
        tensor = engineering_voigt_to_tensor(values)
        self.assertAlmostEqual(tensor[0, 1, 2], 0.2)
        self.assertAlmostEqual(tensor[0, 0, 2], -0.3)
        self.assertAlmostEqual(tensor[0, 0, 1], 0.4)
        np.testing.assert_allclose(tensor_to_engineering_voigt(tensor), values)

    def test_cubic_group_average_is_isotropic_and_trace_preserving(self):
        structure = Structure(Lattice.cubic(4.0), ["Cu"], [[0.0, 0.0, 0.0]])
        operations, report = cartesian_point_group(structure, symprec_A=1.0e-3)
        self.assertEqual(report["spacegroup_number"], 221)
        self.assertEqual(report["point_group_operation_count"], 48)
        raw = np.asarray(
            [[[1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 4.0]]]
        )
        projected = symmetrize_tensors(raw, operations)
        expected = np.eye(3) * np.trace(raw[0]) / 3.0
        np.testing.assert_allclose(projected[0], expected, atol=1.0e-12)
        self.assertAlmostEqual(float(np.trace(projected[0])), float(np.trace(raw[0])))

    def test_orthorhombic_group_keeps_independent_diagonal_terms(self):
        structure = Structure(
            Lattice.orthorhombic(3.0, 4.0, 5.0), ["Si"], [[0.0, 0.0, 0.0]]
        )
        operations, report = cartesian_point_group(structure, symprec_A=1.0e-3)
        self.assertEqual(report["spacegroup_number"], 47)
        raw = np.asarray(
            [[[1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 4.0]]]
        )
        projected = symmetrize_tensors(raw, operations)
        np.testing.assert_allclose(projected[0], np.diag([1.0, 2.0, 4.0]), atol=1.0e-12)

    def test_result_postprocess_preserves_raw_and_writes_auditable_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            reference_dir = result_dir / "reference"
            reference_dir.mkdir()
            structure = Structure(Lattice.cubic(4.0), ["Cu"], [[0.0, 0.0, 0.0]])
            structure.to(filename=reference_dir / "POSCAR", fmt="poscar")
            (reference_dir / "structure_mapping.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "axis_unit_vectors_in_elastic_frame": {
                            "a": [1.0, 0.0, 0.0],
                            "b": [0.0, 1.0, 0.0],
                            "c": [0.0, 0.0, 1.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            original_text = (
                "# T_K alpha_xx alpha_yy alpha_zz alpha_yz_eng alpha_xz_eng "
                "alpha_xy_eng alpha_volume\n"
                "3.0000000000e+02 7.0 8.1 9.2 0.03 -0.04 0.05 24.3\n"
            )
            cartesian = result_dir / "thermal_expansion_cartesian.dat"
            cartesian.write_text(original_text, encoding="utf-8")
            (result_dir / "thermal_expansion_directional.dat").write_text(
                "# T_K alpha_a alpha_b alpha_c alpha_volume F_ani\n"
                "3.0000000000e+02 7.0 8.1 9.2 24.3 0.25\n",
                encoding="utf-8",
            )

            report = process_result_dir(result_dir)

            self.assertEqual(report["status"], "warning_large_symmetry_residual")
            self.assertEqual(
                (result_dir / "thermal_expansion_cartesian_raw.dat").read_text(
                    encoding="utf-8"
                ),
                original_text,
            )
            self.assertTrue(
                (result_dir / "thermal_expansion_cartesian_symmetrized.dat").is_file()
            )
            self.assertTrue(
                (result_dir / "thermal_expansion_directional_symmetrized.dat").is_file()
            )
            self.assertTrue(
                (result_dir / "thermal_expansion_symmetry_residual.dat").is_file()
            )
            saved_report = json.loads(
                (result_dir / "thermal_expansion_symmetry_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved_report["symmetry"]["requested"]["spacegroup_number"], 221)
            projected = np.loadtxt(
                result_dir / "thermal_expansion_cartesian_symmetrized.dat", ndmin=2
            )
            np.testing.assert_allclose(projected[0, 1:4], [8.1, 8.1, 8.1])
            np.testing.assert_allclose(projected[0, 4:7], 0.0, atol=1.0e-12)
            self.assertAlmostEqual(projected[0, 7], 24.3)
            directional = np.loadtxt(
                result_dir / "thermal_expansion_directional_symmetrized.dat", ndmin=2
            )
            self.assertEqual(directional.shape[1], 6)
            self.assertAlmostEqual(directional[0, 5], 0.25)


if __name__ == "__main__":
    unittest.main()
