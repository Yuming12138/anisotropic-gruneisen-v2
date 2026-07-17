#!/usr/bin/env python
import unittest

import numpy as np
from ase import Atoms

from gruneisen_v2_core import (
    apply_engineering_strain,
    choose_supercell_matrix,
    compute_thermal_response,
    engineering_strain_tensor,
    summarize_effective_isotropy,
    validate_elastic_tensor,
)


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


if __name__ == "__main__":
    unittest.main()
