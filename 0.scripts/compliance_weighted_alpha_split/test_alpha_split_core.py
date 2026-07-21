#!/usr/bin/env python

from __future__ import annotations

import unittest

import numpy as np
from ase import Atoms

from alpha_split_core import (
    apply_engineering_strain_vector,
    compare_split_responses,
    compliance_weighted_path,
    compute_alpha_volume_split,
    effective_gamma_1_per_GPa,
    summarize_at_temperature,
)
from v2_runtime_adapter import compute_thermal_response, strain_voigt_to_tensor


class AlphaSplitCoreTests(unittest.TestCase):
    def test_isotropic_path_reduces_to_uniform_linear_strain(self):
        lam = 40.0
        mu = 30.0
        stiffness = np.zeros((6, 6))
        stiffness[:3, :3] = lam
        np.fill_diagonal(stiffness[:3, :3], lam + 2.0 * mu)
        np.fill_diagonal(stiffness[3:, 3:], mu)
        compliance = np.linalg.inv(stiffness)
        bulk = lam + 2.0 * mu / 3.0

        path = compliance_weighted_path(compliance)

        np.testing.assert_allclose(
            path["normalized_direction"],
            np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            float(path["path_scale_1_per_GPa"]), 1.0 / (3.0 * bulk), places=12
        )
        self.assertAlmostEqual(
            float(path["volumetric_compliance_beta_1_per_GPa"]),
            1.0 / bulk,
            places=12,
        )

    def test_general_engineering_strain_uses_engineering_shear_convention(self):
        atoms = Atoms("He", cell=np.eye(3), scaled_positions=[[0, 0, 0]], pbc=True)
        eta = np.asarray([0.01, -0.01, 0.02, 0.04, 0.0, 0.0])
        strained = apply_engineering_strain_vector(atoms, eta)
        expected = np.eye(3) + strain_voigt_to_tensor(eta)
        np.testing.assert_allclose(strained.cell.array, expected)
        self.assertAlmostEqual(float(expected[1, 2]), 0.02)

    def test_effective_gamma_matches_full_compliance_contraction(self):
        compliance = np.diag([0.010, 0.012, 0.009, 0.020, 0.025, 0.018])
        compliance[0, 3] = compliance[3, 0] = 0.001
        compliance[1, 4] = compliance[4, 1] = -0.0005
        path = compliance_weighted_path(compliance)
        gamma_voigt = np.asarray(
            [
                [[1.0, -0.5, 2.0]],
                [[0.8, 1.5, -1.0]],
                [[1.2, 0.2, 0.5]],
                [[-2.0, 0.4, 0.1]],
                [[0.3, -1.0, 0.7]],
                [[0.0, 0.2, -0.4]],
            ]
        )
        direction = np.asarray(path["normalized_direction"])
        gamma_path = np.tensordot(direction, gamma_voigt, axes=(0, 0))
        recovered = effective_gamma_1_per_GPa(
            gamma_path, path["path_scale_1_per_GPa"]
        )
        direct = np.tensordot(
            compliance.T @ np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            gamma_voigt,
            axes=(0, 0),
        )
        np.testing.assert_allclose(recovered, direct, rtol=1.0e-12, atol=1.0e-15)

    def test_split_total_matches_full_six_component_response(self):
        compliance = np.diag([0.010, 0.012, 0.009, 0.020, 0.025, 0.018])
        compliance[0, 3] = compliance[3, 0] = 0.001
        compliance[1, 4] = compliance[4, 1] = -0.0005
        path = compliance_weighted_path(compliance)
        frequencies = np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 4.0]])
        weights = np.asarray([2.0, 1.0])
        gamma_voigt = np.asarray(
            [
                [[1.0, -0.5, 2.0], [0.2, 0.3, -0.7]],
                [[0.8, 1.5, -1.0], [1.0, -0.1, 0.4]],
                [[1.2, 0.2, 0.5], [-0.2, 0.7, 0.6]],
                [[-2.0, 0.4, 0.1], [0.4, -0.5, 1.2]],
                [[0.3, -1.0, 0.7], [0.1, 0.8, -0.4]],
                [[0.0, 0.2, -0.4], [-0.6, 0.2, 0.5]],
            ]
        )
        gamma_path = np.tensordot(
            np.asarray(path["normalized_direction"]), gamma_voigt, axes=(0, 0)
        )
        temperatures = np.asarray([100.0, 300.0])
        split, quality = compute_alpha_volume_split(
            temperatures,
            frequencies,
            gamma_path,
            weights,
            path["path_scale_1_per_GPa"],
            volume_A3=100.0,
            frequency_cutoff_THz=1.0e-4,
        )
        full, _ = compute_thermal_response(
            temperatures_K=temperatures,
            frequencies_THz=frequencies,
            gammas=gamma_voigt,
            weights=weights,
            compliance_1_per_GPa=compliance,
            volume_A3=100.0,
            frequency_cutoff_THz=1.0e-4,
        )
        np.testing.assert_allclose(
            split["alpha_volume_total_per_K"],
            full["alpha_volume_per_K"],
            rtol=1.0e-12,
            atol=1.0e-20,
        )
        self.assertLess(quality["split_identity_max_abs_error_per_K"], 1.0e-20)

    def test_positive_strain_gammas_can_make_negative_volume_contribution(self):
        compliance = np.zeros((6, 6))
        compliance[:3, :3] = np.asarray(
            [[1.0, -1.2, 0.0], [-1.2, 2.0, 0.0], [0.0, 0.0, 1.0]]
        )
        np.fill_diagonal(compliance[3:, 3:], 1.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(compliance) > 0.0))
        path = compliance_weighted_path(compliance)
        gamma_voigt = np.asarray([10.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        gamma_path = float(np.dot(path["normalized_direction"], gamma_voigt))
        effective = float(
            effective_gamma_1_per_GPa(
                np.asarray(gamma_path), path["path_scale_1_per_GPa"]
            )
        )
        self.assertTrue(np.all(gamma_voigt > 0.0))
        self.assertLess(effective, 0.0)

    def test_zero_tolerance_preserves_unresolved_signed_identity(self):
        frequencies = np.asarray([[1.0, 2.0, 3.0]])
        gamma_path = np.asarray([[2.0, -1.0, 1.0e-5]])
        response, quality = compute_alpha_volume_split(
            np.asarray([300.0]),
            frequencies,
            gamma_path,
            np.asarray([1.0]),
            path_scale_1_per_GPa=0.01,
            volume_A3=50.0,
            frequency_cutoff_THz=1.0e-4,
            effective_gamma_zero_tolerance_1_per_GPa=1.0e-6,
        )
        self.assertEqual(quality["positive_mode_count"], 1)
        self.assertEqual(quality["negative_mode_count"], 1)
        self.assertEqual(quality["unresolved_mode_count"], 1)
        self.assertGreater(float(response["unresolved_alpha_fraction"][0]), 0.0)
        self.assertLessEqual(
            float(response["ratio_lower_bound"][0]),
            float(response["ratio_abs_negative_to_positive"][0]),
        )
        self.assertGreaterEqual(
            float(response["ratio_upper_bound"][0]),
            float(response["ratio_abs_negative_to_positive"][0]),
        )
        reconstructed = (
            response["alpha_volume_positive_per_K"]
            + response["alpha_volume_negative_per_K"]
            + response["alpha_volume_unresolved_signed_per_K"]
        )
        np.testing.assert_allclose(reconstructed, response["alpha_volume_total_per_K"])

    def test_exact_zero_modes_are_resolved_zero_not_uncertain_heat_capacity(self):
        response, quality = compute_alpha_volume_split(
            np.asarray([300.0]),
            np.asarray([[1.0, 2.0]]),
            np.asarray([[1.0, 0.0]]),
            np.asarray([1.0]),
            path_scale_1_per_GPa=0.01,
            volume_A3=50.0,
            frequency_cutoff_THz=1.0e-4,
            effective_gamma_zero_tolerance_1_per_GPa=0.0,
        )
        self.assertEqual(quality["exact_zero_mode_count"], 1)
        self.assertEqual(quality["unresolved_mode_count"], 0)
        self.assertAlmostEqual(float(response["unresolved_heat_capacity_fraction"][0]), 0.0)

    def test_strain_comparison_rejects_changed_ratio(self):
        temperatures = np.asarray([100.0, 300.0])
        primary = {
            "temperatures_K": temperatures,
            "alpha_volume_positive_per_K": np.asarray([10.0e-6, 12.0e-6]),
            "alpha_volume_negative_per_K": np.asarray([-5.0e-6, -6.0e-6]),
            "alpha_volume_total_per_K": np.asarray([5.0e-6, 6.0e-6]),
            "ratio_abs_negative_to_positive": np.asarray([0.5, 0.5]),
        }
        fallback = {
            "temperatures_K": temperatures,
            "alpha_volume_positive_per_K": np.asarray([10.0e-6, 12.0e-6]),
            "alpha_volume_negative_per_K": np.asarray([-1.0e-6, -1.2e-6]),
            "alpha_volume_total_per_K": np.asarray([9.0e-6, 10.8e-6]),
            "ratio_abs_negative_to_positive": np.asarray([0.1, 0.1]),
        }
        report = compare_split_responses(primary, fallback)
        self.assertEqual(report["status"], "strain_derivative_unresolved")
        self.assertIn("alphaV_negative_not_converged", report["reasons"])
        self.assertIn("alphaV_ratio_not_converged", report["reasons"])

    def test_target_temperature_must_be_exactly_on_grid(self):
        response = {
            "temperatures_K": np.asarray([290.0, 310.0]),
            "alpha_volume_positive_per_K": np.asarray([1.0e-6, 1.0e-6]),
            "alpha_volume_negative_per_K": np.asarray([-0.5e-6, -0.5e-6]),
            "alpha_volume_unresolved_signed_per_K": np.zeros(2),
            "alpha_volume_unresolved_absolute_bound_per_K": np.zeros(2),
            "alpha_volume_total_per_K": np.asarray([0.5e-6, 0.5e-6]),
            "ratio_abs_negative_to_positive": np.asarray([0.5, 0.5]),
            "excluded_heat_capacity_fraction": np.zeros(2),
            "unresolved_heat_capacity_fraction": np.zeros(2),
        }
        with self.assertRaisesRegex(ValueError, "target_temperature_not_on_grid"):
            summarize_at_temperature(response, 300.0)

    def test_zero_positive_denominator_has_explicit_status(self):
        response, _ = compute_alpha_volume_split(
            np.asarray([300.0]),
            np.asarray([[1.0, 2.0]]),
            np.asarray([[-1.0, -2.0]]),
            np.asarray([1.0]),
            path_scale_1_per_GPa=0.01,
            volume_A3=50.0,
            frequency_cutoff_THz=1.0e-4,
        )
        summary = summarize_at_temperature(response, 300.0)
        self.assertEqual(summary["ratio_raw_status"], "denominator_zero_negative_present")
        self.assertIsNone(summary["ratio_abs_negative_to_positive"])


if __name__ == "__main__":
    unittest.main()
