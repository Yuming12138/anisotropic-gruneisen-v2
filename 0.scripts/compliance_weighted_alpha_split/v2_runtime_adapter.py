#!/usr/bin/env python
"""Import the verified anisotropic-v2 runtime without duplicating path setup."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent / "gruneisen_anisotropy_calcu"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from gruneisen_v2_core import (  # noqa: E402,F401
    ANGSTROM3_TO_M3,
    GPA_TO_PA,
    HYDROSTATIC_VOIGT,
    V2Parameters,
    choose_supercell_matrix,
    compute_thermal_response,
    input_fingerprint,
    json_safe,
    mode_heat_capacity_J_K,
    read_elastic_tensor,
    rows_to_text_table,
    runtime_versions,
    sha256_file,
    stable_json_hash,
    strain_voigt_to_tensor,
    structure_axis_mapping,
    validate_elastic_tensor,
    write_json,
)
from run_gruneisen_thermal_expansion_v2 import (  # noqa: E402,F401
    DiagnosticGruneisenMesh,
    batch_fixed_cell_relax,
    calculate_force_constants,
    choose_device,
    fixed_cell_relax,
    reference_force_stress_report,
    resolve_model_path,
)


V2_CORE_PATH = V2_DIR / "gruneisen_v2_core.py"
V2_RUNNER_PATH = V2_DIR / "run_gruneisen_thermal_expansion_v2.py"
