# Anisotropic Gruneisen v2

Coordinate-consistent six-component strain-Gruneisen workflow for anisotropic
thermal expansion and temperature-dependent mechanism decomposition.

The implementation evaluates

\[
\boldsymbol\alpha(T)=\frac{1}{V}S\mathbf I(T),\qquad
I_j(T)=\frac{1}{N_q}\sum_{\mathbf q\nu}w_{\mathbf q}
C_{\mathbf q\nu}(T)\gamma^j_{\mathbf q\nu},
\]

using the full engineering-Voigt order
`xx yy zz yz xz xy`.  It additionally separates the hydrostatic and
deviatoric thermal driving forces and reports

\[
F_{\rm ani}(T)=
\frac{|\alpha_V^{\rm dev}(T)|}
{|\alpha_V^{\rm hyd}(T)|+|\alpha_V^{\rm dev}(T)|}.
\]

## Key safeguards

- authoritative reference structure: `elastic/POSCAR`;
- complete imported `elastic/ELASTIC_TENSOR`, never overwritten;
- six positive/negative Cartesian engineering strains;
- fixed-cell internal-coordinate relaxation;
- fixed `primitive_matrix = identity` and atom ordering;
- Phonopy native degenerate perturbation;
- full compliance coupling including normal-shear terms;
- actual crystallographic-axis projection;
- imaginary-mode, excluded-heat-capacity and extreme-gamma diagnostics;
- SHA256 input fingerprints and versioned result directories.

## Layout

The implementation is retained in its project-compatible path:

```text
0.scripts/gruneisen_anisotropy_calcu/
├── gruneisen_v2_core.py
├── run_gruneisen_thermal_expansion_v2.py
├── symmetrize_thermal_expansion.py
├── batch_gruneisen_thermal_expansion_v2.py
├── select_v2_representatives.py
├── patch_mattersim_pr166.py
├── test_gruneisen_v2_core.py
├── test_thermal_expansion_symmetry.py
├── ANISOTROPIC_GRUNEISEN_V2_DESIGN.md
└── ANISOTROPIC_GRUNEISEN_V2_IMPLEMENTATION_STATUS.md
```

Material directories are intentionally not part of this repository. Each
material is expected to contain:

```text
material/
├── POSCAR
└── elastic/
    ├── POSCAR
    └── ELASTIC_TENSOR
```

## Recommended runtime

The tested environment is Linux/WSL with:

- MatterSim 1.2.5;
- Phonopy 4.3.1;
- ASE 3.28.0;
- Python 3.12;
- MatterSim 1M checkpoint;
- float64 inference for the production validation stage.

### MatterSim 1.2.4/1.2.5 batching regression

MatterSim releases 1.2.4 and 1.2.5 are affected by the open upstream
[Issue #163](https://github.com/microsoft/mattersim/issues/163). The matching
[PR #166](https://github.com/microsoft/mattersim/pull/166) changes
`GraphConverter` to return `M3GNetData`, allowing PyG to offset
`three_body_indices` correctly when multiple structures are batched.

Check an environment with:

```bash
python 0.scripts/gruneisen_anisotropy_calcu/patch_mattersim_pr166.py --check-only
```

Apply the reviewed one-line patch to an affected installation with:

```bash
python 0.scripts/gruneisen_anisotropy_calcu/patch_mattersim_pr166.py
```

The helper creates a timestamped backup and runs the upstream two-structure
regression test. The v2 runner itself explicitly uses `batch_converter=False`
and evaluates displaced structures one at a time, so this particular regression
does not affect the v2 force loop.

## Tests

```bash
python 0.scripts/gruneisen_anisotropy_calcu/test_gruneisen_v2_core.py
```

## Preflight

```bash
python 0.scripts/gruneisen_anisotropy_calcu/run_gruneisen_thermal_expansion_v2.py \
  --material-dir /path/to/material \
  --preflight-only
```

Batch preflight:

```bash
python 0.scripts/gruneisen_anisotropy_calcu/batch_gruneisen_thermal_expansion_v2.py \
  --roots /path/to/NTE_materials /path/to/PTE_materials \
  --preflight-only
```

## Production calculation

```bash
python 0.scripts/gruneisen_anisotropy_calcu/run_gruneisen_thermal_expansion_v2.py \
  --material-dir /path/to/material \
  --model-size 1M \
  --device cuda \
  --dtype float64 \
  --strain 0.005 \
  --displacement 0.01 \
  --mesh 30 30 30 \
  --resume
```

Production calculations should begin with the representative convergence set,
including a second run at strain `0.0025`, before any full-dataset batch.

## Crystal-symmetry post-processing

After a production result has been selected, project the calculated Cartesian
thermal-expansion tensor onto the point group of the actual reference structure
(`reference/POSCAR`):

```bash
python 0.scripts/gruneisen_anisotropy_calcu/symmetrize_thermal_expansion.py \
  --material-dir /path/to/material
```

For a batch result root containing numbered material directories:

```bash
python 0.scripts/gruneisen_anisotropy_calcu/symmetrize_thermal_expansion.py \
  --results-root /path/to/results \
  --batch-report /path/to/symmetry_postprocess_summary.json
```

The operation is non-destructive. It preserves
`thermal_expansion_cartesian_raw.dat`, writes separate symmetrized Cartesian and
crystallographic-axis tables, and records the removed symmetry residual in
`thermal_expansion_symmetry_report.json`. If symmetry detection differs between
`symprec=0.005` and the stricter `0.001` audit, the stricter point group is used.
Large removed components are reported as warnings rather than silently accepted.

## Computational cost

This workflow is intentionally more expensive than a three-normal-strain
approximation. Each material needs one reference state and twelve strained
states. Every strained state undergoes internal relaxation and a finite-
displacement force-constant calculation. The number of displaced supercells
increases when strain lowers the symmetry.

As a practical planning estimate:

- about 2--5 times the cost of the previous three-component anisotropic
  workflow;
- commonly 10--30 or more times the cost of one standalone phonon calculation;
- low-symmetry structures and large automatically selected supercells can be
  substantially more expensive.

Use preflight and representative benchmarks to measure actual wall time before
allocating a full production batch.

## Status

The scientific core, batch preflight and end-to-end smoke path are implemented.
Production-parameter representative convergence calculations remain the next
stage. See the implementation-status document for details.
