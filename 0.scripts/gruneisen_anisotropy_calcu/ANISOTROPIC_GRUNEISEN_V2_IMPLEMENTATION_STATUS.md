# Anisotropic Gruneisen v2 implementation status

Date: 2026-07-17

This file records the implemented state corresponding to
`ANISOTROPIC_GRUNEISEN_V2_DESIGN.md`.  Legacy scripts and legacy result files
remain unchanged.

## Implemented files

- `gruneisen_v2_core.py`
  - complete 6x6 stiffness/compliance parsing and validation;
  - engineering Voigt convention `xx yy zz yz xz xy`;
  - six Cartesian engineering-strain generators;
  - automatic diagonal supercell selection with minimum length;
  - root-axis to elastic-frame lattice mapping;
  - stable mode heat capacity;
  - six-component Gruneisen thermal integrals;
  - full compliance coupling, Cartesian thermal-expansion tensor, directional
    projection, volumetric trace, hydrostatic/deviatoric decomposition, and
    `F_ani(T)`;
  - temperature-wide effective-isotropy screen;
  - SHA256 input fingerprint and strict JSON output.
- `run_gruneisen_thermal_expansion_v2.py`
  - `--preflight-only` without MatterSim import;
  - authoritative input contract using `elastic/POSCAR` and
    `elastic/ELASTIC_TENSOR`;
  - MatterSim-1M and float64 defaults;
  - fixed identity primitive matrix;
  - fixed-cell internal-coordinate relaxation for all twelve strained states;
  - finite-displacement force constants without MatterSim's legacy
    `primitive_matrix="auto"` wrapper;
  - Phonopy 4.3.1 native degenerate perturbation through `GruneisenMesh`;
  - full-mesh calculation with crystal mesh symmetry disabled;
  - strained imaginary-mode diagnostics;
  - versioned outputs under `gruneisen_aniso_1M_v2/`;
  - no writes to root or `elastic/ELASTIC_TENSOR`.
- `batch_gruneisen_thermal_expansion_v2.py`
  - interleaved NTE/PTE discovery;
  - material list, per-root limit, stable chunking, resume, force and dry-run;
  - fast in-process full-dataset preflight;
  - per-material logs and CSV summary.
- `test_gruneisen_v2_core.py`
  - engineering-shear convention;
  - row-vector cell transformation;
  - automatic supercell selection;
  - elastic positive-definiteness;
  - hydrostatic Gruneisen response and zero `F_ani`.
- `select_v2_representatives.py`
  - selects one valid low-cost NTE and PTE material for every crystal system.

## Runtime verified

Preferred WSL environment:

```text
Python       3.12.13
MatterSim    1.2.5
Phonopy      4.3.1
ASE          3.28.0
NumPy        2.2.6
PyTorch      2.12.0+cu130
model        mattersim-v1.0.0-1M.pth
```

The core unit tests pass in both Windows Python and the WSL MatterSim runtime.

## Full preflight result

All 342 material directories were audited.

```text
total                       342
preflight warning/usable    330
preflight blocked            12
axis mapping successful     338
axis mapping failed           4
```

The 12 blocked materials have non-positive-definite elastic tensors.  The four
axis-mapping failures are:

```text
NTE 124.CaW2O7
NTE 4756.CuSeO4
NTE 5651.Ga2Se3
PTE 3671.CoO2
```

`124.CaW2O7` is already blocked by its elastic tensor.  The other three can
still produce Cartesian components, volumetric alpha, and `F_ani(T)`, but their
reported crystallographic `alpha_a/alpha_b/alpha_c` will remain unavailable
until the structure mapping is resolved.

The durable copied summary is:

```text
v2_preflight_summary.csv
```

The timestamped source batch summary is retained under:

```text
batch_logs_v2/20260717_170702_165164/batch_summary.csv
```

Every material also contains its own versioned `preflight_report.json`,
`run_metadata.json`, reference structures, mapping report, and
`elastic_tensor_used.dat`.

## End-to-end smoke verification

Two deliberately non-production smoke directories were used:

- `PTE_materials/230.Cu/gruneisen_aniso_1M_v2_smoke_cu`
- `PTE_materials/BeO-mp-2542/gruneisen_aniso_1M_v2_smoke_beo`

The Cu run verified the complete thirteen-state force-constant and six-component
Gruneisen execution path.  Its `1x1x1` smoke supercell contains only zero
acoustic modes and is not a physical result.

The BeO smoke run additionally verified finite optical modes, full tensor
projection, strict JSON, strained imaginary-mode diagnostics, and the
temperature-wide effective-isotropy screen.  With the deliberately reduced
smoke parameters, the calculated `F_ani` was about 0.0024--0.0039 from 100 to
300 K, and `alpha_a` and `alpha_b` agreed closely as expected for hexagonal BeO.
These numbers validate code symmetry and data flow only; they must not be cited
because the smoke calculation used a `1x1x1` supercell and skipped internal
relaxation.

## Representative validation set

`v2_representative_materials.csv` and `v2_representative_materials.txt` contain
14 materials: one NTE and one PTE example for each of the seven crystal systems.
They are selected from valid preflight results using the smallest estimated v2
supercell atom count within each class/system.

## Remaining work before production classification

1. Run the 14 representative materials using production supercells, internal
   relaxation, 0.01 A displacement and the 30x30x30 mesh.
2. Run the same representatives at strains 0.005 and 0.0025 and implement the
   final convergence decision from the paired results.
3. Establish reference-force/stress, imaginary-mode, excluded heat-capacity and
   extreme-gamma acceptance thresholds from those representative calculations.
4. Resolve or formally retain the three usable axis-mapping failures as
   Cartesian/volumetric-only cases.
5. Add provenance for the imported elastic calculations.  All imported elastic
   directories currently lack `calculation_metadata.json`.
6. Only after the representative convergence study, launch the 330-material
   production batch.

## Recommended next command pattern

Run from WSL with the latest MatterSim environment.  Quote the project path
because its directory name contains `&`.

```bash
root='/mnt/d/9.Project/9.NTE&PTE_dataset/9.gruneisen_parameter'
py=/home/gmchen/anaconda3/envs/mattersim/bin/python

"$py" "$root/0.scripts/gruneisen_anisotropy_calcu/batch_gruneisen_thermal_expansion_v2.py" \
  --materials-file "$root/0.scripts/gruneisen_anisotropy_calcu/v2_representative_materials.txt" \
  --model-size 1M \
  --dtype float64 \
  --device cuda \
  --strain 0.005 \
  --resume
```

Do not start this production-parameter command until representative runtime and
memory cost have been checked on one small material.
