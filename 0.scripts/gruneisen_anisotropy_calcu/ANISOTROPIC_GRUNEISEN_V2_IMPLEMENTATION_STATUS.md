# Anisotropic Gruneisen v2 implementation status

Date: 2026-07-18

This file records the production-verified state corresponding to
`ANISOTROPIC_GRUNEISEN_V2_DESIGN.md`. Legacy scripts and result files remain unchanged.

## Production scope

The workflow computes the six-component symmetric thermal-expansion tensor within an
anisotropic Gruneisen quasi-harmonic approximation. It reports Cartesian Voigt components,
directional expansion, volumetric expansion, hydrostatic/deviatoric response and `F_ani(T)`.

This is not a full anisotropic free-energy-minimization QHA over independently optimized lattice
parameters at every temperature.

## Verified environment

```text
Python       3.12.13
MatterSim    1.2.5 with the Data -> M3GNetData batch-converter patch
PyTorch      2.8.0+cu126
Phonopy      4.3.1
ASE          3.29.0
NumPy        2.5.1
SciPy        1.18.0
pymatgen     2026.5.4
device       CUDA
dtype        float64
model        MatterSim 1M
```

Use `/home/chenguangming/miniconda3/envs/mattersim125/bin/python`.

## Production algorithm

- `run_gruneisen_thermal_expansion_v2.py`
  - validates structure identity, elastic stability, elastic provenance and axis mapping;
  - uses a minimum-length supercell, `h=0.005`, `0.01 A` displacement and a `20x20x20` mesh;
  - relaxes the twelve strained states at fixed cell;
  - supports fixed-cell `BatchRelaxer` with a 1024-atom cap;
  - keeps finite-displacement force evaluation serial because force batching changed sensitive
    Gruneisen results;
  - writes full Cartesian, directional, volumetric and Gruneisen-integral outputs.
- `run_gruneisen_production_v2.py`
  - accepts a stable primary result directly;
  - runs `h=0.0025` when soft-mode or extreme-gamma gates request a strain check;
  - compares `h=0.005` and `h=0.0025` at 100 K;
  - for a strain-converged fallback, recomputes the response on `24x24x24` from saved force
    constants and requires mesh convergence;
  - writes an in-progress or final `production_decision.json` before and after every stage;
  - returns exit code 2 for scientific rejection or unresolved convergence.
- `batch_gruneisen_thermal_expansion_v2.py`
  - supports production mode, chunking, resume, independent logs and incremental CSV summaries;
  - reports strain and mesh convergence separately.

## MatterSim batching guard

### Local MatterSim 1.2.5 batching patch

The WSL MatterSim 1.2.5 environment includes the one-line fix from upstream
PR #166:

```python
return M3GNetData(**args)
```

instead of `return Data(**args)` in `datasets/utils/converter.py`. The exact
two-structure PyG regression test passes: the second graph's
`three_body_indices` are offset by the first graph's `num_bonds`. The original
installed file is retained as `converter.py.pre_pr166_20260717.bak`, and the
environment contains `mattersim/LOCAL_PATCH_PR166.json` with both SHA256 values.

`patch_mattersim_pr166.py --check-only` can verify the environment. The v2
runner additionally sets `batch_converter=False` explicitly; its force calls
remain single-structure even if the local dependency is later reinstalled.

## Quality gates

Initial readiness thresholds:

```text
reference residual force                 <= 1e-3 eV/A
reference maximum absolute stress        <= 0.1 GPa
excluded heat-capacity fraction          <= 0.05
reference zero/imaginary modes           <= 3 before fallback
strain-induced imaginary modes           none before fallback
maximum absolute Gruneisen parameter     <= 500 before fallback
```

Strain convergence, `h=0.005` versus `h=0.0025`:

```text
normalized integral difference           <= 10%
alpha_V relative difference at 100 K     <= 5%
or alpha_V absolute difference           <= 0.5 micro/K
directional maximum difference           <= 1.0 micro/K
```

Mesh convergence, `20x20x20` versus `24x24x24`, over all temperatures:

```text
normalized integral difference           <= 2%
alpha_V maximum absolute difference      <= 0.5 micro/K
directional maximum difference           <= 1.0 micro/K
F_ani maximum absolute difference        <= 0.01
```

## Mesh-gate evidence

The seven-crystal-system benchmark is retained at:

```text
/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation/
bench_agv2_7systems_20260718/compare_762138.out
```

Five stable systems had `20^3` to `24^3` 300 K volumetric differences no larger than 0.0032%.
The original strain-only production logic nevertheless misclassified Eu2CuO4 as ready because
its strain derivative converged while its mesh did not. The adaptive dense-mesh gate fixes this
loophole.

Fallback mesh results:

```text
ZnO
  integral difference      0.0089%
  alpha_V max difference   0.00057 micro/K
  directional difference  0.00073 micro/K
  F_ani difference         0.000050
  decision                 converged

Eu2CuO4
  integral difference      28.17%
  alpha_V max difference   19.38 micro/K
  directional difference  12.31 micro/K
  F_ani difference         0.0518
  decision                 mesh_convergence_unresolved
```

Final end-to-end jobs using the current code:

```text
job 762180  BeTe     2:59  ready
job 762181  ZnO      4:49  ready_with_fallback; strain and mesh converged
job 762182  Eu2CuO4  5:39  mesh_convergence_unresolved; expected exit 2
```

## BatchRelaxer regression

BatchRelaxer was compared against sequential BFGS while phonon forces remained serial.

```text
BeTe
  maximum position difference       3.79e-7 A
  integral difference               0.057%
  alpha_V maximum difference        < 1e-12 micro/K
  decision                          ready in both modes

V3S4
  maximum position difference       4.47e-5 A
  integral difference               0.078%
  alpha_V maximum difference        0.0053 micro/K
  decision                          ready in both modes

NbPO5 primary/fallback
  maximum position difference       3.64e-6 / 2.76e-6 A
  integral difference               0.40% / 0.37%
  alpha_V maximum difference        0.0060 / 0.0404 micro/K
  decision                          strain_derivative_unresolved in both modes
```

Evidence files are under:

```text
/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation/
bench_agv2_relaxmode_20260718/
```

Batch relaxation is numerically safe at the tested tolerance. It does not guarantee an
end-to-end speedup because dense mesh construction dominates some materials.

## Failure and resume behavior

- Primary and fallback subprocess failures now persist `failed_primary_stage` or
  `failed_fallback_stage` with the command and return code.
- Batch CSV preserves scientific rejection statuses and runner exit code 2.
- Fingerprints include source files, model, runner, core, selected supercell, execution mode,
  batch atom cap and dependency versions.
- Consistent completed production commands skip primary, fallback and mesh subprocesses.

The final three-stage ZnO batch resume completed in `0.33 s`. Failure-persistence evidence is:

```text
/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation/
bench_agv2_failure_persist_v2_20260718/
```

## Full preflight

Job 762183 reran the complete 342-directory preflight with the current code in `1:24`.

```text
total directories                    342
production-runnable                  327
blocked union                         15
non-positive-definite elastic tensor  12
axis-mapping failure                   4
overlap of both failures               1
```

The authoritative summary is:

```text
/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation/
bench_agv2_preflight_final_20260718/batch_logs/20260718_043903_855617/batch_summary.csv
```

The derived launch and exclusion lists are:

```text
/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation/
bench_agv2_preflight_final_20260718/production_ready_327.txt
/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation/
bench_agv2_preflight_final_20260718/production_blocked_15.csv
```

The positive-definite axis-only failures are `0114`, `0130` and `0208`; `0018` has both failure
types. All 342 elastic directories contain `elastic/calculation_metadata.json`.

## Production command

```bash
base=/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation
scripts=$base/anisotropic-gruneisen-v2/0.scripts/gruneisen_anisotropy_calcu
py=/home/chenguangming/miniconda3/envs/mattersim125/bin/python

"$py" "$scripts/batch_gruneisen_thermal_expansion_v2.py" \
  --roots "$base/run_20260717_batch1024_all10/results" \
  --materials 0236 \
  --production \
  --python "$py" \
  --result-subdir gruneisen_aniso_1M_v2_prod \
  --model /home/chenguangming/2.model/mattersim-v1.0.0-1M.pth \
  --device cuda \
  --dtype float64 \
  --mesh 20 20 20 \
  --dense-mesh 24 24 24 \
  --batch-relax-atom-cap 1024 \
  --resume \
  --log-dir "$base/agv2_production_logs"
```

Use matching `--chunk-count N` and distinct `--chunk-index 0..N-1` for parallel Slurm jobs. Do
not include the 15 blocked material IDs.

## Final validation

```text
15 tests passed
all Python scripts compile
all shell scripts pass bash -n
git diff --check passes
changed and new Python files have no lines longer than 100 columns
```

The full 327-material production campaign has not been launched.
