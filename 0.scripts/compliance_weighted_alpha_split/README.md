# Compliance-weighted alpha split

This directory implements a direct positive/negative decomposition of the
volumetric thermal-expansion coefficient for anisotropic crystals.

It is deliberately separate from the six-component anisotropic-v2 result.  The
existing v2 workflow computes the total tensor, directional expansion,
hydrostatic/deviatoric response, and `F_ani`.  This workflow preserves per-mode
information long enough to calculate

\[
\alpha_V^+(T),\qquad \alpha_V^-(T),\qquad
R(T)=\frac{|\alpha_V^-(T)|}{\alpha_V^+(T)}.
\]

## Physical definition

Use engineering-Voigt order `xx yy zz yz xz xy` and define

\[
\mathbf e=(1,1,1,0,0,0)^T,\qquad
\mathbf d=S^T\mathbf e.
\]

For a mode with strain-Gruneisen vector
\(\boldsymbol\gamma_{\mathbf q\nu}\), its actual volumetric weight is

\[
\chi_{\mathbf q\nu}
=\mathbf e^TS\boldsymbol\gamma_{\mathbf q\nu}
=\mathbf d^T\boldsymbol\gamma_{\mathbf q\nu}.
\]

The code does not combine six independently diagonalized Gruneisen arrays by
band index.  Instead, it performs one direct central difference along the mixed
strain path \(\mathbf d\).  This avoids the non-unique band correspondence of
exactly degenerate subspaces.

The path is normalized so its largest absolute principal strain is one:

\[
r=\max|\operatorname{eig}(E(\mathbf d))|,\qquad
\mathbf u=\mathbf d/r,
\]

and the two strained states are

\[
\boldsymbol\eta^\pm=\pm h\mathbf u.
\]

Phonopy returns \(\gamma_{\rm path}=\mathbf u^T\boldsymbol\gamma\).
The physical compliance-weighted coefficient is restored as

\[
\chi=r\gamma_{\rm path}\quad [\mathrm{GPa}^{-1}].
\]

Each mode contributes

\[
\Delta\alpha_{V,\mathbf q\nu}(T)=
\frac{w_{\mathbf q}}{\sum w_{\mathbf q}}
\frac{C_{\mathbf q\nu}(T)}{V}
\frac{\chi_{\mathbf q\nu}}{10^9}.
\]

Positive and negative parts are sums over the sign of this final contribution,
not the sign of an individual strain-Gruneisen component.

## Safeguards

- `elastic/POSCAR` is the force-constant reference structure.
- The complete imported `elastic/ELASTIC_TENSOR` is read without modification.
- The root and elastic structures must pass a strict `StructureMatcher` phase
  consistency check.  The optimized structure is not silently substituted.
- The mixed strain is applied in Cartesian engineering-Voigt convention.
- Internal coordinates are relaxed at fixed cell.
- Each relaxed path state must remain structure-matched to its own unrelaxed
  state and below the configured mapped-displacement limit.
- Phonopy native degenerate perturbation is used on the direct mixed path.
- No universal large-gamma clipping is applied.
- A reporting floor (default `1e-3 micro/K`) censors numerically negligible
  positive or negative contributions without changing the calculated total.
- Imaginary modes, excluded heat capacity, unresolved heat capacity, and split
  conservation error are reported.
- Input/code/model fingerprints protect cached force constants.
- The production wrapper compares `h=0.005` and `h=0.0025`.  The reference
  force constants are fingerprinted independently of `h` and reused.

## Files

```text
compliance_weighted_alpha_split/
├── alpha_split_core.py
├── v2_runtime_adapter.py
├── run_compliance_weighted_alpha_split.py
├── run_alpha_split_production.py
├── compare_alpha_split_runs.py
├── batch_compliance_weighted_alpha_split.py
├── test_alpha_split_core.py
├── test_alpha_split_contract.py
├── COMPLIANCE_WEIGHTED_ALPHA_SPLIT_DESIGN.md
└── IMPLEMENTATION_STATUS.md
```

## Recommended environment

Use the WSL MatterSim environment verified with Phonopy 4.3.1:

```bash
PYTHON=/home/gmchen/anaconda3/envs/mattersim/bin/python
CODE=/mnt/d/9.Project/anisotropic-gruneisen-v2/0.scripts/compliance_weighted_alpha_split
```

## Preflight

```bash
$PYTHON $CODE/run_compliance_weighted_alpha_split.py \
  --material-dir /path/to/material \
  --result-subdir gruneisen_alpha_split_1M_v1/preflight \
  --preflight-only
```

## One strain amplitude

Use this for diagnostics or manual convergence studies:

```bash
$PYTHON $CODE/run_compliance_weighted_alpha_split.py \
  --material-dir /path/to/material \
  --result-subdir gruneisen_alpha_split_1M_v1/h0p005 \
  --model-size 1M \
  --strain 0.005 \
  --mesh 20 20 20 \
  --resume
```

## Production h/h/2 calculation

This is the recommended calculation for a reportable ratio:

```bash
$PYTHON $CODE/run_alpha_split_production.py \
  --python $PYTHON \
  --material-dir /path/to/material \
  --result-subdir gruneisen_alpha_split_1M_v1 \
  --model-size 1M \
  --primary-strain 0.005 \
  --fallback-strain 0.0025 \
  --mesh 20 20 20 \
  --resume
```

The stable selected result is copied to the production result root only when
the aggregate positive, negative, total, and ratio responses pass the strain
comparison at both the convergence temperature (default 100 K) and plotting
temperature (default 300 K), and the fallback calculation passes its quality
gates.

## Batch production

```bash
$PYTHON $CODE/batch_compliance_weighted_alpha_split.py \
  --python $PYTHON \
  --roots /path/to/NTE_materials /path/to/PTE_materials \
  --production \
  --model-size 1M \
  --chunk-count 8 \
  --chunk-index 0 \
  --resume
```

## Main outputs

`alpha_volume_split.dat` columns are:

```text
T_K
alphaV_positive_micro_per_K
alphaV_negative_micro_per_K
alphaV_negative_abs_micro_per_K
alphaV_unresolved_signed_micro_per_K
alphaV_unresolved_abs_bound_micro_per_K
alphaV_total_micro_per_K
ratio_abs_negative_to_positive
ratio_lower_bound
ratio_upper_bound
excluded_Cv_fraction
unresolved_Cv_fraction
unresolved_alpha_fraction
```

The file `alpha_volume_split_target.json` is the direct plotting input.  A
compatibility alias `alpha_volume_split_300K.json` is written only when the
requested target is exactly 300 K.  Use the target file's
`ratio_reportable` field; do not treat every finite raw ratio as production
quality.  A single-strain run always leaves this field false because strain
convergence has not been checked; the production wrapper may promote it after
both strain amplitudes converge.

## Tests

```bash
$PYTHON $CODE/test_alpha_split_core.py
$PYTHON $CODE/test_alpha_split_contract.py
```

The tests cover the isotropic limit, full six-component identity, compliance-
induced sign reversal, unresolved contributions, strain convergence, phase-safe
preflight, cache artifacts, and non-destructive elastic-tensor handling.
