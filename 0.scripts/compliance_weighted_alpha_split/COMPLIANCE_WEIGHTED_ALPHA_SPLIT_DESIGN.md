# Compliance-weighted volumetric alpha sign split: design

## 1. Scope

The goal is to calculate a physically meaningful

\[
R(T)=|\alpha_V^-(T)|/\alpha_V^+(T)
\]

for anisotropic crystals.  This is not a replacement for the full anisotropic
thermal-expansion tensor.  It is a separate per-mode decomposition of the final
volumetric response after elastic-compliance coupling.

## 2. Why the scalar sign rule is insufficient

For isotropic crystals,

\[
\alpha_V=\frac{1}{BV}\sum_m C_m\gamma_m,
\]

so the sign of scalar volume Gruneisen parameter \(\gamma_m\) determines the
sign of the mode contribution.

For a general crystal,

\[
\boldsymbol\alpha=\frac{1}{V}S\sum_m C_m\boldsymbol\gamma_m,
\]

and

\[
\Delta\alpha_{V,m}=\frac{C_m}{V}\mathbf e^TS\boldsymbol\gamma_m.
\]

The sign must therefore be evaluated after multiplication by the complete
elastic compliance tensor.  Positive individual \(\gamma_m^j\) values can
produce a negative final volumetric contribution.

## 3. Direct mixed path

Let

\[
\mathbf d=S^T\mathbf e.
\]

The target single-mode coefficient is

\[
\chi_m=\mathbf d^T\boldsymbol\gamma_m.
\]

Running six independent strain directions and combining equal array indices is
not valid for exactly degenerate modes because each perturbation can select a
different good basis in the degenerate subspace.

Instead, calculate one perturbation directly along \(\mathbf d\).  Define the
engineering strain tensor map \(E(\cdot)\) and

\[
r=\max|\operatorname{eig}(E(\mathbf d))|,
\qquad \mathbf u=\mathbf d/r.
\]

The applied strains are

\[
\boldsymbol\eta^\pm=\pm h\mathbf u.
\]

This normalization ensures the largest principal-strain magnitude is exactly
\(h\) for every material.  In the isotropic limit,

\[
\mathbf d=\frac{1}{3B}(1,1,1,0,0,0)^T,
\]

therefore \(r=1/(3B)\) and \(\mathbf u=(1,1,1,0,0,0)^T\), recovering an
ordinary hydrostatic linear strain.

Phonopy receives `delta_strain=2*h` and returns

\[
\gamma_{\rm path,m}=\mathbf u^T\boldsymbol\gamma_m.
\]

Recover

\[
\chi_m=r\gamma_{\rm path,m}.
\]

## 4. Thermal integration

For q-point weights \(w_q\),

\[
a_m(T)=
\frac{w_q}{\sum_qw_q}
\frac{C_m(T)}{V}
\frac{\chi_m}{10^9}.
\]

The factor \(10^9\) converts GPa\(^{-1}\) to Pa\(^{-1}\).  Then

\[
\alpha_V^+=\sum_{a_m>0}a_m,
\qquad
\alpha_V^-=\sum_{a_m<0}a_m.
\]

If a nonzero effective-gamma tolerance is requested, modes inside the interval
are reported separately as unresolved.  They are never silently discarded:

\[
\alpha_V=\alpha_V^++\alpha_V^-+\alpha_V^{\rm unresolved,signed}.
\]

The default tolerance is zero.  A nonzero tolerance must be justified by a
strain/model convergence study.

For nonzero tolerance, the code reports both an unresolved absolute alpha bound
and conservative lower/upper bounds on the ratio.  Quality gates use the alpha
bound itself, not only the heat-capacity fraction, because a small amount of
heat capacity can still carry a large Gruneisen weight.

Separately, the runner applies a default `1e-3 micro/K` reporting floor to the
300 K positive and negative magnitudes.  This does not alter any mode or total;
it only prevents a value such as `1e-6 micro/K` from being presented as a
resolved physical ratio.  The floor must be included in sensitivity analysis.

## 5. Structure contract

The authoritative calculation structure is `elastic/POSCAR`, because the
compliance tensor is expressed in that Cartesian frame.  The material-root
`POSCAR` supplies the dataset label and phase identity.

Required checks are:

1. equal reduced composition;
2. equal atom count;
3. successful `StructureMatcher.fit` without supercell matching;
4. stable, symmetric, positive-definite complete 6x6 stiffness tensor;
5. finite positive deformation determinants for both path states.

After fixed-cell internal relaxation, each strained state is additionally
matched to its own unrelaxed state.  Element order, fixed-cell identity,
`StructureMatcher.fit`, and the maximum periodic mapped displacement must pass;
otherwise the central difference is rejected as a possible branch change.

`elastic/POSCAR` is always the calculation reference because it is the
structure corresponding to `elastic/ELASTIC_TENSOR`.  The material-root
`POSCAR` and `opt/CONTCAR` are retained only for provenance checks; mismatches
are reported but do not replace or block the elastic reference.

## 6. Calculation stages

Single-strain runner:

1. preflight and input fingerprint;
2. reference force/stress audit;
3. reference finite-displacement force constants;
4. construct `cw_minus` and `cw_plus`;
5. fixed-cell internal-coordinate relaxation;
6. strained force constants;
7. direct Phonopy `GruneisenMesh`;
8. per-mode integration and sign split;
9. quality report and exact 300 K summary.

Production wrapper:

1. run \(h=0.005\);
2. reuse the reference force constants;
3. run \(h=0.0025\);
4. compare aggregate positive, negative, total, and ratio responses at both the
   convergence temperature and the target plotting temperature;
5. publish the smaller-strain result only if convergence and quality gates pass.

## 7. Quality gates

Hard failures include:

- root/elastic phase mismatch;
- unstable or ill-conditioned elastic tensor;
- excessive residual force or stress;
- failed fixed-cell internal relaxation;
- excessive excluded heat-capacity fraction;
- excessive unresolved heat-capacity fraction.

Strain-check triggers include:

- non-acoustic reference imaginary modes;
- strain-induced imaginary modes;
- extreme direct path Gruneisen parameters.

Production reporting additionally requires h/h/2 convergence.  A finite raw
ratio from a single strain is not automatically reportable.

## 8. Output contract

Each single-strain result contains:

```text
reference/
work/strain_0/
work/cw_minus/
work/cw_plus/
preflight_report.json
effective_strain_path.json
elastic_tensor_used.dat
effective_gruneisen_mesh.npz
alpha_volume_split.dat
alpha_volume_split_target.json
alpha_volume_split.png
qha_vs_gruneisen_thermal_expansion.png  # when QHA alpha_V data exist
plot_metadata.json
quality_report.json
run_metadata.json
calculation_complete.json
```

The production root additionally contains:

```text
primary_h0p005/
fallback_h0p0025/
strain_convergence.json
production_decision.json
production_complete.json  # only when accepted
```

Both plots show the volumetric coefficient.  The Gruneisen curve uses the
Cartesian engineering-Voigt selector `e=(1,1,1,0,0,0)`, and the QHA curve is
read from Phonopy's volumetric `thermal_expansion.dat`.  Thus, including for a
triclinic crystal, the compared quantity is
`alpha_V=trace(alpha_Cartesian)=alpha_xx+alpha_yy+alpha_zz`; non-orthogonal
lattice-vector expansion rates are never summed as if they were Cartesian
diagonal tensor components.

## 9. Interpretation rule

The isotropic and anisotropic ratios can share the same final mathematical
meaning - positive and negative contributions to \(\alpha_V\) - only after the
anisotropic modes are split by \(\mathbf e^TS\boldsymbol\gamma_m\).  They still
originate from different elastic physics and should be marked and fitted
separately in the manuscript.
