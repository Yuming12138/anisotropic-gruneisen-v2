# Implementation status

Date: 2026-07-21

Implemented:

- direct compliance-weighted mixed strain path;
- arbitrary six-component Cartesian engineering strain application;
- Phonopy native direct-path degenerate perturbation;
- per-mode positive, negative, and unresolved volumetric contributions;
- exact split-conservation audit;
- strict root/elastic phase check;
- non-destructive preflight and input fingerprints;
- single-material runner;
- h/h/2 production wrapper with shared reference force constants;
- batch dispatcher and incremental CSV summary;
- synthetic scientific tests and preflight/cache contract tests.

Verified environment:

```text
Python   /home/gmchen/anaconda3/envs/mattersim/bin/python
Phonopy  4.3.1
```

Current verification:

```text
test_alpha_split_core.py      10 tests passed
test_alpha_split_contract.py   5 tests passed
```

Not yet completed:

- a real MatterSim end-to-end representative material calculation;
- empirical calibration of nonzero effective-gamma resolution thresholds;
- dense-mesh production fallback analogous to the full v2 24x24x24 check;
- manuscript plotting/collector integration.

Until representative production calculations pass, treat this directory as a
scientifically specified and unit-tested implementation candidate rather than a
fully production-validated dataset generator.
