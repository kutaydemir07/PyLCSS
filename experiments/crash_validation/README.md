# PyLCSS Crash Qualification

This folder contains the pre-ML qualification workflow for the OpenRadioss
crash backend. A successful solver exit is necessary but is not sufficient:
the workflow qualifies measurement channels, numerical quality, convergence,
repeatability, material traceability, and physical/reference correlation.

## Standards basis

- SAE J211/1:2022 — electronic instrumentation for impact tests.
- ISO 6487:2015 + Amd 1:2017 — road-vehicle impact-test instrumentation.
- ISO/TS 18571:2024 — objective comparison of non-ambiguous time histories.
- Altair Radioss computation/result-checking guidance — energy, mass,
  hourglass/contact energy, time-step evolution, and physical review.

PyLCSS provides a compatible signal-processing and traceability contract. It
does not claim that software post-processing certifies acquisition hardware,
nor does it label its transparent engineering correlation score as the
proprietary ISO/TS 18571 rating.

## Run

Baseline:

```powershell
.\.venv\Scripts\python.exe experiments\crash_validation\run_qualification.py --mode baseline
```

Reprocess a saved baseline after measurement/QC code changes:

```powershell
.\.venv\Scripts\python.exe experiments\crash_validation\run_qualification.py `
  --mode baseline-report
```

Full numerical qualification from scratch:

```powershell
.\.venv\Scripts\python.exe experiments\crash_validation\run_qualification.py `
  --mode numerical `
  --mesh-sizes 8 7 6 5 4 3.5 `
  --time-step-scales 0.9 0.67 0.5 `
  --repeat-count 3
```

The time-step study disables mass scaling and uses OpenRadioss
`/DT/NODA/STOP` with Courant scale factors 0.90, 0.67, and 0.50. This produces
genuinely smaller integration steps without changing mass. `/DT/NODA/CST`
remains available for production acceleration, but added mass is a separately
gated approximation and is not used to claim temporal convergence.

To extend an existing mesh study and reuse an already-qualified 8 mm baseline:

```powershell
.\.venv\Scripts\python.exe experiments\crash_validation\run_qualification.py `
  --mode mesh `
  --mesh-sizes 8 7 6 5 4 3.5 `
  --baseline-mesh-size 8
```

To replace only timestep and repeatability cases on a converged 3.5 mm mesh:

```powershell
.\.venv\Scripts\python.exe experiments\crash_validation\run_qualification.py `
  --mode temporal `
  --reference-mesh-size 3.5 `
  --time-step-scales 0.9 0.67 0.5 `
  --repeat-count 3 `
  --resume-existing
```

`--resume-existing` accepts only normally terminated runs whose mesh hash,
mass-scaling policy, and timestep scale match. Progress is saved after each
case.

Reapply the latest quality/report schema to the saved solver histories without
rerunning the expensive matrix:

```powershell
.\.venv\Scripts\python.exe experiments\crash_validation\run_qualification.py `
  --mode report
```

Physical correlation:

```powershell
.\.venv\Scripts\python.exe experiments\crash_validation\run_qualification.py `
  --mode report `
  --benchmark-csv experiments\crash_validation\benchmark_template.csv `
  --material-validation experiments\crash_validation\material_validation_template.json
```

Replace both templates with traceable measured data. The templates intentionally
have `status: fail`; they prevent an unvalidated material card or invented
benchmark curve from being reported as qualified.

## Required benchmark columns

- `time_ms`
- `force_kN`
- `acceleration_g`
- `displacement_mm`
- `force_displacement_force_kN`

The matching JSON sidecar records test ID, specimen, instrumentation, filter
classes, coordinate/sign convention, source, and calibration traceability.
The physical gate additionally requires:

- `status: pass`, approval identity/date, source report, and file hashes;
- geometry revision, material lot, thickness, boundary condition, and at least
  three physical test replicates;
- impact speed and impactor mass within 1% of the simulated condition;
- force, acceleration, and displacement sensor/calibration identifiers;
- force/acceleration CFC values matching the simulation and acquisition rates
  at least 10 times the declared CFC;
- all three comparisons: force-time, crash pulse, and force-displacement.

## Definition of done before ML

`qualification_report.json` must show `status: pass` and
`ml_ground_truth_eligible: true`. That requires:

1. Every selected solver run passes its own crash quality gate.
2. Mesh and time-step convergence meet the configured metric tolerances.
3. Repeat runs meet the numerical coefficient-of-variation limits.
4. The exact material lot/rate curves have a passing coupon validation dossier.
5. Force-displacement and crash-pulse histories correlate with a traceable
   physical or approved reference benchmark.

Until all five are true, results may be used for development and diagnostics,
but not labelled “validated ML ground truth.”

## Current reference evidence

The saved local qualification report was generated with six mesh levels
(8, 7, 6, 5, 4, and 3.5 mm), three mass-neutral timestep levels, and three
independent executions of the finest timestep case:

- Mesh 4.0 to 3.5 mm: every monitored response changed by less than 5%.
- Mean actual timestep: 0.2125 to 0.1580 to 0.1178 microseconds.
- Timestep 0.67 to 0.50: every monitored response changed by less than 0.14%.
- Added mass: 0% for the temporal study.
- Repeatability coefficient of variation: effectively 0%.
- All 11 saved runs pass numerical solver and consistency gates.

Therefore `numerical_status` is `pass`. Overall/physical status intentionally
remains `fail` until a controlled material dossier and a matching physical
component test are supplied. This is a release gate, not a software error.

Altair's official
[RD-E 1700 Box Beam](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/box_beam_example_r.htm)
example uses the same core numerical review axes: mesh sensitivity,
force-displacement, total energy, and hourglass energy. It is a solver-method
verification reference; it does not replace correlation of the actual PyLCSS
component, material lot, and test boundary conditions.
