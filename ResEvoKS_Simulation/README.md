# ResEvoKS_Simulation — Evolutionary Reservoir Computing for Kuramoto–Sivashinsky Chaos

Julia simulation suite for the paper

> **Evolutionary Optimization Reveals Structural Constraints on Reservoir
> Architecture for Spatiotemporal Chaos**, N. Dehghani.

This package evolves echo-state **reservoir computers** to predict the
spatiotemporal chaos of the **Kuramoto–Sivashinsky (KS)** equation. A genetic
algorithm searches reservoir *construction* hyperparameters (size, connectivity
degree, spectral radius, input scaling, ridge regularization); each candidate is
built, trained with a ridge readout, and scored by how long and how accurately
it forecasts the KS field in closed-loop (autonomous) mode.

It is a faithful, modular Julia port of the original MATLAB `64D-KS-Sim` suite.
The per-individual output files are **bit-compatible** with the existing
analysis pipeline (same filenames, same `.mat` variables), so the analysis
scripts run unchanged on Julia-generated data.

-  **What the original code does** → [`REPORT_original_code.md`](REPORT_original_code.md)
-  **The mathematics + per-file reference** → [`docs/`](docs/) wiki
  ([Home](docs/Home.md) · [Mathematics](docs/Mathematics.md) ·
  [Pipeline](docs/Simulation-Pipeline.md) · [API](docs/API-Reference.md) ·
  [Running](docs/Running-and-Reproducibility.md))

---

## Changes from the MATLAB 

1. **Stable KS solver.** The MATLAB/Python ETDRK4 integrator uses a *complex*
   FFT and sits on a numerical knife's edge: roundoff breaks the Hermitian
   symmetry of the real field and the integration may blow up around `t ≈ 360`
   (the Python demo can dodge this by switching `numpy.fft → scipy.fft`). This port
   uses **real FFTs (`rfft`/`irfft`)**, which enforce that symmetry by
   construction — the *same* ETDRK4 scheme, but stable to 100k+ steps on any FFT
   backend. See [Mathematics § KS solver](docs/Mathematics.md#kuramotosivashinsky-solver).

2. **No generation-passing hack.** MATLAB's parallel `ga` shared the generation
   counter through a `.mat` file (because workers don't see the client
   workspace). This port controls the generational loop directly, so the
   generation index is known exactly at evaluation time and every individual is
   saved correctly tagged — no workaround needed. See
   [Pipeline § GA driver](docs/Simulation-Pipeline.md#genetic-algorithm-driver).

---

## Installation

Requires **Julia ≥ 1.9**. From this directory (`julia/ResEvoKS_Simulation`):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs the dependencies (FFTW, Arpack, MAT, Evolutionary, StableRNGs, …)
into the package's own environment.

Run the test suite to confirm everything works:

```bash
julia --project=. -t auto -e 'using Pkg; Pkg.test()'
```

---

## Quick start (small, runs in seconds)

```julia
using ResEvoKS_Simulation

# 1. Generate a (short) KS trajectory:  space × time = 64 × 20000
model = KSModelParams(N=64, d=22.0, tau=0.25, nstep=20_000)
data  = solve_ks(random_initial_condition(model.N), model)

# 2. Output directories  results/RUN1/{Log,matfiles,Figs}
sp = make_run_dirs("results", 1)

# 3. Training / prediction windows
dp = DataParams(train_length=14_000, predict_length=2_000)

# 4. Evolve a small population for a few generations
settings = GASettings(population_size=20, max_generations=8)
result   = optimize_reservoirs(data, dp, sp; settings=settings)

@show result.best_J result.best_genome
```

Every evaluated reservoir is written to `results/RUN1/matfiles/` as
`<gen>_<N>_<degree>_<radius>_<sigma>.mat`.

### Evaluate a single reservoir (no GA)

```julia
genome = [0.9, 6.0, 2048.0, 0.5, 1e-4]   # [radius, degree, size, sigma, beta]
res = evaluate_individual(genome, data, dp)
@show res.J res.resparams.N count(<(0.05), res.err.NRMSE)
```

---

## Reproducing the paper-scale experiment

The full experiment is heavy (300 individuals × 101 generations × several runs,
reservoirs up to 3000 nodes trained on 70k samples). The driver script exposes
all knobs through environment variables:

```bash
# full scale (paper defaults), using all cores:
julia --project=. -t auto scripts/run_optimization.jl

# a smaller, faster configuration:
RESEVO_NRUNS=1 RESEVO_NSTEP=20000 RESEVO_TRAIN=14000 RESEVO_PREDICT=2000 \
RESEVO_POP=40 RESEVO_GENS=15 \
julia --project=. -t auto scripts/run_optimization.jl
```

| Variable | Meaning | Paper default |
|----------|---------|---------------|
| `RESEVO_ROOTDIR` | output root directory | `results` |
| `RESEVO_NRUNS`   | independent GA runs | `10` |
| `RESEVO_N`       | KS grid points | `64` |
| `RESEVO_D`       | KS domain length | `22.0` |
| `RESEVO_TAU`     | KS time step Δt | `0.15` |
| `RESEVO_NSTEP`   | KS steps generated | `100000` |
| `RESEVO_TRAIN`   | training samples | `70000` |
| `RESEVO_PREDICT` | prediction samples | `2000` |
| `RESEVO_POP`     | GA population size | `300` |
| `RESEVO_GENS`    | GA generations (incl. gen 0) | `101` |
| `RESEVO_SEED`    | master RNG seed | `20200501` |
| `RESEVO_FORMAT`  | output format: `mat`, `jld2`, or `both` | `mat` |

See [Running & Reproducibility](docs/Running-and-Reproducibility.md) for guidance
on threads vs. BLAS, memory, and runtime.

### Schematic demo (good vs. bad vs. awful reservoir)

```bash
julia --project=. scripts/demo_good_vs_bad.jl
```

Saves truth/prediction/difference fields to `demo_output/demo_good_vs_bad.mat`
for plotting (the analogue of the Python `KS-reservoir-demo`).

---

## Repository layout

```
julia/ResEvoKS_Simulation/
├── Project.toml              package + dependencies
├── README.md                 this file
├── REPORT_original_code.md   anatomy of the original MATLAB/Python code
├── src/
│   ├── ResEvoKS_Simulation.jl           top-level module (re-exports the API)
│   ├── KuramotoSivashinsky.jl  ETDRK4 KS solver            (← kuramoto_sivashinsky_solve.m)
│   ├── Activation.jl         activation functions          (← ActFunc.m, …)
│   ├── Reservoir.jl          reservoir + input construction(← generate_reservoir.m)
│   ├── Readout.jl            states / train / predict      (← reservoir_layer/train/predict.m)
│   ├── Metrics.jl            NRMSE, NMAE, composite J       (← compute_error.m)
│   ├── IO.jl                 MAT/JLD2 saving + loaders/dirs
│   ├── RunLog.jl             timestamped run logger         (← diary/enable_logging)
│   ├── Evaluation.jl         per-individual objective       (← quickOptimizePredESN)
│   └── Optimization.jl       genetic-algorithm driver       (← KS64D_runOptimizePredESN.m)
├── scripts/
│   ├── run_optimization.jl   end-to-end GA driver           (← KS64D_prepDataAndRun.m)
│   └── demo_good_vs_bad.jl   GA-free schematic demo         (← ks_reservoir.py)
├── test/runtests.jl          unit + integration tests
└── docs/                     wiki (mathematics + reference)
```

---

## Output data contract

For every evaluated reservoir, an artifact is written, named

```
<generation>_<N>_<degree>_<radius>_<sigma>.<ext>   e.g. 12_2048_6_0.873000_0.412000.mat
```

containing the variables

| variable | type | description |
|----------|------|-------------|
| `w_in`   | dense `N × num_inputs` | input weight matrix |
| `w_out`  | dense `num_inputs × N` | trained ridge readout |
| `A`      | sparse `N × N` | spectral-radius-scaled recurrent matrix |
| `resparams` | struct | `num_inputs, radius, degree, N, sigma, beta` |
| `err`    | struct | `NRMSE` (length-`N` vector), `NMAE` (scalar) |
| `J`      | scalar | composite fitness (lower is better) |

### Two output formats — `.mat` and/or `.jld2`

The on-disk format is selectable (`RESEVO_FORMAT`, or `GASettings(save_format=…)`):

- **`:mat`** (default) — MATLAB v5 `.mat`. **Bit-compatible** with the original
  pipeline: the variable names above are exactly what the analysis scripts in
  `original_code/analysis` read, so they work on Julia-generated data unchanged.
- **`:jld2`** — native Julia `.jld2` (HDF5-based). Stores the *actual* Julia
  objects (sparse `A`, and the `ReservoirParams`/`PredictionError` structs), so
  you can keep the entire workflow inside Julia with no MAT round-trip and no
  struct-flattening/precision loss.
- **`:both`** — write both `.mat` and `.jld2`.

Read either format back uniformly with `load_individual(path)` (returns
`(; w_in, w_out, A, resparams, err, J)`); the per-run dataset is `data.<ext>`,
read back with `load_dataset(path)`.

### Run log (timing & failure tracing)

Each run writes a timestamped, flushed log to `RUN<k>/Log/esn_log.txt` — the
Julia analogue of the MATLAB `diary`. It records the run config and, per
generation, the wall-clock `eval`/`save` time, best/mean/median `J`, and
valid/failed/saved counts; every failed individual is logged with its genome and
error, and an aborting error is recorded before re-throwing. Because each line is
flushed, the log survives a crash — useful for tracing where a long campaign
broke. Per-generation timing is also available in `result.history`.

```
[2026-06-20 23:29:32 | +      0.1s] ==== ResEvoKS_Simulation GA run started ====
[2026-06-20 23:29:32 | +      0.4s] config: pop=300 gens=101 threads=8 seed=20201501 format=mat
[2026-06-20 23:29:53 | +     20.5s] gen   0/100  eval=  18.2s save=  0.9s  best=1.7e-1 mean=4.0e-1 median=3.5e-1  valid=287/300 failed=0 saved=300
...
[..................| +  31.3s] ==== GA finished in 31.3 s. best J = ... ====
```

Set `logfile=` on `optimize_reservoirs` to redirect it (or `nothing` for
console-only).

```julia
# evolve and store everything natively in Julia:
optimize_reservoirs(data, dp, sp; settings=GASettings(save_format=:jld2))
ind = load_individual("results/RUN1/matfiles/12_2048_6_0.873000_0.412000.jld2")
ind.resparams.N, ind.J          # ind.resparams is a ReservoirParams struct
```

---

## Citation

If you use this code, please cite the accompanying paper (see the repository's
top-level metadata). Original MATLAB/Python and this Julia port by **Nima Dehghani**.
