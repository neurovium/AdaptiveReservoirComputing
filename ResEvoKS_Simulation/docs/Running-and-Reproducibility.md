# Running & Reproducibility

## Install

Requires **Julia ≥ 1.9**. From `julia/ResEvoKS_Simulation`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

If you ever edit `Project.toml`, run `Pkg.resolve()` then `Pkg.instantiate()`.

Run the tests:

```bash
julia --project=. -t auto -e 'using Pkg; Pkg.test()'
# or directly:
julia --project=. -t 3 test/runtests.jl
```

All 186 assertions should pass. The suite covers the KS solver (including
long-horizon stability), reservoir construction, the readout feature map, the
error metrics and fitness, the `.mat` IO contract, a single end-to-end
evaluation, and a tiny parallel GA run (with a reproducibility check).

---

## Run the experiment

### Full paper scale

```bash
julia --project=. -t auto scripts/run_optimization.jl
```

Defaults: 10 runs × 300 individuals × 101 generations, KS `N=64, d=22, Δt=0.15,
nstep=100000`, train 70000 / predict 2000. **This is heavy** (days to weeks
depending on cores; see runtime notes below).

### Quick functional run

```bash
RESEVO_NRUNS=1 RESEVO_NSTEP=20000 RESEVO_TRAIN=14000 RESEVO_PREDICT=2000 \
RESEVO_POP=40 RESEVO_GENS=15 \
julia --project=. -t auto scripts/run_optimization.jl
```

All knobs are environment variables (see the [README](../README.md#reproducing-the-paper-scale-experiment) table).

### Programmatic use

```julia
using ResEvoKS_Simulation
model = KSModelParams(N=64, d=22.0, tau=0.25, nstep=20_000)
data  = solve_ks(random_initial_condition(model.N), model)
sp    = make_run_dirs("results", 1)
dp    = DataParams(train_length=14_000, predict_length=2_000)
res   = optimize_reservoirs(data, dp, sp;
            settings=GASettings(population_size=40, max_generations=15))
```

---

## Threads vs. BLAS

The GA evaluates its population across **Julia threads**. Start Julia with
threads enabled:

```bash
julia -t auto      # or: JULIA_NUM_THREADS=8 julia
```

Each evaluation internally uses **multithreaded BLAS** (the ridge solve, the big
matrix products). Running both at full width oversubscribes the CPU. The driver
script therefore calls `BLAS.set_num_threads(1)` when `Threads.nthreads() > 1`,
so the parallelism comes from evaluating many individuals at once. If you call
`optimize_reservoirs` yourself in a multithreaded session, do the same:

```julia
using LinearAlgebra
Threads.nthreads() > 1 && BLAS.set_num_threads(1)
```

For a *single* large reservoir evaluation (e.g. the demo), the opposite is
better: one Julia thread, BLAS using all cores.

### Why KrylovKit, not ARPACK?

The spectral-radius step needs the dominant eigenvalue. MATLAB used `eigs`
(ARPACK). ARPACK wraps a Fortran library with **shared, non-reentrant internal
state**: calling it concurrently from the parallel GA segfaults the process.
This port uses `KrylovKit.eigsolve`, a pure-Julia, thread-safe Arnoldi
iteration, which returns the same dominant eigenvalue without the hazard. (The
*analysis* code may still use ARPACK for the Laplacian spectrum, where it runs
single-threaded.)

---

## Reproducibility

Stochastic components are seeded:

- **KS initial condition** — `random_initial_condition(N; rng)`. The driver uses
  `StableRNG(seed + run_index)` per run.
- **Reservoir + input weights** — each GA individual `(seed, gen, j)` gets its
  own `StableRNG`, so its reservoir is deterministic regardless of how threads
  interleave.
- **GA operators** — selection/crossover/mutation draw from a single master
  `StableRNG(settings.seed)`.

`StableRNG` is used (not the default RNG) so streams are **stable across Julia
versions**. Consequently, a given `GASettings.seed` reproduces the same best
genome and the same population trajectory (the test suite asserts this).

Two unavoidable caveats:

1. **KS chaos.** Even with a fixed IC, different machines/BLAS/FFT builds can
   diverge a chaotic KS trajectory pointwise after long times. The scientific
   conclusions are statistical/structural and robust to this; exact-trajectory
   reproduction is not expected across platforms.
2. **Floating-point reductions.** Threaded BLAS may reorder sums; the readout
   solution can differ in the last bits. This does not affect `J` meaningfully.

---

## Runtime & memory expectations

Per individual, cost is dominated by:

- **State collection:** `train_length` steps of an `N×N` sparse mat-vec plus an
  `N×n_u` dense mat-vec → `O(train_length · (nnz(A) + N·n_u))`.
- **Ridge solve:** forming `X Xᵀ` (`O(N²·train_length)`) and solving an `N×N`
  system (`O(N³)`). For `N` up to 3000 and `train_length` 70000 this is the
  expensive part; expect seconds-to-minutes per large individual.

Memory per large individual is dominated by the `N × train_length` state matrix
(e.g. `3000 × 70000` doubles ≈ 1.6 GB). With `T` threads, `T` such matrices may
be live at once — size your `RESEVO_POP`/threads to available RAM, or reduce
`train_length`. To cut disk usage, restrict which generations are saved via
`GASettings(save_generations = 0:10:100)` (the analysis only needs a stratified
subset of generations).

## Output format (`.mat` vs `.jld2`)

Saved individuals and the per-run dataset can be written as MATLAB `.mat`
(default), native Julia `.jld2`, or both — set `RESEVO_FORMAT=mat|jld2|both`
(run script) or `GASettings(save_format=:jld2)` / `save_individual(…; format=…)`
programmatically. Use `.mat` for the existing `original_code/analysis` scripts;
use `.jld2` to stay entirely in Julia (it stores the real `ReservoirParams` /
`PredictionError` structs and the sparse `A` with no flattening). Read either
back uniformly with `load_individual(path)` / `load_dataset(path)`. Choosing
`:both` roughly doubles disk use.

---

## Output location

```
results/RUN<k>/
├── Log/        # esn_log.txt — timestamped run log (timing + failures)
├── matfiles/   # data.{mat,jld2} + one <gen>_<N>_<deg>_<rad>_<sig>.{mat,jld2} per individual
└── Figs/       # reserved for figures
```

Point the analysis scripts in `original_code/analysis` at `results/` (their
`dataDir`) to reproduce the figures.

## Run log

Each run writes `RUN<k>/Log/esn_log.txt` (the Julia analogue of the MATLAB
`diary`): a timestamped, flushed log of the config and, per generation, the
wall-clock `eval`/`save` time, best/mean/median `J`, and valid/failed/saved
counts, plus a line per failed individual (genome + error). Because every line is
flushed, the file is complete up to the last step even if the process crashes —
so over many runs/generations you can trace exactly where and when something
broke. The same per-generation timing is in `GAResult.history`. Use the
`logfile=` keyword of `optimize_reservoirs` to redirect it (`nothing` =
console-only).
