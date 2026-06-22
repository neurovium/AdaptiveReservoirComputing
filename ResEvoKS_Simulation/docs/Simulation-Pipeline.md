# Simulation Pipeline

This page describes the end-to-end data flow, what each source file/function
does, the genetic-algorithm driver, and the on-disk output contract.

---

## End-to-end data flow

```
                ┌─────────────────────────────────────────────────────────┐
                │ scripts/run_optimization.jl  (← KS64D_prepDataAndRun.m)  │
                └─────────────────────────────────────────────────────────┘
                                          │  for each run k:
        ┌─────────────────────────────────┼──────────────────────────────────┐
        ▼                                  ▼                                   ▼
 random_initial_condition          solve_ks (ETDRK4)                  make_run_dirs
        │                                  │                                   │
        └──────────────► data (N × nstep) ─┘                          RUN k/{Log,matfiles,Figs}
                                          │
                                          ▼
                ┌─────────────────────────────────────────────────────────┐
                │ optimize_reservoirs   (← KS64D_runOptimizePredESN.m)     │
                │   generation loop, parallel population evaluation        │
                └─────────────────────────────────────────────────────────┘
                                          │  per individual:
        ┌────────────────┬────────────────┼─────────────────┬────────────────┐
        ▼                ▼                ▼                 ▼                ▼
 decode_genome   build_reservoir  generate_input_weights  reservoir_layer  train_readout
        │                │                │                 │                │
        └─► resparams    └─► A            └─► W_in           └─► states ──────┘
                                                                  │
                                                                  ▼
                                                              predict (autonomous)
                                                                  │
                                                                  ▼
                                                          compute_error → J
                                                                  │
                                                                  ▼
                                                  save_individual → <gen>_<N>_<deg>_<rad>_<sig>.mat
```

---

## Source files (and their MATLAB origins)

### `src/KuramotoSivashinsky.jl`  ← `kuramoto_sivashinsky_solve.m`
ETDRK4 integrator for the KS PDE.
- `KSModelParams` — `N, d, tau, nstep`.
- `random_initial_condition(N; amplitude, rng)` — the `0.6·uniform[−1,1]` IC.
- `solve_ks(init, p)` — returns the `N × nstep` (space × time) trajectory.
  Uses real FFTs for stability (see [Mathematics](Mathematics.md)).

### `src/Activation.jl`  ← `ActFunc.m`, `generate_activationFunction.m`
- `tanh_activation` — the activation used in all paper runs.
- `generalized_logistic(g,k,n,lb,ub)` — the parametric logistic (legacy).
- `rescale_minmax` — MATLAB `rescale` helper.

### `src/Reservoir.jl`  ← `generate_reservoir.m` + inline construction
- `ReservoirParams` — decoded hyperparameters (`num_inputs, radius, degree, N,
  sigma, beta`), field names matching the MATLAB `resparams` struct.
- `generate_reservoir(size, degree; rng)` — sparse uniform `(0,1)` matrix `A`
  and mask `C`.
- `spectral_radius(A; rng)` — dominant `|λ|` via thread-safe KrylovKit.
- `scale_spectral_radius!(A, radius; rng)` — rescale to a target `ρ`.
- `build_reservoir(N, degree, radius; rng)` — generate + scale, retry on
  eigensolver failure (the MATLAB `try/catch` regenerate loop).
- `generate_input_weights(N, num_inputs, sigma; rng)` — disjoint `σ`-uniform
  input blocks.

### `src/Readout.jl`  ← `reservoir_layer.m`, `train.m`, `predict.m`
- `reservoir_layer(input, A, w_in, resparams, train_length, activation)` —
  teacher-forced state collection.
- `square_even_indices!(X)` — the bilinear feature map.
- `train_readout(resparams, states, target)` — ridge readout (mutates `states`
  with the squaring, exactly as MATLAB does).
- `predict(x, A, w_in, resparams, w_out, predict_length, activation)` —
  autonomous closed-loop forecast.

### `src/Metrics.jl`  ← `compute_error.m` + the `J = j1/j2` logic
- `PredictionError` — `NRMSE` (vector), `NMAE` (scalar).
- `compute_error(estimated, correct)` — both metrics, with the verbatim NMAE
  normalization.
- `composite_fitness(err; threshold)` — the composite `J` (`Inf` if no channel
  is below threshold).

### `src/IO.jl`  ← saving (`.mat` and/or `.jld2`) + directory scheme
- `SaveParams`, `make_run_dirs(rootdir, k)` — the `RUN k/{Log,matfiles,Figs}`
  layout.
- `individual_filename(gen, resparams)` — the `%d_%d_%d_%f_%f` base name.
- `save_individual(...; format=:mat)` — writes the variable set in `:mat`,
  `:jld2`, or `:both`.
- `save_dataset(matdir, data, model_params; format=:mat)` — writes
  `data.mat`/`data.jld2`.
- `load_individual(path)` / `load_dataset(path)` — read either format back
  uniformly (dispatch on extension), reconstructing the `ReservoirParams` /
  `PredictionError` structs.

### `src/RunLog.jl`  ← MATLAB `diary(...)` + `enable_logging`
- `RunLogger`, `open_run_logger(path; echo, t0)` — a tee logger writing a
  timestamped, flushed line to a file (and optionally the console).
- `logmsg(logger, msg)` — one `[timestamp | +elapsed s] msg` line.
- `close_logger(logger)` — release the file.

The GA driver uses this to write `RUN<k>/Log/esn_log.txt` (see below).

### `src/Evaluation.jl`  ← the `quickOptimizePredESN` objective body
- `DataParams` — `train_length, predict_length`.
- `decode_genome(genome, num_inputs)` — unpack + integer-snap.
- `evaluate_individual(genome, data, dataparams; …)` — full build → train →
  predict → score, returning an `EvalResult` (`J, resparams, err, A, w_in,
  w_out`).

### `src/Optimization.jl`  ← `KS64D_runOptimizePredESN.m` (the GA driver)
- `GASettings` — all GA knobs; defaults reproduce the paper run.
- `optimize_reservoirs(data, dataparams, saveparams; settings, …)` — the
  generational loop (next section).
- `GAResult` — best genome/`J`, per-generation history, final population.

### `scripts/run_optimization.jl`  ← `KS64D_prepDataAndRun.m`
Outer driver: for each run, make data, create dirs, run the GA. Configurable
via `RESEVO_*` environment variables.

### `scripts/demo_good_vs_bad.jl`  ← `KS-reservoir-demo/ks_reservoir.py`
GA-free schematic: contrast good/bad/awful reservoirs and save their
truth/prediction/difference fields.

---

## Genetic-algorithm driver

The driver implements a transparent **generational GA** whose loop we own. For
each generation `g = 0, 1, …, G−1`:

1. **Evaluate** the population in parallel (`Threads.@threads`). Each individual
   gets a deterministic per-`(seed, g, j)` RNG, so results are independent of
   thread scheduling and fully reproducible.
2. **Persist** every successfully built individual to
   `matdir/<g>_<N>_<deg>_<rad>_<sig>.mat` (subject to `save_generations`).
   Individuals that fail to build are skipped (matching the MATLAB
   regenerate/skip behavior).
3. **Record** per-generation fitness statistics (best/mean/median, count valid).
4. **Reproduce** (unless this was the last generation): elitism + tournament
   selection + BLX-α crossover + Gaussian/reset mutation, with box-clamping and
   integer snapping of the degree and size genes.

### Why no generation-passing hack?

MATLAB's parallel `ga` evaluates the population on worker processes that don't
share the client workspace, so the original code had to write the generation
counter to a shared `.mat` file and read it back inside the objective
(`storeGenNumber`/`retrieveGenNumber`). Because this Julia driver owns the
generation loop, the index `g` is in scope at evaluation time and is passed
straight to `save_individual`. The workaround is unnecessary and is omitted.

### Parallelism

The population is evaluated across Julia threads. The heavy linear algebra
inside each evaluation (state collection, ridge solve) already uses
multithreaded BLAS, so the driver script pins BLAS to one thread when Julia is
started with more than one thread, avoiding oversubscription. See
[Running & Reproducibility](Running-and-Reproducibility.md).

### Run logging & timing

Every run writes `RUN<k>/Log/esn_log.txt` via `RunLog`. Each line is timestamped
and stamped with elapsed seconds, and is **flushed immediately** so the log is
complete on disk even if the process is killed mid-run. The driver logs:

- a header with the config (population, generations, threads, seed, format) and
  data dimensions/bounds;
- one line per generation with the wall-clock **eval** and **save** times, the
  best/mean/median `J`, and the **valid/failed/saved** counts;
- one line per **failed** individual, with its genome and the error message, so
  breakage in a long campaign is traceable to the exact generation/individual;
- a footer with the total time and best genome; an aborting error is logged
  before being re-thrown.

The same per-generation timing is returned in `GAResult.history` (the `seconds`
and `n_failed` fields). Redirect with the `logfile=` keyword (`nothing` =
console-only).

---

## Output contract

For every evaluated individual, an artifact is written in the chosen
format(s) — `:mat` (default), `:jld2`, or `:both`:

- **Name:** `<generation>_<N>_<degree>_<radius>_<sigma>.{mat,jld2}`
  (e.g. `12_2048_6_0.873000_0.412000.mat`; the floats use six decimals, exactly
  like MATLAB `sprintf('%f', …)`).
- **Variables:** `w_in` (dense `N×n_u`), `w_out` (dense `n_u×N`), `A` (sparse
  `N×N`), `resparams` (`num_inputs, radius, degree, N, sigma, beta`),
  `err` (`NRMSE` length-`N` vector, `NMAE` scalar), `J` (scalar).

In `.mat`, `resparams`/`err` are MATLAB structs (Dicts); in `.jld2`, they are the
actual `ReservoirParams`/`PredictionError` Julia structs. The `.mat` variable
names are exactly what the analysis scripts in `original_code/analysis` read, so
they run unchanged on Julia-generated data. The `.jld2` form lets you stay fully
in Julia (read with `load_individual`). A `data.mat`/`data.jld2` per run holds
the KS field and `ModelParams`.

### Choosing the format

| via | how |
|-----|-----|
| run script | `RESEVO_FORMAT=mat\|jld2\|both julia … scripts/run_optimization.jl` |
| GA settings | `optimize_reservoirs(…; settings=GASettings(save_format=:both))` |
| single call | `save_individual(…; format=:jld2)`, `save_dataset(…; format=:both)` |
