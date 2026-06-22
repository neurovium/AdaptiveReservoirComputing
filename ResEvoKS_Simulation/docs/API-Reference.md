# API Reference

All names below are exported by `using ResEvoKS_Simulation`. Full docstrings live with each
function (`?name` at the REPL). Signatures show keyword arguments after `;`.

---

## KS solver — `KuramotoSivashinsky.jl`

### `KSModelParams(; N=64, d=22.0, tau=0.25, nstep=100_000)`
KS integration parameters (grid points, domain length, time step, #steps).

### `random_initial_condition(N; amplitude=0.6, rng=StableRNG(1234)) -> Vector`
The `amplitude·uniform[−1,1]` initial field (MATLAB `0.6*(-1+2*rand)`).

### `solve_ks(init, p::KSModelParams) -> Matrix{Float64}`
Integrate KS via ETDRK4 (real-FFT formulation). Returns the `N × nstep`
(space × time) trajectory.

---

## Activations — `Activation.jl`

### `tanh_activation(x)`
Elementwise `tanh` (the activation used in all paper runs).

### `generalized_logistic(g, k, n, lb, ub) -> Function`
Parametric generalized-logistic activation (legacy; `tanh`-like with
`(2,2,1,−1,1)`).

---

## Reservoir construction — `Reservoir.jl`

### `ReservoirParams(; num_inputs, radius, degree, N, sigma, beta)`
Decoded hyperparameters; field names match the MATLAB `resparams` struct.

### `generate_reservoir(size, degree; rng=StableRNG(0)) -> (A, C)`
Sparse uniform-`(0,1)` recurrent matrix `A` (expected degree `degree`) and its
boolean mask `C = A .> 0`.

### `spectral_radius(A; tol=1e-9, rng=StableRNG(0)) -> Float64`
Dominant eigenvalue magnitude `ρ(A)` via thread-safe `KrylovKit.eigsolve`
(dense fallback for `n ≤ 20`).

### `scale_spectral_radius!(A, radius; tol=1e-9, rng=StableRNG(0)) -> A`
Rescale `A` in place to spectral radius `radius`.

### `build_reservoir(N, degree, radius; rng=StableRNG(0), max_tries=50, tol=1e-9) -> A`
Generate + spectrally scale, retrying on eigensolver failure.

### `generate_input_weights(N, num_inputs, sigma; rng=StableRNG(0)) -> Matrix`
Disjoint per-channel input blocks, weights uniform on `[−sigma, sigma]`.

---

## Readout — `Readout.jl`

### `reservoir_layer(input, A, w_in, resparams, train_length, activation) -> states`
Teacher-forced state collection (`N × train_length`).

### `square_even_indices!(X) -> X`
In-place squaring of even-indexed rows (the bilinear feature map).

### `train_readout(resparams, states, target) -> w_out`
Ridge-regression readout (`num_inputs × N`). Mutates `states` with the squaring.

### `predict(x, A, w_in, resparams, w_out, predict_length, activation) -> (output, x)`
Autonomous closed-loop forecast (`num_inputs × predict_length`) and final state.

---

## Metrics — `Metrics.jl`

### `PredictionError`
Fields `NRMSE::Vector{Float64}`, `NMAE::Float64`.

### `compute_error(estimated, correct) -> PredictionError`
Per-channel NRMSE and the scalar NMAE (verbatim min–max convention).

### `composite_fitness(err; threshold=0.05) -> Float64`
`J = NMAE / #(NRMSE < threshold)`; `Inf` if no channel qualifies.

---

## IO — `IO.jl`

### `SaveParams(; codedir, matdir, figdir)`
Per-run output directories.

### `make_run_dirs(rootdir, run_index) -> SaveParams`
Create `<rootdir>/RUN<k>/{Log,matfiles,Figs}`.

### `individual_filename(gen, resparams::ReservoirParams) -> String`
The `<gen>_<N>_<degree>_<radius>_<sigma>` base name (six-decimal floats).

### `save_individual(matdir, gen, A, w_in, w_out, resparams, err, J; format=:mat) -> path`
Write the per-individual artifact. `format` ∈ `(:mat, :jld2, :both)` (see
`SAVE_FORMATS`). Returns the written path (the `.mat` path for `:both`).

### `load_individual(path) -> NamedTuple`
Read a `.mat` or `.jld2` individual back as `(; w_in, w_out, A, resparams, err,
J)`, with `resparams`/`err` reconstructed as structs in both cases.

### `save_dataset(matdir, data, model_params; format=:mat) -> path`
Write `data.{mat,jld2}` (`data` + `ModelParams`) in the chosen format(s).

### `load_dataset(path) -> NamedTuple`
Read `data.{mat,jld2}` back as `(; data, ModelParams)`.

### `SAVE_FORMATS`
The tuple `(:mat, :jld2, :both)` of valid `format` values.

---

## Run logging — `RunLog.jl`

### `open_run_logger(path; echo=true, t0=time()) -> RunLogger`
Open a timestamped log file at `path` (or `nothing` for console-only). `t0`
anchors the elapsed (`+Xs`) stamp.

### `logmsg(logger, msg)`
Write one `[timestamp | +elapsed s] msg` line (flushed; echoed to stdout if
`echo`).

### `close_logger(logger)`
Close the underlying file.

---

## Evaluation — `Evaluation.jl`

### `DataParams(; train_length=70_000, predict_length=2_000)`
Training/prediction window lengths.

### `decode_genome(genome, num_inputs) -> ReservoirParams`
Unpack `[radius, degree, size, sigma, beta]`; round integer genes; snap size to
a multiple of `num_inputs`.

### `EvalResult`
Fields `J, resparams, err, A, w_in, w_out`.

### `evaluate_individual(genome, data, dataparams; threshold=0.05, activation=tanh_activation, rng=StableRNG(0), return_artifacts=true) -> EvalResult`
Full build → train → predict → score for one genome.

---

## Optimization — `Optimization.jl`

### `GASettings(; …)`
GA configuration. Key fields (defaults reproduce the paper run):
`lb, ub, int_genes, population_size=300, max_generations=101, elite_fraction=0.05,
crossover_fraction=0.8, tournament_size=2, mutation_rate=0.1, mutation_scale=0.1,
reset_rate=0.1, blx_alpha=0.5, threshold=0.05, seed=20200501,
save_generations=:all, save_format=:mat`. A copy-constructor
`GASettings(s; field=val)` overrides selected fields.

### `GAResult`
Fields `best_genome, best_J, history, final_population, elapsed_seconds`. The
`history` is a vector of `(gen, best, mean, median, n_finite)`.

### `optimize_reservoirs(data, dataparams, saveparams; settings=GASettings(), activation=tanh_activation, verbose=true, save_to_disk=true, logfile=:default) -> GAResult`
Run the parallel generational GA, saving every evaluated individual to disk and
writing a timestamped run log. `logfile=:default` → `<codedir>/esn_log.txt`;
pass a path to override, or `nothing` for console-only. `GAResult.history`
records per-generation `(gen, best, mean, median, n_finite, n_failed, seconds)`.
