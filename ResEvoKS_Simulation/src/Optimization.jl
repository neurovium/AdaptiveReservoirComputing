# =============================================================================
# Optimization.jl
# -----------------------------------------------------------------------------
# Genetic-algorithm optimization of reservoir construction hyperparameters.
#
# Julia port of the GA driver in `KS64D_runOptimizePredESN.m`.
#
# Design note
# ===========
# MATLAB's `ga` is parallelized over workers that do not share the client
# workspace, which is why the original code passes the generation number through
# a shared `.mat` file (the `storeGenNumber`/`retrieveGenNumber` hack). It also
# needs every evaluated individual saved to disk, tagged by generation.
#
# This Julia port implements a clean, transparent **generational GA** whose loop
# we control directly. That removes the file-passing workaround entirely (the
# generation index is known exactly at evaluation time) and lets us persist each
# individual's *full* artifact set (`w_in, w_out, A, resparams, err, J`) every
# generation — which is exactly what the downstream analysis pipeline consumes.
#
# The GA follows the standard real-coded stages described in the paper's Methods
# ("initial population … evaluate … select … crossover and mutation … repeat"):
#   * tournament selection with elitism,
#   * blend (BLX-α) crossover,
#   * Gaussian creep mutation + occasional uniform reset,
# with box constraints and integer handling for the degree and size genes
# (mirroring MATLAB's `IntCon = [2,3]`). `Evolutionary.jl` is listed as a
# dependency and provides equivalent operators if a library engine is preferred;
# see the wiki for how to swap it in.
#
# Parallelism: the population is evaluated with `Threads.@threads`. Start Julia
# with `-t auto` (or set `JULIA_NUM_THREADS`) to use multiple cores. Because the
# heavy linear algebra inside each evaluation already uses multithreaded BLAS,
# consider pinning BLAS to 1 thread (`BLAS.set_num_threads(1)`) when evaluating
# many individuals in parallel to avoid oversubscription (see the wiki).
# =============================================================================

module Optimization

using Random
using StableRNGs
using Printf
using Statistics
using Dates
using ProgressMeter

using ..Evaluation: DataParams, evaluate_individual, EvalResult, decode_genome
using ..Activation: tanh_activation
using ..IO: SaveParams, save_individual
using ..RunLog: RunLogger, open_run_logger, logmsg, close_logger, logpath

export GASettings, GAResult, optimize_reservoirs

# -----------------------------------------------------------------------------
# Search-space defaults (verbatim from KS64D_runOptimizePredESN.m)
#   gene 1: spectral radius      ρ ∈ [0.1, 1.0]
#   gene 2: connection degree    d ∈ [2, 10]      (integer)
#   gene 3: reservoir size       n ∈ [300, 3000]  (integer)
#   gene 4: input scaling        σ ∈ [0.1, 1.0]
#   gene 5: ridge regularization β ∈ [1e-4, 2e-4]
# -----------------------------------------------------------------------------
const DEFAULT_LB = [0.1, 2.0, 300.0, 0.1, 1e-4]
const DEFAULT_UB = [1.0, 10.0, 3000.0, 1.0, 2e-4]
const DEFAULT_INT_GENES = (2, 3)   # degree, size

"""
    GASettings

Configuration for the reservoir genetic algorithm. Defaults reproduce the paper
run (`population_size = 300`, `max_generations = 101`, the bounds above).

# Fields
- `lb`, `ub::Vector{Float64}`      : per-gene lower/upper bounds (length 5).
- `int_genes::Tuple`               : gene indices constrained to integers.
- `population_size::Int`           : individuals per generation (300).
- `max_generations::Int`           : number of generations incl. gen 0 (101).
- `elite_fraction::Float64`        : fraction carried over unchanged (0.05).
- `crossover_fraction::Float64`    : fraction of non-elite from crossover (0.8).
- `tournament_size::Int`           : tournament selection pressure (2).
- `mutation_rate::Float64`         : per-gene mutation probability (0.1).
- `mutation_scale::Float64`        : Gaussian creep σ as a fraction of range (0.1).
- `reset_rate::Float64`            : prob. a mutated gene is uniformly resampled (0.1).
- `blx_alpha::Float64`             : BLX-α crossover spread (0.5).
- `threshold::Float64`             : NRMSE threshold ε for fitness (0.05).
- `seed::Int`                      : master RNG seed for reproducibility.
- `save_generations`               : `:all` or a collection of generation indices
                                     whose individuals are written to disk.
- `save_format::Symbol`            : `:mat` (default), `:jld2`, or `:both` — the
                                     on-disk format(s) for saved individuals.
"""
Base.@kwdef struct GASettings
    lb::Vector{Float64}            = copy(DEFAULT_LB)
    ub::Vector{Float64}            = copy(DEFAULT_UB)
    int_genes::Tuple               = DEFAULT_INT_GENES
    population_size::Int           = 300
    max_generations::Int           = 101
    elite_fraction::Float64        = 0.05
    crossover_fraction::Float64    = 0.8
    tournament_size::Int           = 2
    mutation_rate::Float64         = 0.1
    mutation_scale::Float64        = 0.1
    reset_rate::Float64            = 0.1
    blx_alpha::Float64             = 0.5
    threshold::Float64             = 0.05
    seed::Int                      = 20200501
    save_generations               = :all
    save_format::Symbol            = :mat
end

"""
    GASettings(s::GASettings; kwargs...) -> GASettings

Copy-constructor: build a new `GASettings` from an existing one, overriding only
the named fields. Convenient for per-run variation, e.g.
`GASettings(base; seed = base.seed + kk)`.
"""
function GASettings(s::GASettings; kwargs...)
    overrides = Dict(kwargs)
    pick(name) = get(overrides, name, getfield(s, name))
    return GASettings(
        lb=pick(:lb), ub=pick(:ub), int_genes=pick(:int_genes),
        population_size=pick(:population_size), max_generations=pick(:max_generations),
        elite_fraction=pick(:elite_fraction), crossover_fraction=pick(:crossover_fraction),
        tournament_size=pick(:tournament_size), mutation_rate=pick(:mutation_rate),
        mutation_scale=pick(:mutation_scale), reset_rate=pick(:reset_rate),
        blx_alpha=pick(:blx_alpha), threshold=pick(:threshold), seed=pick(:seed),
        save_generations=pick(:save_generations), save_format=pick(:save_format))
end

"""
    GAResult

Outcome of an optimization run.

# Fields
- `best_genome::Vector{Float64}`     : best individual found.
- `best_J::Float64`                  : its composite fitness.
- `history::Vector{NamedTuple}`      : per-generation summary
                                       `(gen, best, mean, median, n_finite,
                                       n_failed, seconds)` — including the
                                       wall-clock `seconds` for that generation.
- `final_population::Matrix{Float64}`: last generation's genomes (5 × pop).
- `elapsed_seconds::Float64`         : wall-clock time.
"""
struct GAResult
    best_genome::Vector{Float64}
    best_J::Float64
    history::Vector{NamedTuple}
    final_population::Matrix{Float64}
    elapsed_seconds::Float64
end

# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

# Clamp a genome to box bounds and snap integer genes to the nearest integer.
function _repair!(g::AbstractVector{<:Real}, s::GASettings)
    @inbounds for i in eachindex(g)
        g[i] = clamp(g[i], s.lb[i], s.ub[i])
    end
    for i in s.int_genes
        g[i] = clamp(round(g[i]), s.lb[i], s.ub[i])
    end
    return g
end

# Draw one random genome uniformly within the bounds (integers snapped).
function _random_genome(rng::AbstractRNG, s::GASettings)
    g = s.lb .+ (s.ub .- s.lb) .* rand(rng, length(s.lb))
    return _repair!(g, s)
end

# Tournament selection: pick `tournament_size` at random, return the fittest.
function _tournament(rng::AbstractRNG, fitness::AbstractVector, k::Int)
    best = rand(rng, 1:length(fitness))
    for _ in 2:k
        c = rand(rng, 1:length(fitness))
        if fitness[c] < fitness[best]
            best = c
        end
    end
    return best
end

# BLX-α crossover of two parent genomes (real-coded blend).
function _blx_crossover(rng::AbstractRNG, p1, p2, s::GASettings)
    child = similar(p1)
    @inbounds for i in eachindex(p1)
        lo, hi = minmax(p1[i], p2[i])
        d = hi - lo
        child[i] = (lo - s.blx_alpha * d) + rand(rng) * (d + 2 * s.blx_alpha * d)
    end
    return child
end

# Gaussian creep mutation with occasional uniform reset, per gene.
function _mutate!(rng::AbstractRNG, g::AbstractVector, s::GASettings)
    @inbounds for i in eachindex(g)
        if rand(rng) < s.mutation_rate
            if rand(rng) < s.reset_rate
                g[i] = s.lb[i] + rand(rng) * (s.ub[i] - s.lb[i])      # uniform reset
            else
                span = s.ub[i] - s.lb[i]
                g[i] += randn(rng) * s.mutation_scale * span          # Gaussian creep
            end
        end
    end
    return g
end

# Produce the next generation from the current population and its fitness.
function _reproduce(rng::AbstractRNG, pop::Matrix{Float64},
                    fitness::Vector{Float64}, s::GASettings)
    n = s.population_size
    nelite = max(1, round(Int, s.elite_fraction * n))

    # Elitism: carry over the best `nelite` genomes unchanged.
    order = sortperm(fitness)                       # ascending (Inf last)
    newpop = Matrix{Float64}(undef, size(pop, 1), n)
    @inbounds for j in 1:nelite
        newpop[:, j] = pop[:, order[j]]
    end

    # Fill the rest by crossover (prob. crossover_fraction) or mutation-only.
    for j in (nelite + 1):n
        i1 = _tournament(rng, fitness, s.tournament_size)
        if rand(rng) < s.crossover_fraction
            i2 = _tournament(rng, fitness, s.tournament_size)
            child = _blx_crossover(rng, view(pop, :, i1), view(pop, :, i2), s)
        else
            child = copy(pop[:, i1])
        end
        _mutate!(rng, child, s)
        _repair!(child, s)
        newpop[:, j] = child
    end
    return newpop
end

# Decide whether a given generation's individuals should be written to disk.
_should_save(gen::Int, s::GASettings) =
    s.save_generations === :all || gen in s.save_generations

# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

"""
    optimize_reservoirs(data, dataparams, saveparams; settings=GASettings(),
                        activation=tanh_activation, verbose=true,
                        save_to_disk=true) -> GAResult

Run the reservoir GA on KS field `data` (`num_inputs × T`). For every generation
the full population is evaluated (in parallel over threads), every evaluated
individual is saved to `saveparams.matdir` as
`<gen>_<N>_<degree>_<radius>_<sigma>.mat` (subject to `settings.save_generations`),
and per-generation fitness statistics are recorded.

Mirrors `KS64D_runOptimizePredESN.m` but with explicit generation control.

# Arguments
- `data`       : KS measurements, `num_inputs × T`.
- `dataparams` : `DataParams` (train/predict lengths).
- `saveparams` : `SaveParams` (output directories).

# Keyword arguments
- `settings`     : `GASettings`.
- `activation`   : reservoir nonlinearity (default `tanh`).
- `verbose`      : echo per-generation progress to the console.
- `save_to_disk` : if `false`, skip writing individual files (e.g. for quick tests).
- `logfile`      : where to write the timestamped run log. `:default` (the
                   default) writes `<codedir>/esn_log.txt`; pass an explicit path
                   to override, or `nothing` for console-only (no file).

## Logging
A timestamped log is written (and flushed line-by-line) recording the run
config, and for each generation the wall-clock `eval`/`save` times, the
best/mean/median `J`, and the valid/failed/saved counts. Every failed individual
is logged with its genome and the error, and an aborting error is recorded before
re-throwing — so across a long multi-run campaign you can trace exactly where and
when a run broke. Per-generation timing is also returned in `GAResult.history`.

Returns a `GAResult`.
"""
function optimize_reservoirs(data::AbstractMatrix, dataparams::DataParams,
                             saveparams::SaveParams;
                             settings::GASettings=GASettings(),
                             activation=tanh_activation,
                             verbose::Bool=true,
                             save_to_disk::Bool=true,
                             logfile=:default)
    s = settings
    nvars = length(s.lb)
    @assert length(s.ub) == nvars "lb and ub must have the same length"

    master = StableRNG(s.seed)
    t_start = time()

    # --- run logger ---------------------------------------------------------
    # Default: a timestamped log in the run's Log/ directory. Pass
    # `logfile = nothing` for console-only, or an explicit path to override.
    logpath_arg = logfile === :default ?
        joinpath(saveparams.codedir, "esn_log.txt") : logfile
    logger = open_run_logger(logpath_arg; echo=verbose, t0=t_start)

    # Everything is wrapped so the log file is always closed, even on error —
    # and because every line is flushed, the log survives a crash mid-run.
    try
        logmsg(logger, "==== ResEvoKS_Simulation GA run started ====")
        logmsg(logger, @sprintf("config: pop=%d gens=%d threads=%d seed=%d format=%s",
                s.population_size, s.max_generations, Threads.nthreads(),
                s.seed, String(s.save_format)))
        logmsg(logger, @sprintf("data: num_inputs=%d T=%d  train=%d predict=%d",
                size(data, 1), size(data, 2), dataparams.train_length,
                dataparams.predict_length))
        logmsg(logger, @sprintf("bounds: lb=%s  ub=%s", string(s.lb), string(s.ub)))

        # --- initial population (generation 0) ------------------------------
        pop = Matrix{Float64}(undef, nvars, s.population_size)
        for j in 1:s.population_size
            pop[:, j] = _random_genome(master, s)
        end

        history = NamedTuple[]
        best_genome = copy(pop[:, 1])
        best_J = Inf

        for gen in 0:(s.max_generations - 1)
            fitness = fill(Inf, s.population_size)
            results = Vector{Union{Nothing,EvalResult}}(nothing, s.population_size)
            # Per-thread failure messages (each thread writes its own index → safe).
            fail_msgs = Vector{Union{Nothing,String}}(nothing, s.population_size)

            # Per-individual reproducible seeds (independent of thread scheduling).
            seeds = [hash((s.seed, gen, j)) % UInt32 for j in 1:s.population_size]

            # Time the (parallel) evaluation of the whole generation.
            eval_seconds = @elapsed begin
                Threads.@threads for j in 1:s.population_size
                    genome = pop[:, j]
                    local_rng = StableRNG(seeds[j])
                    try
                        res = evaluate_individual(genome, data, dataparams;
                                                  threshold=s.threshold,
                                                  activation=activation,
                                                  rng=local_rng,
                                                  return_artifacts=true)
                        fitness[j] = isfinite(res.J) ? res.J : Inf
                        results[j] = res
                    catch err
                        fitness[j] = Inf
                        results[j] = nothing
                        fail_msgs[j] = string(err)
                    end
                end
            end

            # --- persist every evaluated individual for this generation -----
            save_seconds = 0.0
            n_saved = 0
            if save_to_disk && _should_save(gen, s)
                save_seconds = @elapsed begin
                    for j in 1:s.population_size
                        res = results[j]
                        res === nothing && continue
                        save_individual(saveparams.matdir, gen, res.A, res.w_in,
                                        res.w_out, res.resparams, res.err, res.J;
                                        format=s.save_format)
                        n_saved += 1
                    end
                end
            end

            # --- bookkeeping -------------------------------------------------
            finite = filter(isfinite, fitness)
            gen_best = isempty(finite) ? Inf : minimum(finite)
            gen_mean = isempty(finite) ? Inf : mean(finite)
            gen_med  = isempty(finite) ? Inf : median(finite)
            n_failed = count(!isnothing, fail_msgs)
            gen_seconds = eval_seconds + save_seconds

            push!(history, (gen=gen, best=gen_best, mean=gen_mean, median=gen_med,
                            n_finite=length(finite), n_failed=n_failed,
                            seconds=gen_seconds))

            bidx = argmin(fitness)
            if fitness[bidx] < best_J
                best_J = fitness[bidx]
                best_genome = copy(pop[:, bidx])
            end

            # --- per-generation log line (timestamped, with timing) ---------
            logmsg(logger, @sprintf(
                "gen %3d/%d  eval=%6.1fs save=%5.1fs  best=%.4e mean=%.4e median=%.4e  valid=%d/%d failed=%d saved=%d",
                gen, s.max_generations - 1, eval_seconds, save_seconds,
                gen_best, gen_mean, gen_med, length(finite),
                s.population_size, n_failed, n_saved))

            # --- log each failure with its genome, to trace breakage --------
            if n_failed > 0
                for j in 1:s.population_size
                    fail_msgs[j] === nothing && continue
                    logmsg(logger, @sprintf("  FAIL gen %d idx %d genome=%s : %s",
                            gen, j, string(round.(pop[:, j], sigdigits=5)),
                            fail_msgs[j]))
                end
            end

            # --- produce next generation (unless this was the last) ---------
            if gen < s.max_generations - 1
                pop = _reproduce(master, pop, fitness, s)
            end
        end

        elapsed = time() - t_start
        logmsg(logger, @sprintf("==== GA finished in %.1f s. best J = %.6e ====",
                elapsed, best_J))
        logmsg(logger, @sprintf("best genome = %s", string(best_genome)))

        return GAResult(best_genome, best_J, history, pop, elapsed)
    catch err
        # Record the failure point before re-throwing, so the log shows where
        # things broke across a long multi-run campaign.
        logmsg(logger, "==== GA ABORTED with error: $err ====")
        rethrow()
    finally
        close_logger(logger)
    end
end

end # module Optimization
