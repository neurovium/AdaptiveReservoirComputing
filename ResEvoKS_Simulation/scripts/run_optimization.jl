#!/usr/bin/env julia
# =============================================================================
# scripts/run_optimization.jl
# -----------------------------------------------------------------------------
# Top-level driver: generate KS data and evolve reservoirs with the GA.
#
# Julia equivalent of the MATLAB pair
#   KS64D_prepDataAndRun.m   (outer sweep: make data, set dirs, call runner)
#   KS64D_runOptimizePredESN.m (the GA run itself)
#
# Usage
# -----
#   # from the package root, using its environment, with N threads:
#   julia --project=. -t auto scripts/run_optimization.jl
#
#   # override defaults via environment variables (all optional):
#   RESEVO_ROOTDIR=results RESEVO_NRUNS=4 \
#   RESEVO_NSTEP=100000 RESEVO_TRAIN=70000 RESEVO_PREDICT=2000 \
#   RESEVO_POP=300 RESEVO_GENS=101 \
#   julia --project=. -t auto scripts/run_optimization.jl
#
# Each run writes <rootdir>/RUN<k>/{Log,matfiles,Figs}. The per-individual
# `.mat` files in matfiles/ are named <gen>_<N>_<degree>_<radius>_<sigma>.mat
# and are directly consumable by the analysis scripts in original_code/analysis.
#
# NOTE: the defaults below reproduce the *full* paper-scale experiment, which is
# computationally heavy (300 individuals × 101 generations × several runs, each
# training reservoirs up to 3000 nodes on 70k samples). For a quick functional
# run, set the RESEVO_* environment variables to small values (see README).
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ResEvoKS_Simulation
using LinearAlgebra: BLAS
using Dates
using Printf

# Avoid BLAS/Julia-thread oversubscription: when we parallelize the GA over
# Julia threads, pin BLAS to a single thread per evaluation.
if Threads.nthreads() > 1
    BLAS.set_num_threads(1)
end

# --- helper: read an environment variable with a typed default ---------------
getenv(key, default::Integer) = haskey(ENV, key) ? parse(Int, ENV[key]) : default
getenv(key, default::AbstractFloat) = haskey(ENV, key) ? parse(Float64, ENV[key]) : default
getenv(key, default::AbstractString) = get(ENV, key, default)

function main()
    # =============================== configuration ===========================
    rootdir   = getenv("RESEVO_ROOTDIR", "results")
    nruns     = getenv("RESEVO_NRUNS", 10)

    # KS model parameters (paper Methods: N=64, d=22, Δt=0.25)
    model = KSModelParams(
        N     = getenv("RESEVO_N", 64),
        d     = getenv("RESEVO_D", 22.0),
        tau   = getenv("RESEVO_TAU", 0.25),
        nstep = getenv("RESEVO_NSTEP", 100_000),
    )

    # training / prediction window lengths
    dataparams = DataParams(
        train_length   = getenv("RESEVO_TRAIN", 70_000),
        predict_length = getenv("RESEVO_PREDICT", 2_000),
    )

    # Output format for saved individuals/dataset: "mat" (default), "jld2", "both".
    save_format = Symbol(getenv("RESEVO_FORMAT", "mat"))

    # GA settings (paper: pop 300, 101 generations, the 5-gene bounds)
    base_settings = GASettings(
        population_size = getenv("RESEVO_POP", 300),
        max_generations = getenv("RESEVO_GENS", 101),
        seed            = getenv("RESEVO_SEED", 20200501),
        save_format     = save_format,
    )

    println("="^70)
    println("ResEvoKS_Simulation — Evolutionary Reservoir Computing for KS chaos")
    println("threads = ", Threads.nthreads(), "   BLAS threads = ", BLAS.get_num_threads())
    println("rootdir = ", rootdir, "   runs = ", nruns)
    println("KS: N=$(model.N) d=$(model.d) tau=$(model.tau) nstep=$(model.nstep)")
    println("train=$(dataparams.train_length) predict=$(dataparams.predict_length)")
    println("GA: pop=$(base_settings.population_size) gens=$(base_settings.max_generations)")
    println("save format = ", save_format)
    println("="^70)

    for kk in 1:nruns
        println("\n", "#"^70)
        println("# RUN $kk / $nruns   ", now())
        println("#"^70)

        # ---------------- per-run output directories ------------------------
        sp = make_run_dirs(rootdir, kk)
        println("[run $kk] log → ", joinpath(sp.codedir, "esn_log.txt"))

        # ---------------- generate KS data ----------------------------------
        println("[run $kk] generating KS trajectory ...")
        # Per-run reproducible initial condition.
        ic = random_initial_condition(model.N;
                rng = ResEvoKS_Simulation.KuramotoSivashinsky.StableRNG(base_settings.seed + kk))
        data = solve_ks(ic, model)
        @assert all(isfinite, data) "KS trajectory contains non-finite values"
        save_dataset(sp.matdir, data, model; format = save_format)
        @printf("[run %d] KS done: size=%s  max|u|=%.3f\n", kk, size(data),
                maximum(abs, data))

        # ---------------- run the GA ----------------------------------------
        # Vary the GA seed per run so runs are independent (as in the paper).
        settings = GASettings(base_settings;
                              seed = base_settings.seed + 1000 * kk)
        result = optimize_reservoirs(data, dataparams, sp;
                                     settings = settings, verbose = true)

        @printf("[run %d] best J = %.6e   genome = %s\n", kk, result.best_J,
                string(round.(result.best_genome, sigdigits=4)))
    end

    println("\nAll runs complete. Per-individual .mat files are under ",
            joinpath(rootdir, "RUN<k>", "matfiles"), ".")
end

main()
