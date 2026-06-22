#!/usr/bin/env julia
# =============================================================================
# scripts/demo_good_vs_bad.jl
# -----------------------------------------------------------------------------
# GA-free schematic demo: contrast a "good", a "bad", and an "awful" reservoir
# on the same KS trajectory, reproducing the spirit of
# original_code/simulation/KS-reservoir-demo/ks_reservoir.py.
#
# For each configuration it teacher-forces the reservoir, trains the ridge
# readout, runs an autonomous forecast, and reports the short-horizon NRMSE. The
# truth / prediction / difference fields are saved to a `.mat` file so they can
# be plotted with any tool (CairoMakie, Plots, MATLAB, matplotlib).
#
# Usage:
#   julia --project=. scripts/demo_good_vs_bad.jl
#   RESEVO_NSTEP=20000 RESEVO_TRAIN=14000 julia --project=. scripts/demo_good_vs_bad.jl
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ResEvoKS_Simulation
using ResEvoKS_Simulation.Reservoir: StableRNG
using MAT
using JLD2
using Printf
using Statistics

getenv(key, default::Integer) = haskey(ENV, key) ? parse(Int, ENV[key]) : default

function short_horizon_nrmse(pred, truth; horizon=200)
    h = min(horizon, size(pred, 2))
    p = @view pred[:, 1:h]
    t = @view truth[:, 1:h]
    # mean over space of RMSE(t) / std(truth)
    return mean(sqrt.(vec(mean((p .- t) .^ 2; dims=1))) ./ (std(t) + 1e-12))
end

function run_config(label, cfg, data, dataparams)
    @printf("[%s] building reservoir %s\n", label, string(cfg))
    genome = [cfg.radius, Float64(cfg.degree), Float64(cfg.N), cfg.sigma, cfg.beta]
    res = evaluate_individual(genome, data, dataparams;
                              rng = StableRNG(cfg.seed), return_artifacts = true)

    # Autonomous prediction window (the held-out segment).
    pred_window = (dataparams.train_length + 1):(dataparams.train_length + dataparams.predict_length)
    truth = Matrix(@view data[:, pred_window])

    # Re-run the forecast to expose the field (evaluate_individual scored it
    # internally; here we want the array to save/plot).
    states = ResEvoKS_Simulation.Readout.reservoir_layer(data, res.A, res.w_in, res.resparams,
                                              dataparams.train_length, tanh_activation)
    x0 = states[:, end]
    pred, _ = ResEvoKS_Simulation.Readout.predict(x0, res.A, res.w_in, res.resparams,
                                       res.w_out, dataparams.predict_length,
                                       tanh_activation)

    nrmse = short_horizon_nrmse(pred, truth)
    @printf("[%s] N=%d  J=%.4g  short-horizon NRMSE@200=%.4f\n",
            label, res.resparams.N, res.J, nrmse)
    return (truth = truth, pred = pred, diff = pred .- truth, nrmse = nrmse, J = res.J)
end

function main()
    nstep   = getenv("RESEVO_NSTEP", 100_000)
    train   = getenv("RESEVO_TRAIN", 70_000)
    predict = getenv("RESEVO_PREDICT", 2_000)

    model = KSModelParams(N = 64, d = 22.0, tau = 0.25, nstep = nstep)
    dataparams = DataParams(train_length = train, predict_length = predict)

    @printf("Solving KS: N=%d d=%.1f tau=%.2f nstep=%d ...\n",
            model.N, model.d, model.tau, model.nstep)
    data = solve_ks(random_initial_condition(model.N; rng = StableRNG(42)), model)
    @assert all(isfinite, data)
    @printf("KS done: max|u|=%.3f\n", maximum(abs, data))

    # Three illustrative configurations (sizes are multiples of 64).
    configs = (
        good  = (N = 4096, degree = 6, radius = 0.9, sigma = 0.5, beta = 1e-4, seed = 2),
        bad   = (N = 512,  degree = 2, radius = 1.0, sigma = 1.0, beta = 1e-3, seed = 3),
        awful = (N = 64,   degree = 2, radius = 0.5, sigma = 0.1, beta = 10.0, seed = 15),
    )

    results = Dict{String,Any}()
    for (name, cfg) in pairs(configs)
        r = run_config(String(name), cfg, data, dataparams)
        results[String(name)] = Dict(
            "truth" => r.truth, "pred" => r.pred, "diff" => r.diff,
            "nrmse" => r.nrmse, "J" => r.J,
        )
    end

    outdir = joinpath(@__DIR__, "..", "demo_output")
    isdir(outdir) || mkpath(outdir)
    base = joinpath(outdir, "demo_good_vs_bad")

    # Honor the same RESEVO_FORMAT switch as the main run script.
    fmt = Symbol(get(ENV, "RESEVO_FORMAT", "mat"))
    fmt in (:mat, :jld2, :both) ||
        error("RESEVO_FORMAT must be mat, jld2, or both (got $fmt)")
    written = String[]
    if fmt === :mat || fmt === :both
        matwrite(base * ".mat", results); push!(written, base * ".mat")
    end
    if fmt === :jld2 || fmt === :both
        jldopen(base * ".jld2", "w") do f
            for (k, v) in results
                f[k] = v
            end
        end
        push!(written, base * ".jld2")
    end

    println("\nSaved truth/prediction/difference fields to:")
    foreach(p -> println("  ", p), written)
    println("Plot them with your tool of choice (each field is num_inputs × predict_length).")
end

main()
