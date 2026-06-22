#!/usr/bin/env julia
# =============================================================================
# scripts/run_pareto.jl
# -----------------------------------------------------------------------------
# Reservoir size–efficiency Pareto analysis (paper §"Reservoir size–efficiency
# trade-off"). For each requested run and generation, this:
#   1. gathers (N, log J) for every individual,
#   2. computes the 2-D Pareto frontier (minimize size and log-error),
#   3. fits the summary curve f(x)=a e^{-b x}+c,
#   4. saves a scatter + frontier + fit figure.
#
# Driver for ResEvoKS_Analysis.Pareto. Computation is in src/; this script only
# orchestrates and plots.
#
# Usage (environment variables, all optional):
#   RESEVO_RESULTS=results   # root holding RUN<k>/matfiles
#   RESEVO_RUNS=1:10          # run indices
#   RESEVO_GENS=0:10:50      # generations to analyze
#   RESEVO_OUTDIR=analysis_results
#
#   julia --project=. scripts/run_pareto.jl
# =============================================================================

using ResEvoKS_Analysis
using Plots

# --- configuration ----------------------------------------------------------
results_root = get(ENV, "RESEVO_RESULTS", "results")
runs   = eval(Meta.parse(get(ENV, "RESEVO_RUNS", "1:10")))
gens   = eval(Meta.parse(get(ENV, "RESEVO_GENS", "0:10:50")))
outdir = get(ENV, "RESEVO_OUTDIR", "analysis_results")

println("Pareto size–efficiency analysis")
println("  results root : $results_root")
println("  runs         : $runs")
println("  generations  : $gens")
println("  output dir   : $outdir")

for run_index in runs
    matdir = run_matdir(results_root, run_index)
    isdir(matdir) || (@warn "skipping missing run" matdir; continue)

    run_out = joinpath(outdir, "pareto", "RUN$(run_index)")
    mkpath(run_out)

    for gen in gens
        files = files_for_generation(matdir, gen)
        isempty(files) && continue

        # (N, log J) cloud for this generation
        J, N = collect_J_N(matdir, files)
        points = hcat(Float64.(N), log.(J))           # columns: size, log error

        # Pareto frontier + exponential summary fit
        _, frontier = pareto_frontier(points)
        px, py = frontier[:, 1], frontier[:, 2]
        fit = fit_exponential(px, py)
        xs = range(minimum(px), maximum(px); length=200)
        ys = exponential_model(collect(xs), [fit.a, fit.b, fit.c])

        # figure
        plt = scatter(points[:, 1], points[:, 2];
                      label="reservoirs", ms=3, alpha=0.5,
                      xlabel="reservoir size n_r", ylabel="log J",
                      title="Pareto front — RUN $run_index, gen $gen")
        scatter!(plt, px, py; label="Pareto front", color=:red, ms=5)
        plot!(plt, xs, ys; label="a·e^(-b·x)+c  (R²=$(round(fit.r_squared, digits=3)))",
              color=:black, lw=2)

        base = joinpath(run_out, "pareto_run$(run_index)_gen$(gen)")
        savefig(plt, base * ".png")
        savefig(plt, base * ".svg")
        println("  RUN $run_index gen $gen: ",
                "$(size(frontier,1)) frontier pts, ",
                "f(x)=$(round(fit.a,digits=3))·e^(-$(round(fit.b,digits=4))·x)+",
                "$(round(fit.c,digits=3)), R²=$(round(fit.r_squared,digits=3))")
    end
end

println("Done. Figures under $(joinpath(outdir, "pareto")).")
