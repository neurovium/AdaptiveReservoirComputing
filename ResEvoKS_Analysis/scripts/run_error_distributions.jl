#!/usr/bin/env julia
# =============================================================================
# scripts/run_error_distributions.jl
# -----------------------------------------------------------------------------
# Population-level reduction of composite prediction error across generations
# (paper §"Population-level reduction of composite prediction error").
#
# Ports the data view of evoRunGenErrors_64D-ks.jl / sampleGenDistError_64D-ks.jl:
# for each run, the distribution of log10 J per generation, shown as overlaid
# histograms and a per-generation violin/box summary.
#
# Driver for ResEvoKS_Analysis.ErrorStats.
#
# Usage:
#   RESEVO_RESULTS=results RESEVO_RUNS=1:10 RESEVO_GENS=0:10:50 \
#   RESEVO_OUTDIR=analysis_results julia --project=. scripts/run_error_distributions.jl
# =============================================================================

using ResEvoKS_Analysis
using Plots

results_root = get(ENV, "RESEVO_RESULTS", "results")
runs   = eval(Meta.parse(get(ENV, "RESEVO_RUNS", "1:10")))
gens   = collect(eval(Meta.parse(get(ENV, "RESEVO_GENS", "0:10:50"))))
outdir = joinpath(get(ENV, "RESEVO_OUTDIR", "analysis_results"), "errors")
mkpath(outdir)

for run_index in runs
    matdir = run_matdir(results_root, run_index)
    isdir(matdir) || (@warn "skipping missing run" matdir; continue)

    errs = collect_generation_errors(matdir, gens)

    # overlaid log10 J histograms (later generations more transparent)
    p_hist = plot(; xlabel="log10(J)", ylabel="frequency",
                  title="Error distribution by generation — RUN $run_index")
    for (j, gen) in enumerate(gens)
        lg = log_error_distribution(errs[gen])
        isempty(lg) && continue
        histogram!(p_hist, lg; bins=60, alpha=max(0.25, 1 - 0.15 * (j - 1)),
                   label="gen $gen")
    end
    savefig(p_hist, joinpath(outdir, "RUN$(run_index)_error_histograms.png"))
    savefig(p_hist, joinpath(outdir, "RUN$(run_index)_error_histograms.svg"))

    # per-generation median +/- spread of log10 J (Plots-only, no StatsPlots).
    # Whiskers span [min, max] of log10 J; the marker is the median.
    stats = generation_error_stats(matdir, gens)
    keep  = filter(s -> s.n > 0, stats)
    if !isempty(keep)
        gx   = [s.generation for s in keep]
        med  = [s.median for s in keep]
        lo   = [s.median - s.min for s in keep]   # lower whisker length
        hi   = [s.max - s.median for s in keep]   # upper whisker length
        p_box = scatter(gx, med; yerror=(lo, hi), legend=false, marker=:circle,
                        xlabel="generation", ylabel="log10(J)  (median, min–max)",
                        title="Error evolution — RUN $run_index")
        plot!(p_box, gx, med; lw=2, color=:red)
        savefig(p_box, joinpath(outdir, "RUN$(run_index)_error_spread.png"))
        savefig(p_box, joinpath(outdir, "RUN$(run_index)_error_spread.svg"))
    end

    # console summary
    println("RUN $run_index:")
    for s in generation_error_stats(matdir, gens)
        println("  gen $(s.generation): n=$(s.n) (inf=$(s.n_inf)) ",
                "median log10 J=$(round(s.median, digits=3))")
    end
end

println("Done. Error-distribution figures under $outdir.")
