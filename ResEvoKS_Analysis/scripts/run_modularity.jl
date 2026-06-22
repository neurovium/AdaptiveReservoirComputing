#!/usr/bin/env julia
# =============================================================================
# scripts/run_modularity.jl
# -----------------------------------------------------------------------------
# Modularity, connection cost, and NSGA-II multi-objective analysis
# (paper §§"Community detection and modularity", "Connection density, path
# length, and connection cost", "Multi-objective analysis").
#
# Pipeline for one run:
#   1. select generations across evolutionary time + stratified-sample them,
#   2. per reservoir: directed Newman modularity Q + regularized connection cost,
#   3. modularity-vs-cost scatter colored by performance; per-generation means,
#   4. exponential fit of mean connection cost vs generation,
#   5. NSGA-II over the normalized (mod, cost, perf, gen) trade-off; overlay the
#      Pareto-optimal points' closest observed reservoirs.
#
# Driver for ResEvoKS_Analysis.{Sampling,Modularity,Pareto,MultiObjective}.
#
# Usage:
#   RESEVO_RESULTS=results RESEVO_RUN=1 RESEVO_NSELECT=20 RESEVO_NQUARTILE=20 \
#   RESEVO_NSGA=1 RESEVO_OUTDIR=analysis_results \
#   julia --project=. scripts/run_modularity.jl
# =============================================================================

using ResEvoKS_Analysis
using Plots
using StableRNGs
using Statistics

# --- configuration ----------------------------------------------------------
results_root = get(ENV, "RESEVO_RESULTS", "results")
run_index    = parse(Int, get(ENV, "RESEVO_RUN", "1"))
n_select     = parse(Int, get(ENV, "RESEVO_NSELECT", "20"))
n_quartile   = parse(Int, get(ENV, "RESEVO_NQUARTILE", "20"))
do_nsga      = get(ENV, "RESEVO_NSGA", "1") == "1"
outdir       = joinpath(get(ENV, "RESEVO_OUTDIR", "analysis_results"), "modularity", "RUN$(run_index)")
rng          = StableRNG(1234)

matdir = run_matdir(results_root, run_index)
isdir(matdir) || error("missing run directory: $matdir")
mkpath(outdir)

# --- 1. generation selection + stratified sampling --------------------------
gens     = available_generations(matdir)
selected = select_generations(gens, matdir; n_select=n_select)
samples  = stratified_sample_run(matdir, selected; n_per_quartile=n_quartile, rng=rng)
println("Selected generations: ", selected)
println("Sampling $(length(samples)) reservoirs for modularity / cost...")

# --- 2. per-reservoir structural metrics ------------------------------------
modularity   = Float64[]
cost         = Float64[]
performance  = Float64[]   # composite error J
generation   = Int[]
for f in samples
    rec = load_record(matdir, f)
    m = network_metrics(rec.A)
    push!(modularity, m.modularity)
    push!(cost, m.cost)
    push!(performance, rec.J)
    push!(generation, rec.generation)
end

# --- 3. modularity vs cost, colored by performance --------------------------
p_scatter = scatter(modularity, cost; marker_z=log.(performance), c=:berlin,
                    colorbar=true, colorbar_title="log J", legend=false,
                    xlabel="modularity Q", ylabel="connection cost",
                    title="Modularity vs cost (color = log J) — RUN $run_index")
savefig(p_scatter, joinpath(outdir, "modularity_vs_cost.png"))
savefig(p_scatter, joinpath(outdir, "modularity_vs_cost.svg"))

# per-generation means
sorted_gens = sort(unique(generation))
mean_mod  = [mean(modularity[generation .== g]) for g in sorted_gens]
mean_cost = [mean(cost[generation .== g]) for g in sorted_gens]

p_modgen = plot(sorted_gens, mean_mod; lw=2, marker=:circle, legend=false,
                xlabel="generation", ylabel="mean modularity",
                title="Modularity across evolution — RUN $run_index")
savefig(p_modgen, joinpath(outdir, "mean_modularity_by_generation.png"))

# --- 4. exponential fit of mean connection cost vs generation ---------------
costfit = fit_exponential(Float64.(sorted_gens), mean_cost;
                          p0=[maximum(mean_cost) - minimum(mean_cost), 0.05, minimum(mean_cost)])
xs = range(minimum(sorted_gens), maximum(sorted_gens); length=100)
ys = exponential_model(collect(xs), [costfit.a, costfit.b, costfit.c])
p_costgen = plot(sorted_gens, mean_cost; lw=2, marker=:circle, label="observed",
                 xlabel="generation", ylabel="mean connection cost",
                 title="Connection cost decay — RUN $run_index")
plot!(p_costgen, xs, ys; lw=3, ls=:dash, color=:red,
      label="a·e^(-b·x)+c  (R²=$(round(costfit.r_squared, digits=3)))")
savefig(p_costgen, joinpath(outdir, "connection_cost_exponential_fit.png"))
println("Connection-cost fit: half-life ≈ $(round(log(2)/costfit.b, digits=1)) generations, ",
        "R²=$(round(costfit.r_squared, digits=3))")

# --- 5. NSGA-II multi-objective trade-off -----------------------------------
if do_nsga
    println("Running NSGA-II over normalized (modularity, cost, performance, generation)...")
    nm = normalize_metrics(modularity=modularity, connection_cost=cost,
                           performance=performance, generation=generation)
    optimized = run_nsga2(N=1000, p_cr=0.85, p_m=0.5)
    matches = closest_observed(optimized, nm)
    println("NSGA-II returned $(size(optimized,1)) Pareto-optimal points; ",
            "$(length(unique(matches))) distinct closest observed reservoirs.")

    p_nsga = scatter(nm.modularity, nm.connection_cost; marker_z=nm.performance,
                     c=:berlin, colorbar=true, label="", legend=:topright,
                     xlabel="normalized modularity", ylabel="normalized cost",
                     title="NSGA-II trade-off — RUN $run_index")
    scatter!(p_nsga, nm.modularity[matches], nm.connection_cost[matches];
             color=:red, ms=5, label="closest to Pareto front")
    savefig(p_nsga, joinpath(outdir, "nsga2_tradeoff.png"))
    savefig(p_nsga, joinpath(outdir, "nsga2_tradeoff.svg"))
end

println("Done. Modularity figures under $outdir.")
