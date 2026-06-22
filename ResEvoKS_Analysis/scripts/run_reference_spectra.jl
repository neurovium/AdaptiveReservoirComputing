#!/usr/bin/env julia
# =============================================================================
# scripts/run_reference_spectra.jl
# -----------------------------------------------------------------------------
# Random-walk Laplacian spectral densities of canonical graph ensembles
# (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, Stochastic Block Model),
# the "SBM-like envelope" the evolved reservoirs are compared against
# (paper §"Reservoirs occupy an SBM-like spectral signature class").
#
# Julia port of normalizedlaplacianrandomgraphs.py. Driver for
# ResEvoKS_Analysis.{ReferenceGraphs,Spectral}.
#
# Usage:
#   RESEVO_N=500 RESEVO_NINST=100 RESEVO_OUTDIR=analysis_results \
#   julia --project=. scripts/run_reference_spectra.jl
# =============================================================================

using ResEvoKS_Analysis
using Plots
using StableRNGs

N        = parse(Int, get(ENV, "RESEVO_N", "500"))
n_inst   = parse(Int, get(ENV, "RESEVO_NINST", "100"))
outdir   = joinpath(get(ENV, "RESEVO_OUTDIR", "analysis_results"), "reference")
mkpath(outdir)

# --- single-instantiation comparison ----------------------------------------
specs = reference_spectra(n=N; rng=StableRNG(42))
labels = Dict(:er=>"Random (Erdős–Rényi)", :ba=>"Power-law (Barabási–Albert)",
              :ws=>"Small-world (Watts–Strogatz)", :sbm=>"SBM")

p = plot(; xlabel="Re(eigenvalue)", ylabel="density (smoothed)",
         title="Reference spectra (normalized Laplacian)", xlims=(0, 2.5))
for key in (:er, :ba, :ws, :sbm)
    grid, Γ = smoothed_density(specs[key]; sigma=0.03, bins=0:0.005:2.5)
    plot!(p, grid, Γ; lw=2, label=labels[key])
end
vline!(p, [1.0]; color=:black, ls=:dot, alpha=0.5, label="")
savefig(p, joinpath(outdir, "reference_spectral_density.png"))
savefig(p, joinpath(outdir, "reference_spectral_density.svg"))

# --- averaged over many instantiations --------------------------------------
println("Averaging reference spectra over $n_inst instantiations...")
pooled = Dict(k => Float64[] for k in (:er, :ba, :ws, :sbm))
for i in 1:n_inst
    s = reference_spectra(n=N; rng=StableRNG(1000 + i))
    for k in keys(pooled)
        append!(pooled[k], s[k])
    end
end

p_avg = plot(; xlabel="Re(eigenvalue)", ylabel="density (smoothed)",
             title="Averaged reference spectra ($n_inst instantiations)", xlims=(0, 2.5))
for key in (:er, :ba, :ws, :sbm)
    grid, Γ = smoothed_density(pooled[key]; sigma=0.03, bins=0:0.005:2.5)
    plot!(p_avg, grid, Γ; lw=2, label=labels[key])
end
vline!(p_avg, [1.0]; color=:black, ls=:dot, alpha=0.5, label="")
savefig(p_avg, joinpath(outdir, "reference_spectral_density_averaged.png"))
savefig(p_avg, joinpath(outdir, "reference_spectral_density_averaged.svg"))

println("Done. Reference-ensemble figures under $outdir.")
