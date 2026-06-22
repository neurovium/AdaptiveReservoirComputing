#!/usr/bin/env julia
# =============================================================================
# scripts/run_spectral.jl
# -----------------------------------------------------------------------------
# Random-walk Laplacian spectral analysis of evolved reservoirs
# (paper §§"Normalized random-walk Laplacian spectrum", "Smoothed eigenvalue
# distributions", "Network selection", "PCA of Laplacian spectra", "Spectral
# distance using optimal transport").
#
# Pipeline for one run:
#   1. select ~10 generations across evolutionary time + stratified-sample them,
#   2. compute each reservoir's RW-Laplacian eigenvalues,
#   3. PCA of interpolated spectra (colored by generation) + k-means clustering,
#   4. smoothed spectral density: generational averages + evolution heatmap,
#   5. EMD of each spectrum to the generation-0 reference (spectral drift).
#
# Driver for ResEvoKS_Analysis.{Sampling,Spectral,Embedding,SpectralEMD}.
#
# Usage:
#   RESEVO_RESULTS=results RESEVO_RUN=1 RESEVO_NSELECT=10 RESEVO_NQUARTILE=20 \
#   RESEVO_OUTDIR=analysis_results julia --project=. scripts/run_spectral.jl
# =============================================================================

using ResEvoKS_Analysis
using Plots
using StableRNGs
using Statistics

# --- configuration ----------------------------------------------------------
results_root = get(ENV, "RESEVO_RESULTS", "results")
run_index    = parse(Int, get(ENV, "RESEVO_RUN", "1"))
n_select     = parse(Int, get(ENV, "RESEVO_NSELECT", "10"))
n_quartile   = parse(Int, get(ENV, "RESEVO_NQUARTILE", "20"))
target_q     = parse(Int, get(ENV, "RESEVO_TARGETQ", "200"))
outdir       = joinpath(get(ENV, "RESEVO_OUTDIR", "analysis_results"), "spectral", "RUN$(run_index)")
rng          = StableRNG(1234)

matdir = run_matdir(results_root, run_index)
isdir(matdir) || error("missing run directory: $matdir")
mkpath(outdir)

# --- 1. generation selection + stratified sampling --------------------------
gens = available_generations(matdir)
selected = select_generations(gens, matdir; n_select=n_select)
samples  = stratified_sample_run(matdir, selected; n_per_quartile=n_quartile, rng=rng)
sample_gens = parse_generation.(samples)
println("Selected generations: ", selected)
println("Sampled $(length(samples)) reservoirs ",
        "(~$(length(samples) ÷ max(length(selected),1)) per generation)")

# --- 2. Laplacian eigenvalues -----------------------------------------------
println("Computing random-walk Laplacian spectra...")
eigenvalue_list = Vector{Vector{Float64}}(undef, length(samples))
for (i, f) in enumerate(samples)
    eigenvalue_list[i] = laplacian_eigenvalues(load_adjacency(matdir, f))
end

# --- 3. PCA + clustering ----------------------------------------------------
S = spectra_matrix(eigenvalue_list, target_q)
pca = spectra_pca(S; maxoutdim=3)
clus = cluster_spectra(S; max_clusters=10, rng=rng)
println("PCA: first 3 PCs explain $(round(pca.cumulative[min(3,end)]*100, digits=1))% variance")
println("k-means: best k=$(clus.k) (silhouette $(round(clus.silhouette, digits=3)))")

p_pca = scatter(pca.scores[1, :], pca.scores[2, :]; group=sample_gens,
                xlabel="PC1", ylabel="PC2",
                title="PCA of Laplacian spectra — RUN $run_index",
                legendtitle="gen", ms=4, alpha=0.7)
savefig(p_pca, joinpath(outdir, "pca_by_generation.png"))
savefig(p_pca, joinpath(outdir, "pca_by_generation.svg"))

p_clus = scatter(pca.scores[1, :], pca.scores[2, :]; group=clus.assignments,
                 xlabel="PC1", ylabel="PC2",
                 title="k-means clusters (k=$(clus.k)) in PCA space", ms=4, alpha=0.7)
savefig(p_clus, joinpath(outdir, "pca_clusters.png"))
savefig(p_clus, joinpath(outdir, "pca_clusters.svg"))

# --- 4. smoothed density: generational averages + heatmap -------------------
grid, dens = spectral_density_grid(eigenvalue_list; sigma=0.015, bins=0:0.001:2)

p_avg = plot(; xlabel="eigenvalue λ", ylabel="avg spectral density",
             title="Spectral evolution — generational averages")
heat = zeros(length(selected), length(grid))
for (gi, gen) in enumerate(selected)
    cols = findall(==(gen), sample_gens)
    isempty(cols) && continue
    avg = vec(mean(dens[:, cols], dims=2))
    heat[gi, :] = avg
    plot!(p_avg, grid, avg; label="gen $gen", lw=2)
end
savefig(p_avg, joinpath(outdir, "spectral_averages_by_generation.png"))
savefig(p_avg, joinpath(outdir, "spectral_averages_by_generation.svg"))

p_heat = heatmap(grid, selected, heat; xlabel="eigenvalue λ", ylabel="generation",
                 title="Spectral density evolution", color=:plasma)
savefig(p_heat, joinpath(outdir, "spectral_evolution_heatmap.png"))
savefig(p_heat, joinpath(outdir, "spectral_evolution_heatmap.svg"))

# --- 5. EMD to the generation-0 reference -----------------------------------
gen0 = selected[1]
ref_idx = findall(==(gen0), sample_gens)
reference = reduce(vcat, eigenvalue_list[ref_idx])
emd = emd_to_reference(eigenvalue_list, reference; show_progress=false)

p_emd = scatter(sample_gens, emd; xlabel="generation",
                ylabel="EMD to gen $gen0 reference",
                title="Spectral drift (optimal transport) — RUN $run_index",
                legend=false, ms=4, alpha=0.6)
mean_by_gen = [mean(emd[findall(==(g), sample_gens)]) for g in selected]
plot!(p_emd, selected, mean_by_gen; lw=3, color=:red, marker=:circle, label="mean")
savefig(p_emd, joinpath(outdir, "spectral_emd_drift.png"))
savefig(p_emd, joinpath(outdir, "spectral_emd_drift.svg"))

println("Done. Spectral figures under $outdir.")
