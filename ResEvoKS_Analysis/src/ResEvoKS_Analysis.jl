# =============================================================================
# ResEvoKS_Analysis.jl  —  top-level module
# -----------------------------------------------------------------------------
# Phase-2 analysis suite for the evolutionary reservoir-computing project.
#
# Reads the per-individual artifacts written by the ResEvoKS_Simulation simulation and
# reproduces the paper's structural analyses. Companion to the paper
#
#   "Evolutionary Optimization Reveals Structural Constraints on Reservoir
#    Architecture for Spatiotemporal Chaos", N. Dehghani.
#
# Module map (each submodule ports one analysis theme; see
# REPORT_analysis_code.md for the full correspondence to the original files and
# the paper Methods subsections):
#
#   DataAccess       ← file listing / get_J_N / load_adjacency_matrix
#   Sampling         ← generation selection + stratified quartile sampling
#   Pareto           ← size–efficiency frontier + exponential fit
#   Spectral         ← random-walk Laplacian, eigenvalues, smoothed density
#   Embedding        ← interpolation + PCA + k-means/silhouette
#   SpectralEMD      ← optimal-transport spectral distance
#   Modularity       ← communities, directed modularity, connection cost
#   MultiObjective   ← normalization, 4 objectives, NSGA-II
#   ReferenceGraphs  ← ER/BA/WS/SBM reference spectra
#   ErrorStats       ← per-generation J error distributions
#
# Computation only — plotting lives in scripts/. Quick start:
#
#   using ResEvoKS_Analysis
#   matdir = run_matdir("results", 1)               # results/RUN1/matfiles
#   gens   = available_generations(matdir)
#   sel    = select_generations(gens, matdir; n_select=10)
#   samp   = stratified_sample_run(matdir, sel)
#   evs    = [laplacian_eigenvalues(load_adjacency(matdir, f)) for f in samp]
#   grid, M = spectral_density_grid(evs)
# =============================================================================

module ResEvoKS_Analysis

# --- submodules (order matters: later ones use earlier ones) ----------------
include("DataAccess.jl")
include("Sampling.jl")
include("Pareto.jl")
include("Spectral.jl")
include("Embedding.jl")
include("SpectralEMD.jl")
include("Modularity.jl")
include("MultiObjective.jl")
include("ReferenceGraphs.jl")
include("ErrorStats.jl")

using .DataAccess
using .Sampling
using .Pareto
using .Spectral
using .Embedding
using .SpectralEMD
using .Modularity
using .MultiObjective
using .ReferenceGraphs
using .ErrorStats

# --- public API (re-exported from the submodules) ---------------------------
# data access
export IndividualRecord, run_matdir, list_individual_files, parse_generation,
       available_generations, files_for_generation, load_record,
       load_adjacency, read_J_N, collect_J_N
# sampling
export find_suitable_last_gen, select_generations, stratified_sample,
       stratified_sample_run
# pareto / size–efficiency
export pareto_frontier, fit_exponential, ParetoFit, exponential_model
# spectral
export rw_laplacian, laplacian_eigenvalues, smoothed_density,
       spectral_density_grid, spectral_centroid
# embedding
export interpolate_spectrum, spectra_matrix, spectra_pca, SpectraPCA,
       cluster_spectra, ClusterResult
# optimal-transport EMD
export spectral_emd, pairwise_emd, emd_to_reference
# modularity / cost
export detect_communities, directed_modularity, connection_density,
       average_path_length, connection_cost, NetworkMetrics, network_metrics
# multi-objective
export normalize_metric, normalize_metrics, NormalizedMetrics,
       composite_objectives, run_nsga2, closest_observed
# reference graphs
export er_spectrum, ba_spectrum, ws_spectrum, sbm_spectrum, reference_spectra
# error statistics
export collect_generation_errors, log_error_distribution,
       GenerationErrorStats, generation_error_stats

end # module ResEvoKS_Analysis
