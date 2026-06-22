# API Reference

All names below are exported by `using ResEvoKS_Analysis`. Full docstrings live
with each function (`?name` at the REPL). Keyword arguments follow `;`.

---

## Data access — `DataAccess.jl`

### `run_matdir(rootdir, run_index) -> String`
Path `<rootdir>/RUN<run_index>/matfiles`.

### `list_individual_files(matdir; ext=("mat","jld2")) -> Vector{String}`
Artifact base filenames in `matdir`, excluding `data.*` / `history.*`.

### `parse_generation(filename) -> Int`
Generation index from the `"<gen>_…"` filename prefix.

### `available_generations(matdir; ext=("mat","jld2")) -> Vector{Int}`
Sorted unique generations present.

### `files_for_generation(matdir, gen; ext=("mat","jld2")) -> Vector{String}`
Filenames of generation `gen` (exact `"<gen>_"` prefix).

### `load_record(matdir, filename) -> IndividualRecord`
Full load: `A`, `J`, `N`, `NRMSE`, `NMAE`, `resparams`. Fields:
`filename, generation, A, J, N, NRMSE, NMAE, resparams`.

### `load_adjacency(matdir, filename) -> AbstractMatrix`
Just the recurrent matrix `A`.

### `read_J_N(matdir, filename) -> (J::Float64, N::Int)`
Just `J` and `N` (fast; port of `get_J_N`).

### `collect_J_N(matdir, filenames) -> (J::Vector, N::Vector)`
`(J, N)` for many files as parallel vectors.

---

## Sampling — `Sampling.jl`

### `find_suitable_last_gen(generations, matdir; min_individuals=299) -> Int`
Latest generation with ≥ `min_individuals` artifacts (fallback: last).

### `select_generations(generations, matdir; n_select=10, min_individuals=299) -> Vector{Int}`
`n_select` generations evenly spaced from the first to the suitable last.

### `stratified_sample(gen_files, J; n_per_quartile=20, rng=StableRNG(1234)) -> Vector{String}`
Quartile-stratified sample of one generation (≤ `4·n_per_quartile` files).

### `stratified_sample_run(matdir, selected_generations; n_per_quartile=20, rng=StableRNG(1234)) -> Vector{String}`
`stratified_sample` applied to each selected generation, concatenated.

---

## Pareto / size–efficiency — `Pareto.jl`

### `pareto_frontier(points) -> (mask::Vector{Bool}, frontier::Matrix)`
2-objective (minimize both) non-dominated set of an `M×2` matrix.

### `exponential_model(x, p) = p[1]*exp(-p[2]*x) + p[3]`
The frontier-summary model.

### `fit_exponential(px, py; p0=nothing) -> ParetoFit`
Nonlinear LSQ fit of `f(x)=a e^{-bx}+c`. `ParetoFit`: `a, b, c, r_squared, x, y`.

---

## Spectral — `Spectral.jl`

### `rw_laplacian(A) -> L`
Random-walk Laplacian `I − D⁻¹A` (sink rows regularized to zero).

### `laplacian_eigenvalues(A; nev=size(A,1)-2, which=:SM, dense_threshold=20) -> Vector{Float64}`
Sorted real eigenvalues of `L`; ARPACK by default, dense for small/`:all`.

### `smoothed_density(eigenvalues; sigma=0.015, bins=0:0.001:2) -> (grid, Γ)`
Gaussian-smoothed density `Γ`, summing to one.

### `spectral_density_grid(eigenvalue_list; sigma=0.015, bins=0:0.001:2) -> (grid, M)`
Densities of many spectra as columns of `M`.

### `spectral_centroid(grid, density) -> Float64`
Weighted-mean eigenvalue.

---

## Embedding — `Embedding.jl`

### `interpolate_spectrum(eigenvalues, q) -> Vector{Float64}`
Sort + linearly interpolate to length `q`.

### `spectra_matrix(eigenvalue_list, q) -> Matrix{Float64}`
`M×q` stack of interpolated spectra (rows = reservoirs).

### `spectra_pca(S; maxoutdim=3) -> SpectraPCA`
PCA of `S`. `SpectraPCA`: `model, scores (d×M), explained_ratio, cumulative`.

### `cluster_spectra(S; max_clusters=10, rng=nothing) -> ClusterResult`
K-means with silhouette-selected `k`. `ClusterResult`: `assignments, k,
silhouette, scores`.

---

## Optimal-transport EMD — `SpectralEMD.jl`

### `spectral_emd(eigenvals1, eigenvals2) -> Float64`
EMD between two spectra (uniform mass, squared-Euclidean cost; `OptimalTransport`
+ `Tulip`).

### `pairwise_emd(eigenvalue_list; show_progress=true) -> Matrix{Float64}`
Symmetric pairwise EMD matrix.

### `emd_to_reference(eigenvalue_list, reference; show_progress=true) -> Vector{Float64}`
EMD of each spectrum to a fixed reference set.

---

## Modularity / cost — `Modularity.jl`

### `detect_communities(A) -> Vector{Int}`
Label-propagation partition of the weighted digraph.

### `directed_modularity(A, partition; gamma=1.0) -> Float64`
Directed Newman `Q` via `Graphs.modularity(distmx=A, γ=gamma)`.

### `connection_density(A; directed=true) -> Float64`
`E/(n(n−1))` (or `2E/(n(n−1))` when `directed=false`).

### `average_path_length(A) -> Float64`
Mean finite shortest-path length (Floyd–Warshall).

### `connection_cost(A; alpha=1.0, beta=1.0, gamma=1.0, directed=true) -> Float64`
`α Σ|A| + β·density + γ·ℓ`.

### `network_metrics(A; gamma=1.0, alpha=1.0, beta=1.0, cost_gamma=1.0, directed=true) -> NetworkMetrics`
All structural metrics in one pass. `NetworkMetrics`: `modularity, density,
path_length, cost, n_communities`.

---

## Multi-objective — `MultiObjective.jl`

### `normalize_metric(x) -> Vector{Float64}`
Min–max to `[0,1]` (constant → zeros).

### `normalize_metrics(; modularity, connection_cost, performance, generation) -> NormalizedMetrics`
Normalize all four metric vectors.

### `composite_objectives(x) -> NTuple{4,Float64}`
`(O₁,O₂,O₃,O₄)` at a normalized `(mod, cost, perf, gen)` point.

### `run_nsga2(; N=1000, p_cr=0.85, p_m=0.5) -> Matrix{Float64}`
NSGA-II over `[0,1]⁴`; rows are Pareto-optimal decision vectors.

### `closest_observed(optimized_positions, nm::NormalizedMetrics) -> Vector{Int}`
Index of the nearest observed reservoir to each optimized point.

---

## Reference graphs — `ReferenceGraphs.jl`

### `er_spectrum(n, p; rng=StableRNG(42)) -> Vector{Float64}`
Directed Erdős–Rényi RW-Laplacian spectrum.

### `ba_spectrum(n, m; rng=StableRNG(42)) -> Vector{Float64}`
Barabási–Albert (power-law) spectrum.

### `ws_spectrum(n, k, p; rng=StableRNG(42)) -> Vector{Float64}`
Watts–Strogatz (small-world) spectrum.

### `sbm_spectrum(block_sizes, probs; rng=StableRNG(42)) -> Vector{Float64}`
Stochastic Block Model spectrum.

### `reference_spectra(; n=500, p=0.05, m=2, k=6, p_rewire=0.1, within=0.8, between=0.05, rng=StableRNG(42)) -> Dict{Symbol,Vector{Float64}}`
All four ensembles keyed `:er, :ba, :ws, :sbm`.

---

## Error statistics — `ErrorStats.jl`

### `collect_generation_errors(matdir, gens) -> Dict{Int,Vector{Float64}}`
`J` of every individual per generation.

### `log_error_distribution(J; drop_nonfinite=true) -> Vector{Float64}`
`log10 J` with `Inf` fitness dropped.

### `generation_error_stats(matdir, gens) -> Vector{GenerationErrorStats}`
Per-generation `log10 J` summary. `GenerationErrorStats`: `generation, n, n_inf,
mean, median, std, min, max`.
