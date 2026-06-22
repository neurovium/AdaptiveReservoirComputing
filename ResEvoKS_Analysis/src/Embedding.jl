# =============================================================================
# Embedding.jl
# -----------------------------------------------------------------------------
# Fixed-length spectral representation, PCA across reservoirs, and k-means
# clustering of the spectra.
#
# Ports `interpolate_eigenvalues`, the `MultivariateStats.PCA` fit/transform,
# and `apply_kmeans_clustering` (silhouette-selected k) from
# `compute_laplacianSpectrum_stratifiedSelection.jl`.
#
# Paper §"Principal component analysis of Laplacian spectra":
#   Each eigenvalue vector is sorted and interpolated to a common length q,
#   giving Λ̃_i ∈ ℝ^q. The interpolated spectra are stacked into S (M×q) and PCA
#   is applied; the first few PCs visualize the population in spectral space.
# =============================================================================

module Embedding

using Statistics
using LinearAlgebra
using Interpolations: linear_interpolation, Flat
using MultivariateStats: fit, PCA, projection, principalvars, tprincipalvar,
                         transform
using Clustering: kmeans, silhouettes
using Distances: pairwise, Euclidean

export interpolate_spectrum, spectra_matrix, spectra_pca, SpectraPCA,
       cluster_spectra, ClusterResult

# -----------------------------------------------------------------------------
# Fixed-length interpolation
# -----------------------------------------------------------------------------

"""
    interpolate_spectrum(eigenvalues, q) -> Vector{Float64}

Sort `eigenvalues` ascending and linearly interpolate them to exactly `q`
points (flat extrapolation at the ends), yielding the fixed-length spectral
representation `Λ̃ ∈ ℝ^q`. This makes spectra of reservoirs of different sizes
directly comparable. Port of `interpolate_eigenvalues`.
"""
function interpolate_spectrum(eigenvalues::AbstractVector, q::Integer)
    ev = sort(real.(eigenvalues))
    m = length(ev)
    m == q && return Float64.(ev)
    m == 1 && return fill(Float64(ev[1]), q)
    itp = linear_interpolation(1:m, ev; extrapolation_bc=Flat())
    xs = range(1, stop=m, length=q)
    return [Float64(itp(x)) for x in xs]
end

"""
    spectra_matrix(eigenvalue_list, q) -> Matrix{Float64}

Interpolate every spectrum in `eigenvalue_list` to length `q` and stack as rows,
returning the `M×q` matrix `S` whose row `i` is `Λ̃_i`.
"""
function spectra_matrix(eigenvalue_list::AbstractVector, q::Integer)
    M = length(eigenvalue_list)
    S = Matrix{Float64}(undef, M, q)
    for i in 1:M
        S[i, :] = interpolate_spectrum(eigenvalue_list[i], q)
    end
    return S
end

# -----------------------------------------------------------------------------
# PCA
# -----------------------------------------------------------------------------

"""
    SpectraPCA

Result of `spectra_pca`.

# Fields
- `model::PCA`               : the fitted `MultivariateStats.PCA`.
- `scores::Matrix{Float64}`  : `d×M` PC coordinates (column `i` = reservoir `i`).
- `explained_ratio::Vector{Float64}` : variance explained per PC.
- `cumulative::Vector{Float64}`      : cumulative explained-variance ratio.
"""
struct SpectraPCA
    model::PCA
    scores::Matrix{Float64}
    explained_ratio::Vector{Float64}
    cumulative::Vector{Float64}
end

"""
    spectra_pca(S; maxoutdim=3) -> SpectraPCA

Principal-component analysis of the stacked interpolated spectra `S` (`M×q`,
rows = reservoirs). PCA is fit on the `q×M` transpose (features = eigenvalue
positions, observations = reservoirs), matching the original
`fit(PCA, eigenvalue_matrix'; maxoutdim=...)`. Returns the model, the PC scores
(`d×M`), and the explained-variance breakdown.
"""
function spectra_pca(S::AbstractMatrix{<:Real}; maxoutdim::Integer=3)
    X = permutedims(Float64.(S))            # q × M  (features × observations)
    model = fit(PCA, X; maxoutdim=maxoutdim)
    scores = transform(model, X)            # d × M
    pv = principalvars(model)
    tv = tprincipalvar(model)
    ratio = pv ./ tv
    return SpectraPCA(model, scores, ratio, cumsum(ratio))
end

# -----------------------------------------------------------------------------
# K-means clustering (silhouette-selected k)
# -----------------------------------------------------------------------------

"""
    ClusterResult

Result of `cluster_spectra`.

# Fields
- `assignments::Vector{Int}` : cluster label per reservoir.
- `k::Int`                   : selected number of clusters.
- `silhouette::Float64`      : mean silhouette of the selected `k`.
- `scores::Dict{Int,Float64}`: mean silhouette for each tried `k`.
"""
struct ClusterResult
    assignments::Vector{Int}
    k::Int
    silhouette::Float64
    scores::Dict{Int,Float64}
end

"""
    cluster_spectra(S; max_clusters=10, rng=nothing) -> ClusterResult

K-means clustering of the spectra `S` (`M×q`, rows = reservoirs), selecting the
number of clusters `k ∈ 2:max_clusters` that maximizes the mean silhouette.
Port of `apply_kmeans_clustering`.

K-means runs on the `q×M` transpose (each reservoir is one observation /
column), and silhouettes are computed from the pairwise Euclidean distance
matrix between reservoirs. Pass `rng` to seed k-means for reproducibility.
"""
function cluster_spectra(S::AbstractMatrix{<:Real};
                         max_clusters::Integer=10, rng=nothing)
    X = permutedims(Float64.(S))             # q × M
    M = size(X, 2)
    D = pairwise(Euclidean(), X; dims=2)     # M × M reservoir distances
    upper = min(max_clusters, M - 1)

    best_k = 2
    best_sil = -Inf
    best_assign = ones(Int, M)
    scores = Dict{Int,Float64}()

    for k in 2:upper
        model = rng === nothing ? kmeans(X, k) : kmeans(X, k; rng=rng)
        sil = mean(silhouettes(model.assignments, D))
        scores[k] = sil
        if sil > best_sil
            best_sil = sil
            best_k = k
            best_assign = copy(model.assignments)
        end
    end

    return ClusterResult(best_assign, best_k, best_sil, scores)
end

end # module Embedding
