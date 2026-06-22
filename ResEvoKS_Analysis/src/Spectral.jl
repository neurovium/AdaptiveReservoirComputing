# =============================================================================
# Spectral.jl
# -----------------------------------------------------------------------------
# Random-walk normalized graph Laplacian, its eigenvalues, and the smoothed
# eigenvalue density.
#
# Ports `compute_laplacian_eigenvals` and the Gaussian-kernel
# `create_spectral_density_simple` from
# `compute_laplacianSpectrum_stratifiedSelection.jl`.
#
# Paper §"Normalized random-walk Laplacian spectrum":
#   d_i = Σ_j A_ij,  D = diag(d),  L_rw = I − D^{-1} A.
#   Eigenvalues via ARPACK; analyses use the real parts (directed weighted
#   networks can have complex eigenvalues). For row-stochastic D^{-1}A the
#   eigenvalues lie in the disk centered at 1.
#
# Paper §"Smoothed eigenvalue distributions":
#   Γ(x) = (1/m) Σ_i N(x; λ_i, s²),  area-normalized to 1.
#
# Note on the eigensolver: the paper specifies ARPACK. ARPACK is *not*
# thread-safe (this is why the simulation's spectral-radius step uses KrylovKit),
# but the analysis runs single-threaded, so ARPACK is used here to match the
# paper exactly. A dense fallback is provided for tiny / non-convergent cases.
# =============================================================================

module Spectral

using LinearAlgebra
using SparseArrays
using Arpack: eigs
using Distributions: Normal, pdf

export rw_laplacian, laplacian_eigenvalues, smoothed_density,
       spectral_centroid, spectral_density_grid

# -----------------------------------------------------------------------------
# Random-walk Laplacian
# -----------------------------------------------------------------------------

"""
    rw_laplacian(A) -> L

Random-walk normalized Laplacian `L = I − D⁻¹A` of the directed weighted
adjacency matrix `A`, where `D = diag(dᵢ)` and `dᵢ = Σⱼ Aᵢⱼ` is the row weight
(out-strength) of node `i`.

Zero-row-weight nodes (sinks) would divide by zero; following the standard
regularization (and the reference Python port) their inverse degree is set to
`0`, so such a row of `D⁻¹A` is all zeros. Returns a sparse matrix when `A` is
sparse.
"""
function rw_laplacian(A::AbstractMatrix)
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("A must be square"))
    deg = vec(sum(A, dims=2))
    dinv = map(d -> d == 0 ? 0.0 : 1.0 / d, deg)
    Dinv = spdiagm(0 => dinv)
    Iₙ = spdiagm(0 => ones(n))
    return Iₙ - Dinv * A
end

# -----------------------------------------------------------------------------
# Eigenvalues
# -----------------------------------------------------------------------------

"""
    laplacian_eigenvalues(A; nev=size(A,1)-2, which=:SM, dense_threshold=20) -> Vector{Float64}

Real parts of the eigenvalues of the random-walk Laplacian of `A`.

By default it reproduces the original call `eigs(L, nev=n-2, which=:SM)` —
the `n−2` smallest-magnitude eigenvalues via ARPACK. For very small matrices
(`n ≤ dense_threshold`) or if ARPACK fails to converge, it falls back to a dense
`eigvals`. Returned eigenvalues are sorted ascending.

Pass `nev=:all` to request the full spectrum (dense).
"""
function laplacian_eigenvalues(A::AbstractMatrix;
                               nev::Union{Integer,Symbol}=size(A, 1) - 2,
                               which::Symbol=:SM,
                               dense_threshold::Integer=20)
    L = rw_laplacian(A)
    n = size(L, 1)

    if nev === :all || n <= dense_threshold
        vals = eigvals(Matrix(L))
        return sort!(real.(vals))
    end

    k = min(Int(nev), n - 2)
    k < 1 && (k = 1)
    try
        vals, _ = eigs(L; nev=k, which=which)
        return sort!(real.(vals))
    catch err
        @warn "ARPACK eigs failed; falling back to dense eigvals" error=err size=n
        vals = eigvals(Matrix(L))
        return sort!(real.(vals))
    end
end

# -----------------------------------------------------------------------------
# Smoothed spectral density
# -----------------------------------------------------------------------------

"""
    smoothed_density(eigenvalues; sigma=0.015, bins=0:0.001:2) -> (grid, Γ)

Gaussian-smoothed eigenvalue density (paper Eq. for `Γ`):

    Γ(x) = (1/m) Σᵢ N(x; λᵢ, σ²),

evaluated on `bins` and renormalized so it sums to one (`Γ ./= sum(Γ)`, matching
the original discrete normalization). Returns the evaluation grid and `Γ`.
Only the real parts of `eigenvalues` are used.
"""
function smoothed_density(eigenvalues::AbstractVector;
                          sigma::Real=0.015,
                          bins=0:0.001:2)
    grid = collect(float.(bins))
    Γ = zeros(Float64, length(grid))
    for λ in real.(eigenvalues)
        Γ .+= pdf.(Normal(λ, sigma), grid)
    end
    s = sum(Γ)
    s > 0 && (Γ ./= s)
    return grid, Γ
end

"""
    spectral_density_grid(eigenvalue_list; sigma=0.015, bins=0:0.001:2) -> (grid, M)

Apply `smoothed_density` to many spectra and stack the densities as columns.
`M[:, i]` is the density of `eigenvalue_list[i]`. Convenient for generational
averaging and the spectral-evolution heatmap.
"""
function spectral_density_grid(eigenvalue_list::AbstractVector;
                               sigma::Real=0.015, bins=0:0.001:2)
    grid = collect(float.(bins))
    M = Matrix{Float64}(undef, length(grid), length(eigenvalue_list))
    for (i, ev) in enumerate(eigenvalue_list)
        _, Γ = smoothed_density(ev; sigma=sigma, bins=grid)
        M[:, i] = Γ
    end
    return grid, M
end

"""
    spectral_centroid(grid, density) -> Float64

Center of mass `Σ x·Γ(x) / Σ Γ(x)` of a (grid, density) pair — the weighted-mean
eigenvalue used as a one-number summary of a smoothed spectrum.
"""
spectral_centroid(grid::AbstractVector, density::AbstractVector) =
    sum(grid .* density) / sum(density)

end # module Spectral
