# =============================================================================
# SpectralEMD.jl
# -----------------------------------------------------------------------------
# Optimal-transport (Earth Mover's Distance) between reservoir Laplacian spectra.
#
# Ports the `spectral_distance` helper and the pairwise / to-reference EMD loops
# from `compute_laplacianSpectrum_stratifiedSelection.jl`.
#
# Paper §"Spectral distance using optimal transport":
#   Each spectrum is an empirical measure with equal mass 1/m on each eigenvalue.
#   EMD(Λ¹,Λ²) = min_{γ∈Π(a,b)} Σ_ij γ_ij C_ij,  with quadratic ground cost
#   C_ij = (λ_i¹ − λ_j²)².  Solved by OptimalTransport.emd2 with Tulip as the LP
#   solver. This is the unregularized OT cost (squared-2-Wasserstein-type).
# =============================================================================

module SpectralEMD

using OptimalTransport: emd2
using Tulip
using Distances: pairwise, SqEuclidean
using ProgressMeter: Progress, next!, finish!

export spectral_emd, pairwise_emd, emd_to_reference

"""
    spectral_emd(eigenvals1, eigenvals2) -> Float64

Earth Mover's Distance between two eigenvalue sets treated as uniform empirical
measures, with squared-Euclidean ground cost `C_ij = (λ_i − λ_j)²`. Port of the
original `spectral_distance`:

    μ = fill(1/M, M);  ν = fill(1/N, N)
    C = pairwise(SqEuclidean(), λ1', λ2'; dims=2)
    emd2(μ, ν, C, Tulip.Optimizer())

Only the real parts are used. Inputs need not be sorted or of equal length.
"""
function spectral_emd(eigenvals1::AbstractVector, eigenvals2::AbstractVector)
    e1 = real.(collect(eigenvals1))
    e2 = real.(collect(eigenvals2))
    M = length(e1)
    N = length(e2)
    μ = fill(1 / M, M)
    ν = fill(1 / N, N)
    # pairwise expects features × points; eigenvalues are 1-D, so reshape to 1×n
    C = pairwise(SqEuclidean(), reshape(e1, 1, :), reshape(e2, 1, :); dims=2)
    return emd2(μ, ν, C, Tulip.Optimizer())
end

"""
    pairwise_emd(eigenvalue_list; show_progress=true) -> Matrix{Float64}

Symmetric matrix of pairwise spectral EMD between every pair of spectra in
`eigenvalue_list`. The diagonal is zero and only the upper triangle is computed
(then mirrored), as in the original pairwise loop.
"""
function pairwise_emd(eigenvalue_list::AbstractVector; show_progress::Bool=true)
    n = length(eigenvalue_list)
    D = zeros(Float64, n, n)
    pairs = [(i, j) for i in 1:n for j in (i+1):n]
    prog = show_progress ? Progress(length(pairs); desc="Pairwise spectral EMD...") : nothing
    for (i, j) in pairs
        d = spectral_emd(eigenvalue_list[i], eigenvalue_list[j])
        D[i, j] = d
        D[j, i] = d
        prog === nothing || next!(prog)
    end
    prog === nothing || finish!(prog)
    return D
end

"""
    emd_to_reference(eigenvalue_list, reference; show_progress=true) -> Vector{Float64}

EMD from each spectrum in `eigenvalue_list` to a fixed `reference` eigenvalue
set (e.g. the pooled generation-0 spectrum). Returns one distance per spectrum.
"""
function emd_to_reference(eigenvalue_list::AbstractVector, reference::AbstractVector;
                          show_progress::Bool=true)
    ref = sort(real.(collect(reference)))
    n = length(eigenvalue_list)
    out = Vector{Float64}(undef, n)
    prog = show_progress ? Progress(n; desc="EMD to reference...") : nothing
    for i in 1:n
        out[i] = spectral_emd(sort(real.(collect(eigenvalue_list[i]))), ref)
        prog === nothing || next!(prog)
    end
    prog === nothing || finish!(prog)
    return out
end

end # module SpectralEMD
