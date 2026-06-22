# =============================================================================
# ReferenceGraphs.jl
# -----------------------------------------------------------------------------
# Random-walk Laplacian spectra of canonical graph ensembles, used to position
# the evolved reservoirs against an "SBM-like envelope".
#
# Julia port of `normalizedlaplacianrandomgraphs.py` (networkx). Each generator
# builds a graph, forms the same random-walk Laplacian L = I − D⁻¹A used for the
# reservoirs (Spectral.rw_laplacian / Spectral.laplacian_eigenvalues), and
# returns the real parts of its eigenvalues.
#
# Paper §"Reservoirs occupy an SBM-like spectral signature class": the evolved
# reservoir spectra are compared with Erdős–Rényi, Barabási–Albert,
# Watts–Strogatz, and Stochastic Block Model references.
# =============================================================================

module ReferenceGraphs

using Random
using StableRNGs
using LinearAlgebra: eigvals
using Graphs: erdos_renyi, barabasi_albert, watts_strogatz, stochastic_block_model,
              adjacency_matrix, nv
using ..Spectral: rw_laplacian

export er_spectrum, ba_spectrum, ws_spectrum, sbm_spectrum, reference_spectra

# Dense real eigenvalues of the random-walk Laplacian of a Graphs.jl graph.
function _rw_eigs(g)
    A = Float64.(adjacency_matrix(g))
    return sort!(real.(eigvals(Matrix(rw_laplacian(A)))))
end

"""
    er_spectrum(n, p; rng=StableRNG(42)) -> Vector{Float64}

Random-walk Laplacian spectrum of a **directed** Erdős–Rényi graph `G(n, p)`
(density `p`). Mirrors `get_random_spectrum` (directed ER). Defaults in the
Python: `n=500, p=0.05`.
"""
er_spectrum(n::Integer, p::Real; rng::AbstractRNG=StableRNG(42)) =
    _rw_eigs(erdos_renyi(n, p; is_directed=true, rng=rng))

"""
    ba_spectrum(n, m; rng=StableRNG(42)) -> Vector{Float64}

Random-walk Laplacian spectrum of a Barabási–Albert (power-law, undirected)
graph with `m` edges added per new node. Mirrors `get_powerlaw_spectrum`.
Python default: `m=2`.
"""
ba_spectrum(n::Integer, m::Integer; rng::AbstractRNG=StableRNG(42)) =
    _rw_eigs(barabasi_albert(n, m; rng=rng))

"""
    ws_spectrum(n, k, p; rng=StableRNG(42)) -> Vector{Float64}

Random-walk Laplacian spectrum of a Watts–Strogatz small-world graph: ring of
`n` nodes each joined to `k` neighbors, each edge rewired with probability `p`.
Mirrors `get_watts_strogatz_spectrum`. Python defaults: `k=6, p=0.1`.
"""
ws_spectrum(n::Integer, k::Integer, p::Real; rng::AbstractRNG=StableRNG(42)) =
    _rw_eigs(watts_strogatz(n, k, p; rng=rng))

"""
    sbm_spectrum(block_sizes, probs; rng=StableRNG(42)) -> Vector{Float64}

Random-walk Laplacian spectrum of a Stochastic Block Model with community sizes
`block_sizes` and between/within block connection probabilities `probs` (a
symmetric matrix). Mirrors `get_sbm_spectrum`. Python default: two equal blocks
of `n/2` with within-block prob 0.8 and between-block prob 0.05.
"""
function sbm_spectrum(block_sizes::AbstractVector{<:Integer},
                      probs::AbstractMatrix{<:Real}; rng::AbstractRNG=StableRNG(42))
    g = stochastic_block_model(probs, collect(block_sizes); rng=rng)
    return _rw_eigs(g)
end

"""
    reference_spectra(; n=500, p=0.05, m=2, k=6, p_rewire=0.1,
                        within=0.8, between=0.05, rng=StableRNG(42)) -> Dict{Symbol,Vector{Float64}}

Convenience bundle: spectra of all four reference ensembles at the Python
defaults, keyed `:er, :ba, :ws, :sbm`. The SBM uses two equal blocks of `n÷2`.
"""
function reference_spectra(; n::Integer=500, p::Real=0.05, m::Integer=2,
                            k::Integer=6, p_rewire::Real=0.1,
                            within::Real=0.8, between::Real=0.05,
                            rng::AbstractRNG=StableRNG(42))
    half = n ÷ 2
    probs = [within between; between within]
    return Dict(
        :er  => er_spectrum(n, p; rng=rng),
        :ba  => ba_spectrum(n, m; rng=rng),
        :ws  => ws_spectrum(n, k, p_rewire; rng=rng),
        :sbm => sbm_spectrum([half, n - half], probs; rng=rng),
    )
end

end # module ReferenceGraphs
