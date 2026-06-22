# =============================================================================
# Reservoir.jl
# -----------------------------------------------------------------------------
# Construction of the recurrent reservoir matrix and the input weight matrix.
#
# Julia port of:
#   * `generate_reservoir.m`                         -> generate_reservoir
#   * the inline spectral-radius rescaling in        -> scale_spectral_radius!
#     `KS64D_runOptimizePredESN.m` (the eigs loop)
#   * the inline input-weight block construction in  -> generate_input_weights
#     `KS64D_runOptimizePredESN.m`
# =============================================================================

module Reservoir

using SparseArrays
using LinearAlgebra
using Random
using StableRNGs
using KrylovKit: eigsolve

export generate_reservoir, scale_spectral_radius!, build_reservoir,
       generate_input_weights, ReservoirParams, spectral_radius

"""
    ReservoirParams

Decoded reservoir construction hyperparameters for a single GA individual.
Mirrors the MATLAB `resparams` struct so the saved `.mat` files have identical
field names.

# Fields
- `num_inputs::Int` : number of input/output channels (`= N` of the KS grid).
- `radius::Float64` : target spectral radius `ρ` of the recurrent matrix.
- `degree::Int`     : target average connection degree `d`.
- `N::Int`          : reservoir size `n_r` (snapped to a multiple of `num_inputs`).
- `sigma::Float64`  : input scaling `σ`.
- `beta::Float64`   : ridge regularization `β`.
"""
Base.@kwdef struct ReservoirParams
    num_inputs::Int
    radius::Float64
    degree::Int
    N::Int
    sigma::Float64
    beta::Float64
end

"""
    generate_reservoir(size, degree; rng) -> (A, C)

Port of `generate_reservoir.m`. Build a sparse, directed, nonnegative recurrent
matrix `A` of dimension `size × size` whose nonzero entries are uniform on
`(0, 1)` and whose expected average degree is `degree`. `C = (A .> 0)` is the
boolean connectivity mask.

The sparsity (nonzero probability) is `degree / size`, matching
`sparsity = degree/size; A = sprand(size, size, sparsity)`.
"""
function generate_reservoir(size::Integer, degree::Integer;
                            rng::AbstractRNG=StableRNG(0))
    sparsity = degree / size
    # sprand with a uniform(0,1) value generator reproduces MATLAB `sprand`.
    A = sprand(rng, Float64, size, size, sparsity)
    C = A .> 0
    return A, C
end

"""
    spectral_radius(A; tol=1e-9, rng=...) -> Float64

Dominant eigenvalue magnitude `max_ℓ |λ_ℓ(A)|`, the spectral radius `ρ(A)`. This
mirrors MATLAB's `max(abs(eigs(A,1,'lm')))`.

The sparse Arnoldi iteration is provided by **KrylovKit.jl** (`eigsolve`), a
pure-Julia, **thread-safe** eigensolver. This is a deliberate substitution for
ARPACK (`Arpack.jl`): ARPACK wraps a Fortran library with shared internal state
that is *not* thread-safe, and calling it concurrently from the parallel GA
segfaults. KrylovKit gives the same dominant-eigenvalue result without that
hazard. For very small matrices a dense `eigvals` is used directly.

A deterministic start vector (`rng`) keeps the result reproducible and avoids
contention on the global RNG under threading.
"""
function spectral_radius(A::AbstractMatrix; tol::Real=1e-9,
                         rng::AbstractRNG=StableRNG(0))
    n = size(A, 1)
    if n <= 20
        return maximum(abs, eigvals(Matrix(A)))
    end
    x0 = randn(rng, eltype(A), n)                  # deterministic Arnoldi seed
    vals, _, info = eigsolve(A, x0, 1, :LM; tol=tol, maxiter=300)
    if isempty(vals)
        error("eigensolver returned no eigenvalues (converged=$(info.converged))")
    end
    return maximum(abs, vals)
end

"""
    scale_spectral_radius!(A, radius; tol=1e-3) -> A

Rescale `A` in place so that its spectral radius equals `radius`, reproducing

    lambda = max(abs(eigs(A,1,'lm')));  A = (A ./ lambda) * radius;

Returns the rescaled `A`. Throws if the dominant eigenvalue is zero (the caller
retry-loop should then regenerate the reservoir).
"""
function scale_spectral_radius!(A::AbstractMatrix, radius::Real; tol::Real=1e-9,
                                rng::AbstractRNG=StableRNG(0))
    lambda = spectral_radius(A; tol=tol, rng=rng)
    if lambda == 0 || !isfinite(lambda)
        error("non-finite or zero spectral radius (λ = $lambda)")
    end
    A .*= (radius / lambda)
    return A
end

"""
    build_reservoir(N, degree, radius; rng, max_tries=50, tol=1e-3) -> A

Generate and spectrally-scale a reservoir, retrying on eigensolver failure.
This reproduces the `while success == 0 ... try eigs ... catch regenerate end`
loop in `KS64D_runOptimizePredESN.m`.
"""
function build_reservoir(N::Integer, degree::Integer, radius::Real;
                         rng::AbstractRNG=StableRNG(0),
                         max_tries::Integer=50, tol::Real=1e-9)
    for attempt in 1:max_tries
        A, _ = generate_reservoir(N, degree; rng=rng)
        try
            scale_spectral_radius!(A, radius; tol=tol, rng=rng)
            return A
        catch err
            @debug "regenerating reservoir (attempt $attempt): $err"
        end
    end
    error("failed to build a well-conditioned reservoir after $max_tries tries")
end

"""
    generate_input_weights(N, num_inputs, sigma; rng) -> Matrix{Float64}

Port of the inline input-weight construction. Each input channel `i` drives its
own disjoint block of `q = N / num_inputs` reservoir nodes, with weights drawn
uniformly on `[-sigma, sigma]`; all other entries are zero.

Requires `N` divisible by `num_inputs` (guaranteed by the size-snapping in the
evaluation step).
"""
function generate_input_weights(N::Integer, num_inputs::Integer, sigma::Real;
                                rng::AbstractRNG=StableRNG(0))
    @assert N % num_inputs == 0 "reservoir size N=$N must be divisible by num_inputs=$num_inputs"
    q = N ÷ num_inputs
    w_in = zeros(Float64, N, num_inputs)
    for i in 1:num_inputs
        ip = sigma .* (-1 .+ 2 .* rand(rng, q))   # uniform [-sigma, sigma]
        w_in[(i - 1) * q + 1:i * q, i] .= ip
    end
    return w_in
end

end # module Reservoir
