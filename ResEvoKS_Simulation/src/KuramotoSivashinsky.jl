# =============================================================================
# KuramotoSivashinsky.jl
# -----------------------------------------------------------------------------
# Generation of Kuramoto-Sivashinsky (KS) spatiotemporal-chaos time series.
#
# Julia port of `kuramoto_sivashinsky_solve.m` from the MATLAB `64D-KS-Sim` suite.
#
# The 1-D Kuramoto-Sivashinsky equation on a periodic domain of length `d` is
#
#       u_t + u u_x + u_xx + u_xxxx = 0 .
#
# We integrate it with the fourth-order Exponential Time-Differencing
# Runge-Kutta method (ETDRK4) of Cox & Matthews (2002), using the
# contour-integral evaluation of the ETD coefficients due to Kassam &
# Trefethen (2005). Spatial derivatives are computed spectrally.
#
# -----------------------------------------------------------------------------
# IMPORTANT NUMERICAL NOTE (why we use *real* FFTs)
# -----------------------------------------------------------------------------
# `kuramoto_sivashinsky_solve.m` and the Python demo use the *complex* FFT (`fft`/`ifft`).
# That scheme is on a numerical knife's edge: the KS field `u` is real, so its
# spectrum must be Hermitian-symmetric, but floating-point roundoff in a generic
# complex FFT slowly injects a spurious anti-Hermitian component that is
# amplified by the linearly unstable KS modes and could blow the integration up at
# `t ≈ 360` (≈ step 1450 at Δt = 0.25). The Python demo dodged this by switching
# from `numpy.fft` to the (coincidentally more symmetric) `scipy.fft`; with a
# plain complex FFT, Julia's FFTW behaves like `numpy.fft` and diverges.
#
# This port instead uses the **real FFT** (`rfft`/`irfft`), which represents the
# field by its nonnegative-wavenumber half and therefore *enforces* Hermitian
# symmetry by construction. This is algebraically the identical Trefethen
# ETDRK4 scheme (same wavenumbers, same `Q, f1, f2, f3` coefficients, same
# stages) but is numerically robust and FFT-backend-independent — stable to
# 100k+ steps at Δt = 0.25. The resulting trajectory is a valid KS solution;
# because KS is chaotic, no implementation reproduces another's trajectory
# pointwise across machines, and the paper relies on statistical/structural
# properties rather than a specific trajectory.
# =============================================================================

module KuramotoSivashinsky

using FFTW
using Random
using StableRNGs

export KSModelParams, solve_ks, random_initial_condition

"""
    KSModelParams

Container for the Kuramoto-Sivashinsky integration parameters. Mirrors the
`ModelParams` MATLAB struct.

# Fields
- `N::Int`       : number of spatial grid points (e.g. `64`).
- `d::Float64`   : periodic domain length `L` (e.g. `22.0`).
- `tau::Float64` : integration time step `h` / `Δt` (e.g. `0.25`).
- `nstep::Int`   : number of ETDRK4 time steps to generate.

# Notes
The example here uses `Δt = 0.25`, while the original MATLAB
driver `KS64D_prepDataAndRun.m` used `tau = 0.15`. Both are valid; the value is
a free parameter exposed here. We default to the Methods value `0.25`.
"""
Base.@kwdef struct KSModelParams
    N::Int       = 64
    d::Float64   = 22.0
    tau::Float64 = 0.25
    nstep::Int   = 100_000
end

"""
    random_initial_condition(N; amplitude=0.6, rng=...) -> Vector{Float64}

Reproduce the MATLAB initial condition `0.6*(-1 + 2*rand(1,N))`, i.e. a random
field uniform on `[-amplitude, amplitude]`. Pass an explicit `rng` (e.g. a
`StableRNG`) for reproducible trajectories.
"""
function random_initial_condition(N::Integer; amplitude::Real=0.6,
                                  rng::AbstractRNG=StableRNG(1234))
    return amplitude .* (-1 .+ 2 .* rand(rng, N))
end

"""
    solve_ks(init, p::KSModelParams) -> Matrix{Float64}

Integrate the KS equation from initial condition `init` (length `N`) and return
the trajectory as an `N × nstep` matrix `u`, where `u[i, n]` is the field value
at spatial grid point `i` and time step `n`.

This **space × time** orientation matches the `measurements` matrix consumed by
the reservoir pipeline (`num_inputs × time`). It folds in the transpose that the
MATLAB driver applied to `kuramoto_sivashinsky_solve`'s output, so callers need not transpose.

# Algorithm (ETDRK4; same scheme as `kuramoto_sivashinsky.m`, real-FFT formulation)
1. Nonnegative wavenumbers `k = (0:N/2) · (2π/d)`, with the Nyquist entry zeroed
   (Trefethen convention: removes the ambiguous odd-derivative Nyquist mode).
2. Linear Fourier multiplier `L = k² − k⁴`.
3. Exponential integrators `E = exp(hL)`, `E2 = exp(hL/2)`.
4. ETD coefficients `Q, f1, f2, f3` via mean over `M = 16` contour points.
5. Nonlinear term `−½ ∂_x(u²)` evaluated in physical space each step.
"""
function solve_ks(init::AbstractVector{<:Real}, p::KSModelParams)
    N = p.N
    @assert length(init) == N "init length ($(length(init))) must equal N ($N)"
    @assert iseven(N) "N must be even for the real-FFT formulation (got N=$N)"

    h = p.tau                       # time step (MATLAB: h = ModelParams.tau)
    d = p.d
    nmax = p.nstep

    # --- nonnegative wavenumbers for the real FFT ---------------------------
    # rfft of a length-N real vector has N/2 + 1 modes (k = 0 … N/2).
    kvec = Float64.(0:(N ÷ 2)) .* (2π / d)
    kvec[end] = 0.0                 # zero the Nyquist (matches Trefethen's [..0..])

    # --- linear Fourier multiplier  L = k^2 - k^4 ---------------------------
    L = kvec .^ 2 .- kvec .^ 4

    # --- exponential integrators --------------------------------------------
    E  = exp.(h .* L)
    E2 = exp.(h .* L ./ 2)

    # --- ETDRK4 coefficients by contour integration (M points) --------------
    M = 16
    # r = exp(iπ (j - 1/2) / M), j = 1..M  (roots on the upper unit circle)
    r = exp.(im .* π .* ((1:M) .- 0.5) ./ M)
    # LR[n, j] = h*L[n] + r[j]   ((N/2+1) × M)
    LR = (h .* L) .+ transpose(r)

    # Each coefficient is h * real(mean over the M contour points)
    Q  = h .* real.(vec(_mean_cols((exp.(LR ./ 2) .- 1) ./ LR)))
    f1 = h .* real.(vec(_mean_cols((-4 .- LR .+ exp.(LR) .* (4 .- 3 .* LR .+ LR .^ 2)) ./ LR .^ 3)))
    f2 = h .* real.(vec(_mean_cols((2 .+ LR .+ exp.(LR) .* (-2 .+ LR)) ./ LR .^ 3)))
    f3 = h .* real.(vec(_mean_cols((-4 .- 3 .* LR .- LR .^ 2 .+ exp.(LR) .* (4 .- LR)) ./ LR .^ 3)))

    # --- nonlinear prefactor  g = -0.5i k -----------------------------------
    g = -0.5im .* kvec

    # --- main time-stepping loop (ETDRK4) -----------------------------------
    v = rfft(Float64.(init))                  # spectral state (length N/2+1)
    uu = Matrix{Float64}(undef, N, nmax)      # physical trajectory (N × nstep)

    # Nonlinear term in spectral space:  N(v) = g .* rfft( (irfft v)^2 ).
    @inline nonlinear(vv) = g .* rfft(irfft(vv, N) .^ 2)

    @inbounds for n in 1:nmax
        Nv = nonlinear(v)
        a  = E2 .* v .+ Q .* Nv
        Na = nonlinear(a)
        b  = E2 .* v .+ Q .* Na
        Nb = nonlinear(b)
        c  = E2 .* a .+ Q .* (2 .* Nb .- Nv)
        Nc = nonlinear(c)
        v  = E .* v .+ Nv .* f1 .+ 2 .* (Na .+ Nb) .* f2 .+ Nc .* f3
        uu[:, n] = irfft(v, N)
    end

    return uu
end

# Column-wise mean over the M contour points (dims=2), matching MATLAB
# `mean(., 2)`. Returns an (N/2+1) × 1 column so `vec(...)` gives a vector.
@inline _mean_cols(A::AbstractMatrix) = sum(A, dims=2) ./ size(A, 2)

end # module KuramotoSivashinsky
