# =============================================================================
# Activation.jl
# -----------------------------------------------------------------------------
# Reservoir node activation functions.
#
# Julia port of `ActFunc.m` and `generate_activationFunction.m`.
#
# The production runs in `KS64D_runOptimizePredESN.m` hard-code the activation
# to `tanh` (`ActFunc = @(x) tanh(x)`). The parametric generalized-logistic
# activation from the MATLAB code is retained here for completeness and
# experimentation, but `tanh` is the default used to reproduce the paper.
# =============================================================================

module Activation

export tanh_activation, generalized_logistic, rescale_minmax

"""
    tanh_activation(x)

Elementwise hyperbolic tangent — the activation used in all paper runs
(Eq. for the reservoir update uses `f(·) = tanh`). Works on scalars or arrays.
"""
@inline tanh_activation(x) = tanh.(x)

"""
    rescale_minmax(x, lb, ub)

Affine min-max rescaling of `x` into `[lb, ub]`, reproducing MATLAB's
`rescale(x, lb, ub)`. If `x` is constant, returns `lb` (matching MATLAB's
degenerate behavior of mapping a flat signal to the lower bound).
"""
function rescale_minmax(x::AbstractArray, lb::Real, ub::Real)
    lo, hi = extrema(x)
    if hi == lo
        return fill(float(lb), size(x))
    end
    return lb .+ (ub - lb) .* (x .- lo) ./ (hi - lo)
end

"""
    generalized_logistic(g, k, n, lb, ub) -> Function

Build the parametric generalized-logistic activation from
`generate_activationFunction.m`:

    f(x) = rescale( g / (1 + exp(-k*((x+1) - n))),  lb, ub )

# Arguments
- `g`  : gain (redundant with `[lb, ub]`; usually `1`).
- `k`  : nonlinearity slope (higher → closer to a step function).
- `n`  : horizontal shift relative to `0` (`>1` shifts right, `<1` left).
- `lb`, `ub` : output range (e.g. `[-1, 1]` for a tanh-like map,
  `[0, 1]` for a sigmoid-like map).

# Examples
- `generalized_logistic(2, 2, 1, -1, 1)` ≈ `tanh`.
- `generalized_logistic(1, 2, 1, 0, 1)`  ≈ logistic sigmoid.

Note: the rescaling is applied over the whole input array (as MATLAB's
`rescale` does), so this is a *vector* nonlinearity, not a strictly elementwise
one. Use `tanh_activation` for the paper configuration.
"""
function generalized_logistic(g::Real, k::Real, n::Real, lb::Real, ub::Real)
    return function (x::AbstractArray)
        s = g .* (1 ./ (1 .+ exp.(-k .* ((x .+ 1) .- n))))
        return rescale_minmax(s, lb, ub)
    end
end

end # module Activation
