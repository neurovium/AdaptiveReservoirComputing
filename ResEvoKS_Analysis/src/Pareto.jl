# =============================================================================
# Pareto.jl
# -----------------------------------------------------------------------------
# Reservoir size-efficiency trade-off: the empirical Pareto frontier in the
# (size, log-error) plane and its exponential summary fit.
#
# Ports `find_pareto_frontier` and the `LsqFit` exponential model from
# `find_pareto_netsizeError_64D-KS.jl`.
#
# Paper §"Reservoir size-efficiency trade-off":
#   Each reservoir is a point  x_i = n_{r,i},  y_i = log J_i.
#   A point (x_i,y_i) is Pareto efficient if no other (x_j,y_j) has
#       x_j <= x_i  AND  y_j <= y_i   with at least one strict inequality.
#   The frontier is summarized by  f(x) = a e^{-b x} + c  (nonlinear LSQ).
# =============================================================================

module Pareto

using LsqFit: curve_fit, coef

export pareto_frontier, fit_exponential, ParetoFit, exponential_model

# -----------------------------------------------------------------------------
# Pareto frontier
# -----------------------------------------------------------------------------

"""
    pareto_frontier(points) -> (mask, frontier)

Two-objective (minimize both coordinates) Pareto frontier of a set of 2-D
points. `points` is an `M×2` matrix whose rows are `(x_i, y_i)`; in the
size-efficiency analysis `x = N` (reservoir size) and `y = log J`.

Returns
- `mask::Vector{Bool}` : `mask[i]` true iff row `i` of `points` is non-dominated.
- `frontier::Matrix`   : the non-dominated rows (a `K×2` matrix).

A point `(x_i,y_i)` is **dominated** when some other distinct point `(x_j,y_j)`
satisfies `x_j ≤ x_i` and `y_j ≤ y_i`. Non-dominated points form the frontier.
This matches the original `find_pareto_frontier` (which deduplicated points and
tested coordinate-wise `≥` against all others).
"""
function pareto_frontier(points::AbstractMatrix{<:Real})
    size(points, 2) == 2 || throw(ArgumentError("points must be M×2"))
    n = size(points, 1)
    nondominated = trues(n)
    @inbounds for i in 1:n
        xi, yi = points[i, 1], points[i, 2]
        for j in 1:n
            i == j && continue
            xj, yj = points[j, 1], points[j, 2]
            # j dominates i if j is <= in both and strictly < in at least one
            if xj <= xi && yj <= yi && (xj < xi || yj < yi)
                nondominated[i] = false
                break
            end
        end
    end
    frontier = points[nondominated, :]
    # sort the frontier by x for a clean, monotone curve
    frontier = frontier[sortperm(frontier[:, 1]), :]
    return nondominated, frontier
end

# -----------------------------------------------------------------------------
# Exponential summary fit
# -----------------------------------------------------------------------------

"""
    exponential_model(x, p) = p[1]*exp(-p[2]*x) + p[3]

The frontier-summary model `f(x) = a e^{-b x} + c` (`p = [a, b, c]`). Broadcasts
over `x`.
"""
exponential_model(x, p) = @. p[1] * exp(-p[2] * x) + p[3]

"""
    ParetoFit

Result of `fit_exponential`.

# Fields
- `a, b, c::Float64` : fitted parameters of `f(x)=a e^{-b x}+c`.
- `r_squared::Float64` : coefficient of determination on the fitted points.
- `x::Vector{Float64}`, `y::Vector{Float64}` : the frontier points used.
"""
struct ParetoFit
    a::Float64
    b::Float64
    c::Float64
    r_squared::Float64
    x::Vector{Float64}
    y::Vector{Float64}
end

"""
    fit_exponential(px, py; p0=nothing) -> ParetoFit

Fit `f(x) = a e^{-b x} + c` to the frontier points `(px, py)` by nonlinear
least squares. The default initial guess reproduces the original script:
`p0 = [max(py)-min(py), 0.001, min(py)]`. Also returns the `R²` goodness of fit.
"""
function fit_exponential(px::AbstractVector{<:Real}, py::AbstractVector{<:Real};
                         p0::Union{Nothing,AbstractVector{<:Real}}=nothing)
    x = Float64.(px)
    y = Float64.(py)
    guess = p0 === nothing ?
            [maximum(y) - minimum(y), 0.001, minimum(y)] : Float64.(p0)

    fit = curve_fit(exponential_model, x, y, guess)
    p = coef(fit)

    yhat = exponential_model(x, p)
    ss_res = sum(abs2, y .- yhat)
    ss_tot = sum(abs2, y .- (sum(y) / length(y)))
    r2 = ss_tot == 0 ? 1.0 : 1 - ss_res / ss_tot

    return ParetoFit(p[1], p[2], p[3], r2, x, y)
end

end # module Pareto
