# =============================================================================
# MultiObjective.jl
# -----------------------------------------------------------------------------
# Post-hoc NSGA-II multi-objective analysis of the (modularity, connection cost,
# performance, generation) trade-off.
#
# Ports `normalize`, `composite_objective_function`, the `NSGA2` optimization,
# and `find_closest_matches` from `calculate_modularity_stratifiedSelection.jl`.
#
# Paper §"Multi-objective analysis":
#   Metrics are min-max normalized to [0,1]:  x_norm = (x−min)/(max−min).
#   Four composite objectives:
#       O1 = norm_perf / (1 + norm_gen)             (improvement across gens)
#       O2 = norm_cost / (1 + norm_mod)             (structural efficiency)
#       O3 = norm_perf / (1 + norm_mod)             (performance–modularity)
#       O4 = norm_perf * (1 + norm_cost)            (performance–cost)
#   NSGA-II (Metaheuristics.jl): N=1000, p_cr=0.85, p_m=0.5, bounds [0,1]^4.
#   Optimized points are matched to the closest observed reservoir by Euclidean
#   distance.
#
#   IMPORTANT: this NSGA-II is independent of the GA that evolved the reservoirs;
#   it is applied post hoc to the normalized metrics to characterize trade-offs.
# =============================================================================

module MultiObjective

using Metaheuristics: optimize, positions, NSGA2
using LinearAlgebra: norm

export normalize_metric, normalize_metrics, NormalizedMetrics,
       composite_objectives, run_nsga2, closest_observed

# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

"""
    normalize_metric(x) -> Vector{Float64}

Min–max normalize a vector to `[0,1]`: `(x − min)/(max − min)`. A constant
vector maps to all-zeros (the original divides by the range; here we guard the
zero-range case). Paper Eq. for `x_norm`.
"""
function normalize_metric(x::AbstractVector{<:Real})
    lo, hi = extrema(x)
    rng = hi - lo
    rng == 0 && return zeros(Float64, length(x))
    return (Float64.(x) .- lo) ./ rng
end

"""
    NormalizedMetrics

The four normalized metric vectors (each in `[0,1]`, aligned by reservoir).

# Fields
- `modularity`, `connection_cost`, `performance`, `generation` :: `Vector{Float64}`
"""
struct NormalizedMetrics
    modularity::Vector{Float64}
    connection_cost::Vector{Float64}
    performance::Vector{Float64}
    generation::Vector{Float64}
end

"""
    normalize_metrics(; modularity, connection_cost, performance, generation) -> NormalizedMetrics

Min–max normalize each of the four raw metric vectors. `performance` is the
composite error `J`, `generation` the generation index. Paper §"Multi-objective
analysis" (normalized metrics: prediction error, generation, modularity,
connection cost).
"""
function normalize_metrics(; modularity::AbstractVector{<:Real},
                            connection_cost::AbstractVector{<:Real},
                            performance::AbstractVector{<:Real},
                            generation::AbstractVector{<:Real})
    return NormalizedMetrics(
        normalize_metric(modularity),
        normalize_metric(connection_cost),
        normalize_metric(performance),
        normalize_metric(generation),
    )
end

# -----------------------------------------------------------------------------
# Composite objectives
# -----------------------------------------------------------------------------

"""
    composite_objectives(x) -> NTuple{4,Float64}

The four composite objectives evaluated at a normalized point
`x = (norm_modularity, norm_cost, norm_performance, norm_generation)`:

    O1 = perf / (1 + gen)      # improvement across generations
    O2 = cost / (1 + mod)      # structural efficiency
    O3 = perf / (1 + mod)      # performance–modularity
    O4 = perf * (1 + cost)     # performance–cost

The additive `1+` terms keep `O1`–`O3` finite when a normalized metric is `0`
and keep cost contributing positively to `O4`. Argument order matches the
original `composite_objective_function`.
"""
function composite_objectives(x)
    norm_mod, norm_cost, norm_perf, norm_gen = x
    o1 = norm_perf / (1 + norm_gen)
    o2 = norm_cost / (1 + norm_mod)
    o3 = norm_perf / (1 + norm_mod)
    o4 = norm_perf * (1 + norm_cost)
    return (o1, o2, o3, o4)
end

# -----------------------------------------------------------------------------
# NSGA-II
# -----------------------------------------------------------------------------

"""
    run_nsga2(; N=1000, p_cr=0.85, p_m=0.5) -> optimized_positions::Matrix{Float64}

Run NSGA-II over the four composite objectives on the bounded domain `[0,1]⁴`
(`x = [mod, cost, perf, gen]`), reproducing the paper's configuration
(`N=1000, p_cr=0.85, p_m=0.5`). Returns the Pareto-optimal decision vectors as
rows of a matrix (the `positions(status)` of the original).

The objective wrapper returns `(fx, gx, hx)` with no constraints, as
Metaheuristics expects.
"""
function run_nsga2(; N::Integer=1000, p_cr::Real=0.85, p_m::Real=0.5)
    objective = x -> begin
        o1, o2, o3, o4 = composite_objectives(x)
        ([o1, o2, o3, o4], [0.0], [0.0])
    end
    bounds = [0.0 0.0 0.0 0.0; 1.0 1.0 1.0 1.0]    # 2×4: row 1 lower, row 2 upper
    algorithm = NSGA2(N=N, p_cr=p_cr, p_m=p_m)
    status = optimize(objective, bounds, algorithm)
    return Matrix{Float64}(positions(status))
end

# -----------------------------------------------------------------------------
# Match optimized points to observed reservoirs
# -----------------------------------------------------------------------------

"""
    closest_observed(optimized_positions, nm::NormalizedMetrics) -> Vector{Int}

For each optimized (Pareto) point, the index of the observed reservoir whose
normalized `(modularity, cost, performance, generation)` vector is closest in
Euclidean distance. Port of `find_closest_matches` (the optimized point order is
`[mod, cost, perf, gen]`, matching `composite_objectives`).
"""
function closest_observed(optimized_positions::AbstractMatrix{<:Real},
                          nm::NormalizedMetrics)
    observed = hcat(nm.modularity, nm.connection_cost, nm.performance, nm.generation)  # M×4
    M = size(observed, 1)
    matches = Vector{Int}(undef, size(optimized_positions, 1))
    for k in 1:size(optimized_positions, 1)
        opt = @view optimized_positions[k, :]
        best_i, best_d = 1, Inf
        for i in 1:M
            d = 0.0
            @inbounds for c in 1:4
                d += (opt[c] - observed[i, c])^2
            end
            if d < best_d
                best_d = d
                best_i = i
            end
        end
        matches[k] = best_i
    end
    return matches
end

end # module MultiObjective
