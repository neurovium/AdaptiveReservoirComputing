# =============================================================================
# Metrics.jl
# -----------------------------------------------------------------------------
# Prediction-error metrics and the composite genetic-algorithm fitness score.
#
# Julia port of:
#   * `compute_error.m`  -> compute_error  (NRMSE per channel, scalar NMAE)
#   * the J = j1/j2 logic in `KS64D_runOptimizePredESN.m` -> composite_fitness
#
# The min-max normalization used for NMAE is reproduced *exactly* as written in
# `compute_error.m` (it normalizes both signals using the min/max of the
# predicted output). This precise convention defines the fitness `J` and so
# must not be "cleaned up".
# =============================================================================

module Metrics

using Statistics

export PredictionError, compute_error, composite_fitness

"""
    PredictionError

Result of `compute_error`, mirroring the MATLAB `err` struct so the saved
`.mat` files expose the same field names.

# Fields
- `NRMSE::Vector{Float64}` : per-channel normalized RMSE (length `num_inputs`),
  normalized by the standard deviation (√variance) of each true channel.
- `NMAE::Float64`          : scalar normalized mean absolute error.
"""
struct PredictionError
    NRMSE::Vector{Float64}
    NMAE::Float64
end

"""
    compute_error(estimated, correct) -> PredictionError

Port of `compute_error.m`. Both `estimated` and `correct` are
`num_inputs × predict_length` (channels × time).

## NRMSE (per channel)
For each channel `k`:

    NRMSE_k = sqrt( mean_t (yhat_k(t) - y_k(t))^2 / var_t(y_k) )

where `var` is the sample variance (MATLAB `var`, i.e. normalized by `T-1`),
computed over time for each channel.

## NMAE (scalar)  — verbatim from `compute_error.m`
    normA = correct  - min(estimated)        % min over ALL entries of estimated
    normB = estimated - min(estimated)
    normA = normA ./ max(normA)              % max over ALL entries of normA
    normB = normB ./ max(normA)
    NMAE  = sum(|normB - normA|) / (num_inputs * predict_length)
"""
function compute_error(estimated::AbstractMatrix, correct::AbstractMatrix)
    nInputDim, nEstimatePoints = size(estimated)
    @assert size(correct) == size(estimated) "estimated and correct must match in size"

    # --- NRMSE per channel ---------------------------------------------------
    # MATLAB `var(correctOutput)` operates column-wise; here channels are rows,
    # so we take the variance along time (dims=2) for each channel.
    correctVariance = vec(var(correct; dims=2))               # length num_inputs
    meanerror = vec(sum((estimated .- correct) .^ 2; dims=2)) ./ nEstimatePoints
    NRMSE = sqrt.(meanerror ./ correctVariance)

    # --- NMAE (exact min-max convention from compute_error.m) ----------------
    minEst = minimum(estimated)
    normA = correct .- minEst
    normB = estimated .- minEst
    maxNormA = maximum(normA)
    normA = normA ./ maxNormA
    normB = normB ./ maxNormA
    NMAE = sum(abs.(normB .- normA)) / (nInputDim * nEstimatePoints)

    return PredictionError(NRMSE, NMAE)
end

"""
    composite_fitness(err::PredictionError; threshold=0.05) -> J

Port of the GA fitness from `KS64D_runOptimizePredESN.m`:

    j1 = err.NMAE
    j2 = sum(err.NRMSE < threshold)          % # channels below ε
    J  = j1 / j2

Lower `J` is better. If no channel satisfies `NRMSE_k < threshold` (`j2 = 0`),
the score is `+Inf`, marking the worst possible fitness under minimization —
matching the MATLAB convention (and the paper's prose) that such an individual
"is not capturing the underlying dynamics across the spatial field". Note this is
the explicit zero-denominator rule: `J = j1/j2` is never evaluated when `j2 = 0`
(MATLAB's `positive/0` would also give `Inf`, but we guard it explicitly).

Robustness extension (beyond the paper's explicit `j2 = 0` case): if the
autonomous rollout diverged so that `NMAE` is `NaN`/`Inf` (e.g. one clean channel
slips under threshold while others blow up), the ratio is also non-finite; we map
it to `+Inf` as well, since a divergent reservoir is likewise "worst fitness".
The result is therefore always finite-or-`Inf`, never `NaN`.
"""
function composite_fitness(err::PredictionError; threshold::Real=0.05)
    j1 = err.NMAE
    j2 = count(<(threshold), err.NRMSE)
    j2 == 0 && return Inf                  # explicit zero-denominator rule (paper)
    J = j1 / j2
    return isfinite(J) ? J : Inf           # diverged NMAE → worst fitness
end

end # module Metrics
