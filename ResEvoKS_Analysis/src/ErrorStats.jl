# =============================================================================
# ErrorStats.jl
# -----------------------------------------------------------------------------
# Per-generation distributions of the composite error J across an evolutionary
# run — the population-level error-reduction view.
#
# Ports the data-collection cores of `evoRunGenErrors_64D-ks.jl` (per-generation
# log10 J histograms) and `sampleGenDistError_64D-ks.jl` (per-generation error
# distributions). Only the *computation* is ported; the plotting lives in
# scripts/run_error_distributions.jl.
#
# Paper §"Population-level reduction of composite prediction error".
# =============================================================================

module ErrorStats

using Statistics
using ..DataAccess: files_for_generation, read_J_N

export collect_generation_errors, log_error_distribution,
       GenerationErrorStats, generation_error_stats

"""
    collect_generation_errors(matdir, gens) -> Dict{Int,Vector{Float64}}

For each generation in `gens`, the vector of composite errors `J` of every
individual in that generation (read from disk). Missing generations map to an
empty vector. Port of the per-generation `allJ` collection loop.
"""
function collect_generation_errors(matdir::AbstractString,
                                   gens::AbstractVector{<:Integer})
    out = Dict{Int,Vector{Float64}}()
    for gen in gens
        files = files_for_generation(matdir, gen)
        out[Int(gen)] = [read_J_N(matdir, f)[1] for f in files]
    end
    return out
end

"""
    log_error_distribution(J; drop_nonfinite=true) -> Vector{Float64}

`log10` of the errors `J`, with non-finite entries (`Inf` fitness — reservoirs
where no channel beat the NRMSE threshold) dropped by default. This is the
quantity the original histograms plot (`sort(log10.(allJ))`).
"""
function log_error_distribution(J::AbstractVector{<:Real}; drop_nonfinite::Bool=true)
    vals = Float64.(J)
    drop_nonfinite && (vals = filter(isfinite, vals))
    return log10.(vals)
end

"""
    GenerationErrorStats

Summary statistics of one generation's error distribution (finite `J` only).

# Fields
- `generation::Int`
- `n::Int`            : number of finite-`J` individuals.
- `n_inf::Int`        : number of `Inf`/non-finite individuals (failed reservoirs).
- `mean, median, std, min, max :: Float64` : statistics of `log10 J`.
"""
struct GenerationErrorStats
    generation::Int
    n::Int
    n_inf::Int
    mean::Float64
    median::Float64
    std::Float64
    min::Float64
    max::Float64
end

"""
    generation_error_stats(matdir, gens) -> Vector{GenerationErrorStats}

Per-generation summary of the `log10 J` distribution, one entry per generation
in `gens` (in order). Generations with no finite individuals report `NaN`
statistics but still record `n_inf`.
"""
function generation_error_stats(matdir::AbstractString,
                                gens::AbstractVector{<:Integer})
    errs = collect_generation_errors(matdir, gens)
    out = GenerationErrorStats[]
    for gen in gens
        J = errs[Int(gen)]
        finite = filter(isfinite, J)
        n_inf = length(J) - length(finite)
        if isempty(finite)
            push!(out, GenerationErrorStats(Int(gen), 0, n_inf,
                                            NaN, NaN, NaN, NaN, NaN))
        else
            lg = log10.(finite)
            push!(out, GenerationErrorStats(Int(gen), length(finite), n_inf,
                                            mean(lg), median(lg),
                                            length(lg) > 1 ? std(lg) : 0.0,
                                            minimum(lg), maximum(lg)))
        end
    end
    return out
end

end # module ErrorStats
