# =============================================================================
# Sampling.jl
# -----------------------------------------------------------------------------
# Generation selection and stratified sampling for the structural analyses.
#
# Ports `find_suitable_last_gen` + the even-spacing generation selection and
# `select_stratified_samples` that appear (identically) in both
# `compute_laplacianSpectrum_stratifiedSelection.jl` and
# `calculate_modularity_stratifiedSelection.jl`.
#
# Paper §"Network selection for spectral analysis":
#   "We selected [N] generations evenly spaced across evolutionary time,
#    including the initial population and the final generation containing at
#    least 299 individuals ... reservoirs were ranked according to prediction
#    error J ... divided into quartiles ... up to 20 reservoirs sampled from
#    each quartile ... up to 80 reservoirs per generation."
#
# Sampling that the original seeded with `Random.seed!(1234)` is parameterized
# here by an explicit `StableRNG`, so a given seed reproduces the same sample
# across Julia versions.
# =============================================================================

module Sampling

using Random: AbstractRNG
using StableRNGs
using StatsBase: sample
using ..DataAccess: list_individual_files, files_for_generation, parse_generation,
                    read_J_N

export find_suitable_last_gen, select_generations, stratified_sample,
       stratified_sample_run

# -----------------------------------------------------------------------------
# Generation selection
# -----------------------------------------------------------------------------

"""
    find_suitable_last_gen(generations, matdir; min_individuals=299) -> Int

The latest generation that holds at least `min_individuals` artifacts, scanning
from the newest backwards. Falls back to the very last generation if none meets
the threshold. Port of the original `find_suitable_last_gen`.

`generations` is a sorted vector of available generation indices (see
`DataAccess.available_generations`).
"""
function find_suitable_last_gen(generations::AbstractVector{<:Integer},
                                matdir::AbstractString; min_individuals::Integer=299)
    for g in Iterators.reverse(generations)
        n = length(files_for_generation(matdir, g))
        n >= min_individuals && return Int(g)
    end
    return Int(generations[end])
end

"""
    select_generations(generations, matdir; n_select=10, min_individuals=299) -> Vector{Int}

Pick `n_select` generations evenly spaced across evolutionary time: the first
generation, the suitable last generation (≥ `min_individuals` individuals), and
`n_select-2` evenly spaced generations between them. Duplicates (which arise for
short runs) are removed, so the result may contain fewer than `n_select` values.

This reproduces the original spacing exactly:
    step = (last - first) / (n_select - 1)
    gens = [first, round.(first .+ step .* (1:n_select-2))..., last]
The spectral script used `n_select=10`; the modularity script used `ngen=20`.
"""
function select_generations(generations::AbstractVector{<:Integer},
                            matdir::AbstractString;
                            n_select::Integer=10, min_individuals::Integer=299)
    isempty(generations) && throw(ArgumentError("no generations available"))
    first_gen = Int(generations[1])
    last_gen = find_suitable_last_gen(generations, matdir; min_individuals=min_individuals)

    last_gen == first_gen && return collect(Int.(generations))

    step = (last_gen - first_gen) / (n_select - 1)
    middle = round.(Int, first_gen .+ step .* (1:(n_select - 2)))
    sel = unique(vcat(first_gen, middle, last_gen))
    return sort!(sel)
end

# -----------------------------------------------------------------------------
# Stratified sampling within a generation
# -----------------------------------------------------------------------------

"""
    stratified_sample(gen_files, J; n_per_quartile=20, rng=StableRNG(1234)) -> Vector{String}

Stratified sample of one generation's artifacts. `gen_files` are that
generation's filenames and `J[i]` is the composite error of `gen_files[i]`.

The population is ranked by `J` (best→worst), split into 4 equal-count
quartiles, and up to `n_per_quartile` files are drawn without replacement from
each quartile — yielding up to `4·n_per_quartile` (default 80) files. Port of
`select_stratified_samples`.
"""
function stratified_sample(gen_files::AbstractVector{<:AbstractString},
                           J::AbstractVector{<:Real};
                           n_per_quartile::Integer=20,
                           rng::AbstractRNG=StableRNG(1234))
    @assert length(gen_files) == length(J) "files and J must align"
    order = sortperm(J)                       # best (low J) first
    sorted = collect(gen_files)[order]
    n_total = length(sorted)
    out = String[]
    for q in 1:4
        start_idx = floor(Int, (q - 1) * n_total / 4) + 1
        end_idx = floor(Int, q * n_total / 4)
        start_idx > end_idx && continue
        quartile = sorted[start_idx:end_idx]
        k = min(n_per_quartile, length(quartile))
        append!(out, sample(rng, quartile, k; replace=false))
    end
    return out
end

"""
    stratified_sample_run(matdir, selected_generations; n_per_quartile=20,
                          rng=StableRNG(1234)) -> Vector{String}

Apply `stratified_sample` to every generation in `selected_generations`,
reading each generation's `J` values from disk, and concatenate. Returns the
combined list of sampled filenames in generation order — the `all_samples`
vector of the original scripts.
"""
function stratified_sample_run(matdir::AbstractString,
                               selected_generations::AbstractVector{<:Integer};
                               n_per_quartile::Integer=20,
                               rng::AbstractRNG=StableRNG(1234))
    all_samples = String[]
    for gen in selected_generations
        gen_files = files_for_generation(matdir, gen)
        isempty(gen_files) && continue
        J = [read_J_N(matdir, f)[1] for f in gen_files]
        append!(all_samples,
                stratified_sample(gen_files, J; n_per_quartile=n_per_quartile, rng=rng))
    end
    return all_samples
end

end # module Sampling
