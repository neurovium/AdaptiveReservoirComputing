# =============================================================================
# DataAccess.jl
# -----------------------------------------------------------------------------
# Read the per-individual artifacts written by the ResEvoKS_Simulation simulation.
#
# Ports the file-listing / filename-parsing / `get_J_N` / `load_adjacency_matrix`
# logic that is repeated across every original analysis script
# (`compute_laplacianSpectrum_*.jl`, `calculate_modularity_*.jl`,
# `find_pareto_netsizeError_*.jl`, ...).
#
# The on-disk contract (see REPORT_analysis_code.md) is one file per evaluated
# reservoir, named
#
#     <gen>_<N>_<degree>_<radius>_<sigma>.{mat,jld2}
#
# inside  <results>/RUN<k>/matfiles/  , holding the variables
#     A, w_in, w_out, resparams, err, J.
#
# Both MATLAB `.mat` and native Julia `.jld2` are supported transparently, so
# the analysis does not care which format the simulation chose to write.
# =============================================================================

module DataAccess

using MAT
using JLD2
using SparseArrays

export IndividualRecord,
       run_matdir, list_individual_files, parse_generation,
       available_generations, files_for_generation,
       load_record, load_adjacency, read_J_N,
       collect_J_N

# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------

"""
    IndividualRecord

A single evaluated reservoir loaded from disk. Field names mirror the on-disk
contract so downstream analyses read like the saved data.

# Fields
- `filename::String`  : the artifact's base filename (e.g. `"10_512_6_0.9_0.5.mat"`).
- `generation::Int`   : generation index parsed from the filename prefix.
- `A`                 : recurrent matrix after spectral scaling (sparse or dense).
- `J::Float64`        : composite fitness `NMAE / #(NRMSE < ε)`.
- `N::Int`            : reservoir size (`resparams.N`).
- `NRMSE::Vector{Float64}` : per-channel normalized RMSE (may be empty if absent).
- `NMAE::Float64`     : scalar normalized MAE (`NaN` if absent).
- `resparams::Dict{String,Any}` : the raw hyperparameter dict.
"""
struct IndividualRecord
    filename::String
    generation::Int
    A::Any
    J::Float64
    N::Int
    NRMSE::Vector{Float64}
    NMAE::Float64
    resparams::Dict{String,Any}
end

# -----------------------------------------------------------------------------
# Directory / filename helpers
# -----------------------------------------------------------------------------

"""
    run_matdir(rootdir, run_index) -> String

Path to the `matfiles` directory of run `run_index` under `rootdir`, i.e.
`<rootdir>/RUN<run_index>/matfiles`. This is the layout the simulation writes
(`make_run_dirs`).
"""
run_matdir(rootdir::AbstractString, run_index::Integer) =
    joinpath(rootdir, string("RUN", run_index), "matfiles")

"""
    list_individual_files(matdir; ext=("mat","jld2")) -> Vector{String}

All per-individual artifact filenames in `matdir`, excluding the dataset files
`data.mat` / `data.jld2` and the legacy `history.mat`. Returns base filenames
(not full paths). Both `.mat` and `.jld2` are included by default; pass `ext`
to restrict (e.g. `ext=("mat",)` to match the MATLAB-only analyses).
"""
function list_individual_files(matdir::AbstractString; ext=("mat", "jld2"))
    isdir(matdir) || throw(ArgumentError("not a directory: $matdir"))
    excluded = Set(["data.mat", "data.jld2", "history.mat", "history.jld2"])
    keep(f) = !(f in excluded) &&
              any(e -> endswith(f, string('.', e)), ext) &&
              # an individual filename starts with "<gen>_"
              occursin(r"^\d+_", f)
    return sort!(filter(keep, readdir(matdir)))
end

"""
    parse_generation(filename) -> Int

The generation index encoded as the filename prefix `"<gen>_..."`.
"""
parse_generation(filename::AbstractString) =
    parse(Int, split(basename(filename), "_")[1])

"""
    available_generations(matdir; ext=("mat","jld2")) -> Vector{Int}

Sorted unique generation indices present in `matdir`.
"""
function available_generations(matdir::AbstractString; ext=("mat", "jld2"))
    files = list_individual_files(matdir; ext=ext)
    return sort!(unique(parse_generation.(files)))
end

"""
    files_for_generation(matdir, gen; ext=("mat","jld2")) -> Vector{String}

All artifact filenames in `matdir` belonging to generation `gen`. Matching is on
the exact `"<gen>_"` prefix (so generation `1` does not also match `10_...`).
"""
function files_for_generation(matdir::AbstractString, gen::Integer; ext=("mat", "jld2"))
    prefix = string(gen, '_')
    return filter(f -> startswith(f, prefix),
                  list_individual_files(matdir; ext=ext))
end

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

# Read a saved artifact into a plain `Dict{String,Any}` regardless of format.
function _read_dict(path::AbstractString)
    if endswith(path, ".mat")
        return matread(path)
    elseif endswith(path, ".jld2")
        d = Dict{String,Any}()
        jldopen(path, "r") do f
            for k in keys(f)
                d[k] = f[k]
            end
        end
        return d
    else
        throw(ArgumentError("unsupported extension: $path"))
    end
end

# `resparams` may be a Dict (from .mat) or a struct (from .jld2). Normalize to a
# Dict{String,Any} so callers have one access pattern.
function _resparams_dict(rp)::Dict{String,Any}
    rp isa AbstractDict && return Dict{String,Any}(string(k) => v for (k, v) in rp)
    d = Dict{String,Any}()
    for name in propertynames(rp)
        d[string(name)] = getproperty(rp, name)
    end
    return d
end

# `err` may be a Dict (.mat) or a PredictionError-like struct (.jld2).
function _err_fields(err)
    if err isa AbstractDict
        nrmse = haskey(err, "NRMSE") ? err["NRMSE"] : Float64[]
        nmae = haskey(err, "NMAE") ? err["NMAE"] : NaN
    else
        nrmse = hasproperty(err, :NRMSE) ? getproperty(err, :NRMSE) : Float64[]
        nmae = hasproperty(err, :NMAE) ? getproperty(err, :NMAE) : NaN
    end
    return Float64.(vec(collect(nrmse))), Float64(nmae)
end

"""
    load_record(matdir, filename) -> IndividualRecord

Load one individual artifact fully (recurrent matrix, fitness, size, error
breakdown, hyperparameters). Works for both `.mat` and `.jld2`.
"""
function load_record(matdir::AbstractString, filename::AbstractString)
    path = joinpath(matdir, filename)
    d = _read_dict(path)

    rp = _resparams_dict(d["resparams"])
    N = Int(round(rp["N"]))

    nrmse, nmae = haskey(d, "err") ? _err_fields(d["err"]) : (Float64[], NaN)

    return IndividualRecord(
        filename,
        parse_generation(filename),
        d["A"],
        Float64(d["J"]),
        N,
        nrmse,
        nmae,
        rp,
    )
end

"""
    load_adjacency(matdir, filename) -> AbstractMatrix

Load just the recurrent / adjacency matrix `A` (the network used for all
structural analyses). Cheaper than `load_record` when only `A` is needed.
"""
function load_adjacency(matdir::AbstractString, filename::AbstractString)
    return _read_dict(joinpath(matdir, filename))["A"]
end

"""
    read_J_N(matdir, filename) -> (J::Float64, N::Int)

Port of the original `get_J_N`: read only the composite error `J` and the
reservoir size `N` from an artifact (fast; avoids loading `A`).
"""
function read_J_N(matdir::AbstractString, filename::AbstractString)
    d = _read_dict(joinpath(matdir, filename))
    N = Int(round(_resparams_dict(d["resparams"])["N"]))
    return Float64(d["J"]), N
end

"""
    collect_J_N(matdir, filenames) -> (J::Vector{Float64}, N::Vector{Int})

Read `(J, N)` for many artifacts, returned as parallel vectors in the order of
`filenames`. Convenience for the Pareto / error-distribution analyses.
"""
function collect_J_N(matdir::AbstractString, filenames::AbstractVector{<:AbstractString})
    J = Vector{Float64}(undef, length(filenames))
    N = Vector{Int}(undef, length(filenames))
    for (i, f) in enumerate(filenames)
        J[i], N[i] = read_J_N(matdir, f)
    end
    return J, N
end

end # module DataAccess
