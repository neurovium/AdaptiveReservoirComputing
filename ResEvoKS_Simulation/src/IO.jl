# =============================================================================
# IO.jl
# -----------------------------------------------------------------------------
# Reading/writing of simulation artifacts.
#
# Two on-disk formats are supported, selectable per call via `format`:
#
#   :mat   — MATLAB v5 `.mat` (default). *Bit-compatible* with the original
#            pipeline so the existing analysis scripts keep working unchanged.
#   :jld2  — native Julia `.jld2` (HDF5-based). Stores the actual Julia objects
#            (sparse `A`, the `ReservoirParams`/`PredictionError` structs), so a
#            user can keep the entire workflow inside Julia with no MAT round-trip
#            and no precision/struct-flattening loss.
#   :both  — write both `.mat` and `.jld2`.
#
# Per-individual files are named
#
#     <gen>_<N>_<degree>_<radius>_<sigma>.{mat,jld2}
#
# (e.g. `12_2048_6_0.873000_0.412000.mat`) and, in both formats, expose the
# variables `w_in, w_out, A, resparams, err, J` — exactly the names the analysis
# code (`find_pareto_netsizeError_64D-KS.jl`, `evoRunGenErrors_64D-ks.jl`, …)
# reads from the `.mat` files.
# =============================================================================

module IO

using MAT
using JLD2
using Printf
using SparseArrays

using ..Reservoir: ReservoirParams
using ..Metrics: PredictionError

export individual_filename, save_individual, resparams_to_dict, err_to_dict,
       SaveParams, make_run_dirs, save_dataset, load_individual, load_dataset,
       SAVE_FORMATS

"""Valid values for the `format` keyword of `save_individual`/`save_dataset`."""
const SAVE_FORMATS = (:mat, :jld2, :both)

# Validate a requested format and return it (throws an informative error).
function _check_format(format::Symbol)
    format in SAVE_FORMATS ||
        throw(ArgumentError("format must be one of $(SAVE_FORMATS), got :$format"))
    return format
end

"""
    SaveParams

Per-run output directory layout, mirroring the MATLAB `saveparams` struct.

# Fields
- `codedir::String` : logs / diary (MATLAB `saveparams.codedir`, "Log/").
- `matdir::String`  : `data.{mat,jld2}` + per-individual files ("matfiles/").
- `figdir::String`  : figures ("Figs/").

Note: `matdir` holds the per-individual files regardless of format (the name is
kept for continuity with the MATLAB pipeline).
"""
Base.@kwdef struct SaveParams
    codedir::String
    matdir::String
    figdir::String
end

"""
    make_run_dirs(rootdir, run_index) -> SaveParams

Create `<rootdir>/RUN<k>/{Log,matfiles,Figs}` and return the `SaveParams`.
Reproduces the directory scheme of `KS64D_prepDataAndRun.m`.
"""
function make_run_dirs(rootdir::AbstractString, run_index::Integer)
    base = joinpath(rootdir, "RUN$(run_index)")
    sp = SaveParams(
        codedir = joinpath(base, "Log"),
        matdir  = joinpath(base, "matfiles"),
        figdir  = joinpath(base, "Figs"),
    )
    for d in (sp.codedir, sp.matdir, sp.figdir)
        isdir(d) || mkpath(d)
    end
    return sp
end

"""
    individual_filename(gen, resparams) -> String

Reproduce MATLAB `sprintf('%d_%d_%d_%f_%f', gen, N, degree, radius, sigma)`
(the base name, without extension). `%f` prints six decimal places, e.g. radius
`0.873` → `"0.873000"`.
"""
function individual_filename(gen::Integer, resparams::ReservoirParams)
    return @sprintf("%d_%d_%d_%f_%f", gen, resparams.N, resparams.degree,
                    resparams.radius, resparams.sigma)
end

"""
    resparams_to_dict(p::ReservoirParams) -> Dict

Convert to a `Dict` whose keys match the MATLAB `resparams` struct fields, so
`matread(...)["resparams"]["N"]` etc. work identically. Numeric values are
stored as `Float64` (MATLAB doubles), matching how the original `.mat` files
stored them.
"""
function resparams_to_dict(p::ReservoirParams)
    return Dict{String,Any}(
        "num_inputs" => Float64(p.num_inputs),
        "radius"     => Float64(p.radius),
        "degree"     => Float64(p.degree),
        "N"          => Float64(p.N),
        "sigma"      => Float64(p.sigma),
        "beta"       => Float64(p.beta),
    )
end

# Inverse of resparams_to_dict (used when loading `.mat` files).
function dict_to_resparams(d::AbstractDict)
    return ReservoirParams(
        num_inputs = Int(round(d["num_inputs"])),
        radius     = Float64(d["radius"]),
        degree     = Int(round(d["degree"])),
        N          = Int(round(d["N"])),
        sigma      = Float64(d["sigma"]),
        beta       = Float64(d["beta"]),
    )
end

"""
    err_to_dict(e::PredictionError) -> Dict

Convert to a `Dict` matching the MATLAB `err` struct: `NRMSE` (row-like vector)
and `NMAE` (scalar). NRMSE is stored as a `1 × num_inputs` matrix to mirror the
MATLAB row vector that the analysis code reads.
"""
function err_to_dict(e::PredictionError)
    return Dict{String,Any}(
        "NRMSE" => reshape(collect(e.NRMSE), 1, :),
        "NMAE"  => e.NMAE,
    )
end

# Inverse of err_to_dict (used when loading `.mat` files).
function dict_to_err(d::AbstractDict)
    return PredictionError(vec(Float64.(d["NRMSE"])), Float64(d["NMAE"]))
end

# --- per-format writers ------------------------------------------------------

# Write the MATLAB-compatible `.mat` (Dicts, sparse A, Float64 scalars).
function _write_mat(path, A, w_in, w_out, resparams::ReservoirParams,
                    err::PredictionError, J::Real)
    matwrite(path, Dict{String,Any}(
        "w_in"      => Matrix(w_in),
        "w_out"     => Matrix(w_out),
        "A"         => issparse(A) ? A : sparse(A),
        "resparams" => resparams_to_dict(resparams),
        "err"       => err_to_dict(err),
        "J"         => Float64(J),
    ))
    return path
end

# Write native Julia objects to `.jld2` (keeps structs + sparsity exactly).
function _write_jld2(path, A, w_in, w_out, resparams::ReservoirParams,
                     err::PredictionError, J::Real)
    jldsave(path;
        w_in      = Matrix(w_in),
        w_out     = Matrix(w_out),
        A         = issparse(A) ? A : sparse(A),
        resparams = resparams,
        err       = err,
        J         = Float64(J),
    )
    return path
end

"""
    save_individual(matdir, gen, A, w_in, w_out, resparams, err, J; format=:mat)
        -> String

Write one individual's artifact(s) with the variable set
`{w_in, w_out, A, resparams, err, J}`.

`format` is one of `:mat` (default), `:jld2`, or `:both`:
- `:mat`  writes `<base>.mat`  (MATLAB-compatible Dicts; `A` stays sparse).
- `:jld2` writes `<base>.jld2` (native Julia structs; full fidelity).
- `:both` writes both.

Returns the path to the written file (for `:both`, the `.mat` path). Use
`load_individual` to read either format back uniformly.
"""
function save_individual(matdir::AbstractString, gen::Integer,
                         A::AbstractMatrix, w_in::AbstractMatrix,
                         w_out::AbstractMatrix, resparams::ReservoirParams,
                         err::PredictionError, J::Real; format::Symbol=:mat)
    _check_format(format)
    base = joinpath(matdir, individual_filename(gen, resparams))
    matpath = base * ".mat"
    jldpath = base * ".jld2"

    if format === :mat || format === :both
        _write_mat(matpath, A, w_in, w_out, resparams, err, J)
    end
    if format === :jld2 || format === :both
        _write_jld2(jldpath, A, w_in, w_out, resparams, err, J)
    end

    return format === :jld2 ? jldpath : matpath
end

"""
    load_individual(path) -> NamedTuple

Read a saved individual back from either a `.mat` or `.jld2` file, dispatching on
the extension. Returns `(; w_in, w_out, A, resparams, err, J)` with `resparams`
as a `ReservoirParams` and `err` as a `PredictionError` in both cases.
"""
function load_individual(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext == ".mat"
        d = matread(path)
        return (w_in = d["w_in"], w_out = d["w_out"], A = d["A"],
                resparams = dict_to_resparams(d["resparams"]),
                err = dict_to_err(d["err"]), J = Float64(d["J"]))
    elseif ext == ".jld2"
        d = JLD2.load(path)
        return (w_in = d["w_in"], w_out = d["w_out"], A = d["A"],
                resparams = d["resparams"], err = d["err"], J = Float64(d["J"]))
    else
        throw(ArgumentError("unknown extension '$ext' (expected .mat or .jld2)"))
    end
end

"""
    save_dataset(matdir, data, model_params; format=:mat) -> String

Write the KS field `data` and the `ModelParams` as `data.{mat,jld2}` per the
selected `format`, matching `save([matdir,'data.mat'],'ModelParams','data')` in
`KS64D_prepDataAndRun.m`. Returns the path written (for `:both`, the `.mat`).
"""
function save_dataset(matdir::AbstractString, data::AbstractMatrix, model_params;
                      format::Symbol=:mat)
    _check_format(format)
    matpath = joinpath(matdir, "data.mat")
    jldpath = joinpath(matdir, "data.jld2")
    mp = (N = model_params.N, d = model_params.d,
          tau = model_params.tau, nstep = model_params.nstep)

    if format === :mat || format === :both
        matwrite(matpath, Dict{String,Any}(
            "data" => Matrix(data),
            "ModelParams" => Dict{String,Any}(
                "N" => Float64(mp.N), "d" => Float64(mp.d),
                "tau" => Float64(mp.tau), "nstep" => Float64(mp.nstep),
            ),
        ))
    end
    if format === :jld2 || format === :both
        jldsave(jldpath; data = Matrix(data), ModelParams = mp)
    end

    return format === :jld2 ? jldpath : matpath
end

"""
    load_dataset(path) -> NamedTuple

Read a `data.mat`/`data.jld2` back as `(; data, ModelParams)`. For `.mat`,
`ModelParams` is a `Dict`; for `.jld2`, a `NamedTuple`.
"""
function load_dataset(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext == ".mat"
        d = matread(path)
        return (data = d["data"], ModelParams = d["ModelParams"])
    elseif ext == ".jld2"
        d = JLD2.load(path)
        return (data = d["data"], ModelParams = d["ModelParams"])
    else
        throw(ArgumentError("unknown extension '$ext' (expected .mat or .jld2)"))
    end
end

end # module IO
