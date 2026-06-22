# =============================================================================
# Evaluation.jl
# -----------------------------------------------------------------------------
# Single-individual evaluation: decode a GA genome into reservoir parameters,
# build and train the reservoir, run autonomous prediction, score it, and
# (optionally) persist the artifact.
#
# Julia port of the nested `quickOptimizePredESN` objective inside
# `KS64D_runOptimizePredESN.m`.
# =============================================================================

module Evaluation

using Random
using StableRNGs

using ..Reservoir: ReservoirParams, build_reservoir, generate_input_weights
using ..Readout:   reservoir_layer, train_readout, predict
using ..Metrics:   compute_error, composite_fitness, PredictionError
using ..Activation: tanh_activation
using ..IO:        save_individual

export DataParams, decode_genome, EvalResult, evaluate_individual

"""
    DataParams

Training/prediction window lengths. Mirrors the MATLAB `dataparams` struct.

# Fields
- `train_length::Int`   : number of teacher-forced training samples (e.g. 70000).
- `predict_length::Int` : number of autonomous prediction samples (e.g. 2000).
"""
Base.@kwdef struct DataParams
    train_length::Int   = 70_000
    predict_length::Int = 2_000
end

"""
    decode_genome(genome, num_inputs) -> ReservoirParams

Decode the 5-gene GA genome into a `ReservoirParams`, reproducing the unpacking
in `quickOptimizePredESN`:

    radius        = genome[1]
    degree        = round(genome[2])                 (integer gene)
    approx_size   = round(genome[3])                 (integer gene)
    N             = floor(approx_size / num_inputs) * num_inputs
    sigma         = genome[4]
    beta          = genome[5]

The integer genes (degree, size) are rounded here; in the MATLAB driver they
are enforced as integers by `IntCon = [2,3]`. The size is then snapped down to
a multiple of `num_inputs` so each input channel drives an equal block.
"""
function decode_genome(genome::AbstractVector{<:Real}, num_inputs::Integer)
    radius      = float(genome[1])
    degree      = Int(round(genome[2]))
    approx_size = Int(round(genome[3]))
    N           = (approx_size ÷ num_inputs) * num_inputs
    sigma       = float(genome[4])
    beta        = float(genome[5])
    return ReservoirParams(num_inputs=num_inputs, radius=radius, degree=degree,
                           N=N, sigma=sigma, beta=beta)
end

"""
    EvalResult

Everything produced by evaluating one individual, so a caller (the GA driver or
a post-hoc save step) can persist or inspect it without recomputation.

# Fields
- `J::Float64`               : composite fitness (objective; lower is better).
- `resparams::ReservoirParams`
- `err::PredictionError`
- `A`, `w_in`, `w_out`       : the trained reservoir matrices.
"""
struct EvalResult
    J::Float64
    resparams::ReservoirParams
    err::PredictionError
    A::Any
    w_in::Matrix{Float64}
    w_out::Matrix{Float64}
end

"""
    evaluate_individual(genome, data, dataparams; kwargs...) -> EvalResult

Full objective evaluation for one genome. Port of `quickOptimizePredESN`.

# Arguments
- `genome`    : length-5 vector `[radius, degree, size, sigma, beta]`.
- `data`      : `num_inputs × T` KS field (`measurements`).
- `dataparams`: `DataParams`.

# Keyword arguments
- `threshold=0.05` : NRMSE threshold `ε` for the fitness denominator.
- `activation=tanh_activation` : reservoir nonlinearity (paper uses `tanh`).
- `rng=StableRNG(...)` : RNG for reservoir + input-weight construction. Provide
  a per-individual seed for reproducibility (paper fixes seeds where possible).
- `return_artifacts=true` : if `false`, the heavy matrices in the returned
  `EvalResult` are empty (useful when only `J` is needed and memory matters).

Returns an `EvalResult`. Construction failures (e.g. eigensolver never
converging) propagate as exceptions; the GA driver catches them and assigns the
worst fitness.
"""
function evaluate_individual(genome::AbstractVector{<:Real}, data::AbstractMatrix,
                             dataparams::DataParams;
                             threshold::Real=0.05,
                             activation=tanh_activation,
                             rng::AbstractRNG=StableRNG(0),
                             return_artifacts::Bool=true)
    num_inputs = size(data, 1)
    resparams = decode_genome(genome, num_inputs)

    # --- build reservoir (sparse, then spectral-radius scaled, retry on fail)
    A = build_reservoir(resparams.N, resparams.degree, resparams.radius; rng=rng)

    # --- input weights (disjoint σ·uniform[-1,1] blocks) ---------------------
    w_in = generate_input_weights(resparams.N, num_inputs, resparams.sigma; rng=rng)

    # --- collect teacher-forced states --------------------------------------
    states = reservoir_layer(data, A, w_in, resparams, dataparams.train_length,
                             activation)

    # --- train ridge readout (targets = KS field over the train window) ------
    target = @view data[:, 1:dataparams.train_length]
    w_out = train_readout(resparams, states, target)

    # --- autonomous prediction from the last training state -----------------
    x_state = states[:, end]
    outdata, _ = predict(x_state, A, w_in, resparams, w_out,
                         dataparams.predict_length, activation)

    # --- error over the held-out prediction window --------------------------
    pred_window = (dataparams.train_length + 1):(dataparams.train_length + dataparams.predict_length)
    indata = @view data[:, pred_window]
    err = compute_error(outdata, Matrix(indata))

    # --- composite fitness J -------------------------------------------------
    J = composite_fitness(err; threshold=threshold)

    if return_artifacts
        return EvalResult(J, resparams, err, A, w_in, w_out)
    else
        empty = Matrix{Float64}(undef, 0, 0)
        return EvalResult(J, resparams, err, empty, empty, empty)
    end
end

end # module Evaluation
