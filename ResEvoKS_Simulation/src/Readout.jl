# =============================================================================
# Readout.jl
# -----------------------------------------------------------------------------
# Reservoir state collection, ridge-regression readout training, and autonomous
# closed-loop prediction.
#
# Julia port of:
#   * `reservoir_layer.m` -> reservoir_layer
#   * `train.m`           -> train_readout
#   * `predict.m`         -> predict
#
# The "bilinear" readout feature map (squaring every even-indexed reservoir
# unit) is reproduced exactly, including MATLAB's 1-based even-index convention
# `2:2:N`.
# =============================================================================

module Readout

using LinearAlgebra
using SparseArrays

export reservoir_layer, train_readout, predict, square_even_indices!

"""
    square_even_indices!(X)

In-place squaring of every even-indexed *row* of `X` (MATLAB `X(2:2:N,:)`).
Works for both a matrix of states (`N × T`) and a single state vector (`N`).
This is the symmetry-breaking feature map of Pathak et al.; the readout sees
`[x_1, x_2², x_3, x_4², …]`.
"""
function square_even_indices!(X::AbstractMatrix)
    @inbounds for j in axes(X, 2)
        for i in 2:2:size(X, 1)
            X[i, j] = X[i, j]^2
        end
    end
    return X
end

function square_even_indices!(x::AbstractVector)
    @inbounds for i in 2:2:length(x)
        x[i] = x[i]^2
    end
    return x
end

"""
    reservoir_layer(input, A, w_in, resparams, train_length, activation) -> states

Port of `reservoir_layer.m`. Drive the reservoir with the **true** input
(teacher forcing / open loop) and record the state trajectory.

    states[:, 1]   = 0                                   (zero initial state)
    states[:, i+1] = f( A·states[:, i] + w_in·input[:, i] ),  i = 1 … L-1

# Arguments
- `input`        : `num_inputs × T` driving signal (KS field).
- `A`            : `N × N` recurrent matrix.
- `w_in`         : `N × num_inputs` input matrix.
- `resparams`    : `ReservoirParams` (uses `N`).
- `train_length` : number of state columns `L` to collect.
- `activation`   : nonlinearity `f` (defaults set by caller; paper uses `tanh`).

Returns the `N × train_length` state matrix.
"""
function reservoir_layer(input::AbstractMatrix, A::AbstractMatrix,
                         w_in::AbstractMatrix, resparams, train_length::Integer,
                         activation)
    N = resparams.N
    states = zeros(Float64, N, train_length)
    @inbounds for i in 1:(train_length - 1)
        states[:, i + 1] = activation(A * states[:, i] .+ w_in * input[:, i])
    end
    return states
end

"""
    train_readout(resparams, states, target) -> w_out

Port of `train.m`. Ridge-regression readout with the even-index squaring
feature map:

    states[2:2:N, :] .= states[2:2:N, :].^2
    w_out = target · statesᵀ · (states·statesᵀ + β·I)⁻¹

# Arguments
- `resparams` : `ReservoirParams` (uses `N`, `beta`).
- `states`    : `N × T` collected states (modified in place by the squaring,
  exactly as MATLAB mutates `states`).
- `target`    : `num_inputs × T` target outputs (the KS field over the train
  window).

Returns `w_out` of size `num_inputs × N`.

Implementation note: MATLAB uses `pinv(states*states' + βI)`. For the
well-conditioned Tikhonov system (`β > 0`) a direct solve is numerically
equivalent and far cheaper; we solve `(G + βI) w_outᵀ = (target·statesᵀ)ᵀ`.
"""
function train_readout(resparams, states::AbstractMatrix, target::AbstractMatrix)
    N = resparams.N
    β = resparams.beta

    # Apply the bilinear feature map in place (mirrors MATLAB mutation).
    square_even_indices!(states)

    # Gram matrix and cross-covariance.
    G = states * transpose(states)          # N × N
    G[diagind(G)] .+= β                      # + β I  (Tikhonov ridge)
    P = target * transpose(states)          # num_inputs × N

    # Solve (G) w_outᵀ = Pᵀ  ⇒  w_out = (G \ Pᵀ)ᵀ. `Symmetric` for stability.
    w_out = transpose(Symmetric(G) \ transpose(P))
    return Matrix(w_out)
end

"""
    predict(x, A, w_in, resparams, w_out, predict_length, activation) -> (output, x)

Port of `predict.m`. Run the trained reservoir **autonomously**: each predicted
output is fed back as the next input.

    for i = 1 … predict_length
        x_aug          = square_even_indices(x)
        output[:, i]   = w_out · x_aug
        x              = f( A·x + w_in·output[:, i] )

Returns the `num_inputs × predict_length` `output` and the final reservoir
state `x`.
"""
function predict(x::AbstractVector, A::AbstractMatrix, w_in::AbstractMatrix,
                 resparams, w_out::AbstractMatrix, predict_length::Integer,
                 activation)
    num_inputs = resparams.num_inputs
    output = zeros(Float64, num_inputs, predict_length)
    x = copy(x)
    @inbounds for i in 1:predict_length
        x_aug = copy(x)
        square_even_indices!(x_aug)
        out = w_out * x_aug
        output[:, i] = out
        x = activation(A * x .+ w_in * out)
    end
    return output, x
end

end # module Readout
