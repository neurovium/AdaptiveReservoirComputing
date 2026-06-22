# =============================================================================
# ResEvoKS_Simulation.jl  —  top-level module
# -----------------------------------------------------------------------------
# Evolutionary Reservoir Computing for Kuramoto-Sivashinsky spatiotemporal chaos.
#
# This is the Julia port of the MATLAB `64D-KS-Sim` simulation suite that
# accompanies the paper
#
#   "Evolutionary Optimization Reveals Structural Constraints on Reservoir
#    Architecture for Spatiotemporal Chaos", N. Dehghani.
#
# Module map (each submodule ports one or more MATLAB files; see
# REPORT_original_code.md for the full correspondence):
#
#   KuramotoSivashinsky  ← kuramoto_sivashinsky_solve.m              (ETDRK4 KS solver)
#   Activation           ← ActFunc.m, generate_activationFunction.m
#   Reservoir            ← generate_reservoir.m + spectral/input construction
#   Readout              ← reservoir_layer.m, train.m, predict.m
#   Metrics              ← compute_error.m + the J = j1/j2 fitness
#   IO                   ← .mat saving compatible with the analysis pipeline
#   Evaluation           ← the quickOptimizePredESN objective body
#   Optimization         ← the GA driver (KS64D_runOptimizePredESN.m)
#
# Quick start (see scripts/run_optimization.jl and the wiki for full usage):
#
#   using ResEvoKS_Simulation
#   p    = KSModelParams(N=64, d=22.0, tau=0.25, nstep=20_000)
#   ic   = random_initial_condition(p.N)
#   data = solve_ks(ic, p)
#   sp   = make_run_dirs("results", 1)
#   dp   = DataParams(train_length=14_000, predict_length=2_000)
#   res  = optimize_reservoirs(data, dp, sp;
#               settings=GASettings(population_size=30, max_generations=10))
# =============================================================================

module ResEvoKS_Simulation

# --- submodules (order matters: later ones depend on earlier ones) ----------
include("KuramotoSivashinsky.jl")
include("Activation.jl")
include("Reservoir.jl")
include("Readout.jl")
include("Metrics.jl")
include("IO.jl")
include("RunLog.jl")
include("Evaluation.jl")
include("Optimization.jl")

using .KuramotoSivashinsky
using .Activation
using .Reservoir
using .Readout
using .Metrics
using .IO
using .RunLog
using .Evaluation
using .Optimization

# --- public API (re-exported from the submodules) ---------------------------
# KS solver
export KSModelParams, solve_ks, random_initial_condition
# activations
export tanh_activation, generalized_logistic
# reservoir construction
export ReservoirParams, generate_reservoir, build_reservoir,
       scale_spectral_radius!, spectral_radius, generate_input_weights
# readout
export reservoir_layer, train_readout, predict, square_even_indices!
# metrics
export PredictionError, compute_error, composite_fitness
# IO
export SaveParams, make_run_dirs, save_individual, save_dataset,
       individual_filename, load_individual, load_dataset, SAVE_FORMATS
# logging
export RunLogger, open_run_logger, logmsg, close_logger
# evaluation + optimization
export DataParams, decode_genome, EvalResult, evaluate_individual
export GASettings, GAResult, optimize_reservoirs

end # module ResEvoKS_Simulation
