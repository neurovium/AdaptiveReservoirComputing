# =============================================================================
# RunLog.jl
# -----------------------------------------------------------------------------
# A small, dependency-light run logger for tracing long optimization runs.
#
# This is the Julia analogue of the MATLAB `diary(...)` + `enable_logging`
# machinery in `KS64D_runOptimizePredESN.m`: a timestamped, wall-clock-stamped
# log written to the run's `Log/` directory so that, when launching many runs or
# many generations, one can trace progress and see exactly where (and when) a
# run broke.
#
# Design choices that matter for long runs:
#   * Every line carries an absolute timestamp AND the elapsed seconds since the
#     run started, so timing is visible at a glance.
#   * Each write is flushed immediately, so a crash leaves a complete log on disk
#     up to the last successful line.
#   * The same line can be echoed to the console (so stdout and the file agree).
# =============================================================================

module RunLog

using Dates
using Printf

export RunLogger, open_run_logger, logmsg, close_logger, logpath

"""
    RunLogger

A lightweight tee logger. Construct with [`open_run_logger`](@ref); write lines
with [`logmsg`](@ref); release the file with [`close_logger`](@ref).

# Fields
- `io::Union{Nothing,IOStream}` : the open log file, or `nothing` (console only).
- `echo::Bool`                  : also print each line to `stdout`.
- `t0::Float64`                 : reference wall-clock time (`time()`) for the
                                  "+Xs" elapsed stamp.
- `path::String`                : the log file path (empty if console-only).
"""
struct RunLogger
    io::Union{Nothing,IOStream}
    echo::Bool
    t0::Float64
    path::String
end

"""
    open_run_logger(path; echo=true, t0=time()) -> RunLogger

Open a log file at `path` (creating it / truncating any previous file). Pass
`path = nothing` for a console-only logger (no file). `t0` anchors the elapsed
stamp; pass the run's start time so file timing matches the reported total.
"""
function open_run_logger(path::Union{Nothing,AbstractString}; echo::Bool=true,
                         t0::Float64=time())
    io = path === nothing ? nothing : open(String(path), "w")
    return RunLogger(io, echo, t0, path === nothing ? "" : String(path))
end

"""
    logmsg(lg::RunLogger, msg)

Write one timestamped line: `[YYYY-mm-dd HH:MM:SS | +<elapsed>s] <msg>`. The line
is flushed to the file immediately and (if `lg.echo`) printed to `stdout`.
"""
function logmsg(lg::RunLogger, msg::AbstractString)
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    el = time() - lg.t0
    line = @sprintf("[%s | +%9.1fs] %s", ts, el, msg)
    if lg.io !== nothing
        println(lg.io, line)
        flush(lg.io)
    end
    lg.echo && println(line)
    return nothing
end

"""
    close_logger(lg::RunLogger)

Close the underlying file (no-op for a console-only logger).
"""
close_logger(lg::RunLogger) = lg.io !== nothing && close(lg.io)

"""
    logpath(lg::RunLogger) -> String

The log file path (empty string if console-only).
"""
logpath(lg::RunLogger) = lg.path

end # module RunLog
