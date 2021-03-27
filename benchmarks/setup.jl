#=
helpers for benchmark run scripts
=#

using Printf
import DataFrames
import CSV
examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common_JuMP.jl"))
include(joinpath(examples_dir, "common_native.jl"))

function setup_benchmark_dataframe()
    perf = DataFrames.DataFrame(
        example = String[],
        inst_set = String[],
        real_T = Type{<:Real}[],
        inst_num = Int[],
        inst_data = Tuple[],
        extender = String[],
        solver = String[],
        solver_options = Tuple[],
        n = Int[],
        p = Int[],
        q = Int[],
        nu = Float64[],
        cone_types = Vector{String}[],
        num_cones = Int[],
        max_q = Int[],
        status = String[],
        solve_time = Float64[],
        iters = Int[],
        primal_obj = Float64[],
        dual_obj = Float64[],
        rel_obj_diff = Float64[],
        compl = Float64[],
        x_viol = Float64[],
        y_viol = Float64[],
        z_viol = Float64[],
        time_rescale = Float64[],
        time_initx = Float64[],
        time_inity = Float64[],
        time_unproc = Float64[],
        time_loadsys = Float64[],
        time_upsys = Float64[],
        time_upfact = Float64[],
        time_uprhs = Float64[],
        time_getdir = Float64[],
        time_search = Float64[],
        setup_time = Float64[],
        check_time = Float64[],
        total_time = Float64[],
        )
    DataFrames.allowmissing!(perf, 9:DataFrames.ncol(perf))
    return perf
end

get_extender(inst::Tuple, ::Type{<:ExampleInstanceJuMP{<:Real}}) = (length(inst) > 1 ? string(inst[2]) : "")
get_extender(inst::Tuple, ::Type{<:ExampleInstance{<:Real}}) = ""

function write_perf(
    perf::DataFrames.DataFrame,
    results_path::Union{String, Nothing},
    new_perf::NamedTuple,
    )
    push!(perf, new_perf)
    if !isnothing(results_path)
        CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
    end
    return
end
