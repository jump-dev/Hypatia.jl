#=
helpers for benchmark run scripts
=#

using Test
using Printf
import DataFrames
import CSV
examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common_JuMP.jl"))
include(joinpath(examples_dir, "common_native.jl"))

setup_benchmark_dataframe() = DataFrames.DataFrame(
    example = String[],
    inst_set = String[],
    real_T = Type{<:Real}[],
    count = Int[],
    inst_data = Tuple[],
    extender = String[],
    # solver = String[],
    # solver_options = Tuple[],
    # stepper = Symbol[],
    # toa = Bool[],
    # curve = Bool[],
    # shift = Int[],
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

function record_instance(
    perf,
    results_path,
    ex_type::Type,
    real_T::Type,
    inst_set,
    inst,
    count::Int,
    new_default_options,
    verbose,
    )
    ex_type_T = ex_type{real_T}

    total_time = @elapsed p = run_instance(ex_type_T, inst..., default_options = new_default_options, verbose = verbose)

    example = string(ex_type)
    inst_data = inst[1]
    extender = string(get_extender(inst, ex_type_T))
    inst_perf = (; example, inst_set, real_T, count, inst_data, extender, p..., total_time)

    push!(perf, inst_perf)
    if !isnothing(results_path)
        CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
    end

    return
end

get_extender(inst, ::Type{<:ExampleInstanceJuMP{<:Real}}) = (length(inst) > 1 ? inst[2] : nothing)
get_extender(inst, ::Type{<:ExampleInstance{<:Real}}) = nothing
