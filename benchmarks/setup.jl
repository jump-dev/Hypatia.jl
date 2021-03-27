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
    solver = String[],
    solver_options = Tuple[],
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
    prim_obj = Float64[],
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

# # TODO more arguments can be simplified, things can be removed, eg stepper, new_default_options
# function test_instances(
#     stepper,
#     instance_set,
#     ex_insts,
#     ex_type,
#     default_options,
#     results_path,
#     )
#     (inst_set, real_T, time_limit) = instance_set
#     haskey(ex_insts, inst_set) || continue
#     inst_subset = ex_insts[inst_set]
#     isempty(inst_subset) && continue
#     ex_type_T = ex_type{real_T}
#     new_default_options = (; default_options..., time_limit = time_limit, stepper = stepper)
#
#     println("\nstarting $ex_type_T $inst_set tests")
#     @testset "$ex_type_T $inst_set" begin
#     for (inst_num, inst) in enumerate(inst_subset)
#         test_info = "inst $inst_num: $(inst[1])"
#         @testset "$test_info" begin
#             println(test_info, " ...")
#             test_time = @elapsed p = run_instance(ex_type_T, inst..., default_options = new_default_options, verbose = false)
#             extender = (length(inst) > 1 && ex_type_T <: ExampleInstanceJuMP{Float64}) ? inst[2] : nothing
#             push!(perf, (string(ex_type), inst_set, real_T, inst_num, inst[1], string(extender), solver_options, p..., test_time))
#             if !isnothing(results_path)
#                 CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
#             end
#             @printf("%8.2e seconds\n", test_time)
#         end
#     end
#     end
#
#     return
# end

get_extender(inst, ::Type{<:ExampleInstanceJuMP{<:Real}}) = (length(inst) > 1 ? inst[2] : nothing)
get_extender(inst, ::Type{Any}) = nothing

function record_instance(
    perf,
    results_path,
    ex_type_T::Type{<:ExampleInstance{real_T}},
    inst,
    inst_num::Int,
    new_default_options,
    verbose,
    ) where {real_T <: Real}
# where {ex_type_T <: ExampleInstance{real_T}, real_T <: Real}
    test_time = @elapsed p = run_instance(ex_type_T, inst..., default_options = new_default_options, verbose = verbose)

    extender = string(get_extender(inst, ex_type_T))
    ex_type = string(basename(ex_type_T))
    inst_perf = (string(ex_type), inst_set, real_T, inst_num, inst[1], extender, solver_options, p..., test_time)

    push!(perf, inst_perf)
    if !isnothing(results_path)
        CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
    end

    @printf("%8.2e seconds\n", test_time)
    return
end
