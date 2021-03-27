#=
auxiliary code for testing examples tests from the examples folder
=#

using Test
using Printf
import DataFrames
import CSV
import Hypatia.Solvers
examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common_JuMP.jl"))
include(joinpath(examples_dir, "common_native.jl"))

perf = DataFrames.DataFrame(
    example = String[],
    inst_set = String[],
    real_T = Type{<:Real}[],
    count = Int[],
    inst_data = Tuple[],
    extender = String[],
    stepper = Symbol[],
    toa = Bool[],
    curve = Bool[],
    shift = Int[],
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

use_curve_search(::Solvers.Stepper) = true
use_curve_search(stepper::Solvers.PredOrCentStepper) = stepper.use_curve_search
use_correction(::Solvers.Stepper) = true
use_correction(stepper::Solvers.PredOrCentStepper) = stepper.use_correction
shift(::Solvers.Stepper) = 0
shift(stepper::Solvers.CombinedStepper) = stepper.shift_alpha_sched

function test_instances(mod_type, steppers, instance_sets, ex_insts, ex_type, default_options, results_path)
    for stepper in steppers, (inst_set, real_T, time_limit) in instance_sets
        haskey(ex_insts, inst_set) || continue
        inst_subset = ex_insts[inst_set]
        isempty(inst_subset) && continue
        ex_type_T = ex_type{real_T}
        new_default_options = (; default_options..., time_limit = time_limit, stepper = stepper)

        println("\nstarting $ex_type_T $inst_set tests")
        @testset "$ex_type_T $inst_set" begin
        for (inst_num, inst) in enumerate(inst_subset)
            test_info = "inst $inst_num: $(inst[1])"
            @testset "$test_info" begin
                println(test_info, " ...")
                test_time = @elapsed p = run_instance(ex_type_T, inst..., default_options = new_default_options, verbose = false)
                extender = (length(inst) > 1 && mod_type == "JuMP") ? inst[2] : nothing
                push!(perf, (string(ex_type), inst_set, real_T, inst_num, inst[1], string(extender), nameof(typeof(stepper)),
                    use_correction(stepper), use_curve_search(stepper), shift(stepper), p..., test_time))
                isnothing(results_path) || CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
                @printf("%8.2e seconds\n", test_time)
            end
        end
        end
    end
    return
end

return (perf, test_instances)
