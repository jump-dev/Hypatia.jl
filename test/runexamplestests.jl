#=
Copyright 2019, Chris Coey and contributors

run examples tests from the examples folder and display basic benchmarks

to run benchmarks, use the bench1 instance set and run on cmd line:
~/julia/julia test/runexamplestests.jl &> ~/bench1/bench1.txt
=#

examples_dir = joinpath(@__DIR__, "../examples")
# TODO maybe put these common files in a module
include(joinpath(examples_dir, "common.jl"))
include(joinpath(examples_dir, "common_JuMP.jl"))
include(joinpath(examples_dir, "common_native.jl"))
import DataFrames
import CSV
using Printf
import TimerOutputs
import DataStructures

# path to write results DataFrame to CSV, if any
results_path = joinpath(homedir(), "bench1", "bench1.csv")
# results_path = nothing

# options to solvers
timer = TimerOutputs.TimerOutput()
default_solver_options = (
    # verbose = false,
    verbose = true,
    iter_limit = 150,
    timer = timer,
    # system_solver = Solvers.NaiveDenseSystemSolver{Float64}(),
    # system_solver = Solvers.SymIndefDenseSystemSolver{Float64}(),
    system_solver = Solvers.QRCholDenseSystemSolver{Float64}(),
    )

# instance sets and real types to run and corresponding time limits (seconds)
instance_sets = [
    # ("minimal", Float64, 15),
    # ("minimal", Float32, 15),
    # ("minimal", BigFloat, 15),
    # ("fast", Float64, 15),
    # ("slow", Float64, 120),
    ("bench1", Float64, 1800),
    ]

# types of models to run and corresponding options and example names
model_types = [
    # "native",
    "JuMP",
    ]

# list of names of native examples to run
native_example_names = [
    "densityest",
    "envelope",
    "expdesign",
    "linearopt",
    "matrixcompletion",
    "matrixregression",
    "maxvolume",
    "polymin",
    "portfolio",
    "sparsepca",
    ]

# list of names of JuMP examples to run
JuMP_example_names = [
    # "centralpolymat",
    # "conditionnum",
    # "contraction",
    "densityest",
    # "envelope",
    "expdesign",
    # # "lotkavolterra", # TODO PolyJuMP error
    # "lyapunovstability",
    "matrixcompletion",
    "matrixquadratic",
    "matrixregression",
    # "maxvolume",
    # "muconvexity",
    "nearestpsd",
    "polymin",
    # "polynorm",
    "portfolio",
    # # "regionofattr", # TODO PolyJuMP error
    # "robustgeomprog",
    # "secondorderpoly",
    # "semidefinitepoly",
    "shapeconregr",
    "signomialmin",
    ]

# start the tests
@info("starting examples tests")
for (inst_set, real_T, time_limit) in instance_sets
    @info("each $inst_set $real_T instance should take <$time_limit seconds")
end

# load instances
instances = DataStructures.DefaultOrderedDict{Type{<:ExampleInstance}, Any}(() -> DataStructures.DefaultOrderedDict{String, Any}(() -> Tuple[]))
for mod_type in model_types, ex_name in eval(Symbol(mod_type, "_example_names"))
    include(joinpath(examples_dir, ex_name, mod_type * ".jl"))
end

perf = DataFrames.DataFrame(
    example = Type{<:ExampleInstance}[],
    inst_set = String[],
    real_T = Type{<:Real}[],
    count = Int[],
    inst_data = Tuple[],
    extender = Any[],
    test_time = Float64[],
    build_time = Float64[],
    status = Symbol[],
    solve_time = Float64[],
    iters = Int[],
    prim_obj = Float64[],
    dual_obj = Float64[],
    obj_diff = Float64[],
    compl = Float64[],
    x_viol = Float64[],
    y_viol = Float64[],
    z_viol = Float64[],
    ext_n = Int[],
    ext_p = Int[],
    ext_q = Int[],
    )

isnothing(results_path) || CSV.write(results_path, perf)
all_tests_time = time()

@testset "examples tests" begin
    for (ex_type, ex_insts) in instances, (inst_set, real_T, time_limit) in instance_sets
        instances = ex_insts[inst_set]
        isempty(instances) && continue
        ex_type_T = ex_type{real_T}
        println("\nstarting instances for $ex_type_T $inst_set\n")
        solver_options = (default_solver_options..., time_limit = time_limit)
        for (inst_num, inst) in enumerate(instances)
            test_info = "$ex_type_T $inst_set $inst_num: $inst"
            @testset "$test_info" begin
                println(test_info, "...")
                test_time = @elapsed (extender, build_time, r) = test(ex_type_T, inst..., default_solver_options = solver_options)
                push!(perf, (
                    ex_type, inst_set, real_T, inst_num, inst[1], extender, test_time, build_time,
                    r.status, r.solve_time, r.num_iters, r.primal_obj, r.dual_obj,
                    r.obj_diff, r.compl, r.x_viol, r.y_viol, r.z_viol,
                    r.n, r.p, r.q,
                    ))
                isnothing(results_path) || CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append=true)
                @printf("... %8.2e seconds\n", test_time)
            end
        end
    end

    @printf("\nexamples tests total time: %8.2e seconds\n\n", time() - all_tests_time)
    DataFrames.show(perf, allrows = true, allcols = true)
    println("\n")
    @show sum(perf[:iters])
    show(timer)
    println("\n")
end

;
