#=
Copyright 2019, Chris Coey and contributors

run examples tests from the examples folder and display basic benchmarks

TODO when linear operators are working, update linear operators tests in native tests add tests here
=#

examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common.jl"))
using DataFrames
using Printf
using TimerOutputs

# options to solvers
timer = TimerOutput()
default_solver_options = (
    verbose = false,
    iter_limit = 250,
    timer = timer,
    )

# instance sets and real types to run and corresponding time limits (seconds)
instance_sets = [
    (MinimalInstances, Float64, 15),
    (MinimalInstances, Float32, 15),
    (MinimalInstances, BigFloat, 60),
    # (FastInstances, Float64, 15),
    # (SlowInstances, Float64, 120),
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
    "centralpolymat",
    "conditionnum",
    "contraction",
    "densityest",
    "envelope",
    "expdesign",
    "lotkavolterra",
    "lyapunovstability",
    "matrixcompletion",
    "matrixquadratic",
    "matrixregression",
    "maxvolume",
    "muconvexity",
    "nearestpsd",
    "polymin",
    "polynorm",
    "portfolio",
    "regionofattr",
    "robustgeomprog",
    "secondorderpoly",
    "semidefinitepoly",
    "shapeconregr",
    "signomialmin",
    ]

# types of models to run and corresponding options and example names
model_types = [
    "native",
    "JuMP",
    ]

# start the tests
@info("starting examples tests")
for (inst_set, real_T, time_limit) in instance_sets
    @info("each $inst_set $real_T instance should take <$time_limit seconds")
end

example_types = Tuple{String, Type{<:ExampleInstance}}[]
for mod_type in model_types, ex in eval(Symbol(mod_type, "_example_names"))
    ex_type = include(joinpath(examples_dir, ex, mod_type * ".jl"))
    push!(example_types, (ex, ex_type))
end

perf = DataFrame(
    example = String[],
    real_T = String[],
    inst = Int[],
    inst_data = Tuple[],
    test_time = Float64[],
    solve_time = Float64[],
    iters = Int[],
    status = Symbol[],
    prim_obj = Float64[],
    dual_obj = Float64[],
    )

all_tests_time = time()

@testset "examples tests" begin
    for (ex_name, ex_type) in example_types, (inst_set, real_T, time_limit) in instance_sets
        solver_options = (default_solver_options..., time_limit = time_limit)
        ex_type_T = ex_type{real_T}
        instances = example_tests(ex_type_T, inst_set())
        println("\nstarting $(length(instances)) instances for $ex_type_T $inst_set\n")
        for (inst_num, inst) in enumerate(instances)
            test_info = "$ex_type_T $inst_set $inst_num: $(inst[1])"
            @testset "$test_info" begin
                println(test_info, "...")
                test_time = @elapsed r = test(ex_type_T, inst..., default_solver_options = solver_options)
                push!(perf, (
                    string(ex_type), string(real_T), inst_num, inst[1], test_time,
                    r.solve_time, r.num_iters, r.status, r.primal_obj, r.dual_obj))
                @printf("... %8.2e seconds\n", test_time)
            end
        end
    end

    @printf("\nexamples tests total time: %8.2e seconds\n\n", time() - all_tests_time)
    show(perf, allrows = true, allcols = true)
    println("\n")
    # show(timer)
    # println("\n")
end

;
