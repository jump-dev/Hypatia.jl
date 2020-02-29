#=
Copyright 2019, Chris Coey and contributors
=#

using Test
using DataFrames
using Printf
import Hypatia
const SO = Hypatia.Solvers

instance_sets = [
    "fast",
    # "slow",
    ]

example_names = [
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

# TODO could use a single TimerOutputs object for all tests and display that at end also

T = Float64
options = (atol = 10eps(T)^0.25, solver = SO.Solver{T}(
    verbose = false, iter_limit = 250, time_limit = 12e2,
    system_solver = SO.QRCholDenseSystemSolver{T}(),
    ))

@info("starting native examples tests")

for ex in example_names
    include(joinpath(@__DIR__, "../examples", ex, "native.jl"))
end

perf = DataFrame(
    example = String[],
    inst = Int[],
    inst_data = String[],
    test_time = Float64[],
    solve_time = Float64[],
    iters = Int[],
    status = Symbol[],
    prim_obj = T[],
    dual_obj = T[],
    )

all_tests_time = time()
@testset "native examples tests" begin
    @testset "$example, $inst_set" for example in example_names, inst_set in instance_sets
        println()
        test_function = eval(Symbol("test_", example))
        instances = eval(Symbol("instances_", example, "_", inst_set))
        @testset "$example $inst: $inst_data" for (inst, inst_data) in enumerate(instances)
            inst_string = string(inst_data)
            println(example, " ", inst, ": ", inst_data, " ...")
            test_time = @elapsed r = test_function(T, inst_data, options = options)
            push!(perf, (example, inst, inst_string, test_time, r.solve_time, r.num_iters, r.status, r.primal_obj, r.dual_obj))
            @printf("... %8.2e seconds\n", test_time)
        end
    end
    println("")
    @info("native examples tests time: $(time() - all_tests_time) seconds")
    println("")
    show(perf, allrows = true, allcols = true)
    println("\n")
end

# TODO update linear operators tests below when support linear operators again

# instance_sets = [
#     "linops",
#     ]
#
# example_names = [
#     "densityest",
#     "envelope",
#     "expdesign",
#     "linearopt",
#     "matrixcompletion",
#     "matrixregression",
#     "maxvolume",
#     "polymin",
#     "portfolio",
#     "sparsepca",
#     ]

# tol = sqrt(sqrt(eps(T)))
# options = (atol = 10 * tol, solver = SO.Solver{T}(
#     verbose = true, init_use_indirect = true, reduce = false, preprocess = false, iter_limit = 250,
#     time_limit = 12e2, tol_feas = tol / 10, tol_abs_opt = tol / 10, tol_rel_opt = tol / 10,
#     system_solver = SO.NaiveIndirectSystemSolver{T}()))
#
# @info("starting native examples linear operators tests")
# @testset "native examples linear operators tests" begin
#     @testset "densityest" begin test_densityest.(instances_densityest_linops, T = T, options = options) end
#     @testset "expdesign" begin test_expdesign.(instances_expdesign_linops, T = T, options = options) end
#     @testset "polymin" begin test_polymin.(instances_polymin_linops, T = T, options = options) end
#     @testset "portfolio" begin test_portfolio.(instances_portfolio_linops, T = T, options = options) end
#     @testset "sparsepca" begin test_sparsepca.(instances_sparsepca_linops, T = T, options = options) end
# end
