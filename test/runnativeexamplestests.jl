#=
Copyright 2019, Chris Coey and contributors
=#

using Test
import Hypatia
const SO = Hypatia.Solvers

instance_sets = [
    "fast",
    # "slow",
    ]

example_names = [
    "densityest",
    "envelope",
    # "expdesign",
    "linearopt",
    # "matrixcompletion",
    # "matrixregression",
    # "maxvolume",
    # "polymin",
    # "portfolio",
    # "sparsepca",
    ]

T = Float64
options = (atol = 10eps(T)^0.25, solver = SO.Solver{T}(
    verbose = false, iter_limit = 250, time_limit = 12e2,
    system_solver = SO.QRCholDenseSystemSolver{T}(),
    ))

@info("starting native examples tests")

for ex_name in example_names
    include(joinpath(@__DIR__, "../examples", ex_name, "native.jl"))
end

perf = Dict() # TODO maybe use a dataframe or table instead of dictionary

println("\nprinting: (test_time, solve_time, num_iters, status, primal_obj, dual_obj)\n")
@testset "native examples tests" begin
    @testset "$ex_name, $inst_set" for ex_name in example_names, inst_set in instance_sets
        instances = eval(Symbol("instances_", ex_name, "_", inst_set))
        test_function = eval(Symbol("test_", ex_name))
        @testset "$ex_name $i $inst" for (i, inst) in enumerate(instances)
            println(ex_name, " ", i, ": ", inst, " ...")
            test_time = @elapsed r = test_function(T, inst, options = options)
            perf[inst] = (test_time, r.solve_time, r.num_iters, r.status, r.primal_obj, r.dual_obj)
            print(perf[inst], "\n\n")
        end
    end
end

println()
display(perf)

# TODO currently broken
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
