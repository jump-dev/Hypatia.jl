#=
Copyright 2019, Chris Coey and contributors
=#

using Test
import Hypatia
const SO = Hypatia.Solvers

@info("starting native examples tests")

examples_dir = joinpath(@__DIR__, "../examples")

include(joinpath(examples_dir, "densityest/native.jl"))
include(joinpath(examples_dir, "envelope/native.jl"))
include(joinpath(examples_dir, "expdesign/native.jl"))
include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "matrixcompletion/native.jl"))
include(joinpath(examples_dir, "polymin/native.jl"))
include(joinpath(examples_dir, "portfolio/native.jl"))
include(joinpath(examples_dir, "sparsepca/native.jl"))

real_types = [
    Float64,
    # Float32,
    ]

@info("starting miscellaneous tests")
@testset "miscellaneous tests: $T" for T in real_types
    options = (atol = sqrt(sqrt(eps(T))),
        solver = SO.Solver{T}(verbose = true, iter_limit = 250, time_limit = 12e2,
        system_solver = SO.NaiveSystemSolver{T}()))

    @testset "densityest" begin test_densityest.(instances_densityest_few, T = T, options = options) end

    # @testset "envelope" begin test_envelope.(instances_envelope_few, T = T, options = options) end
    #
    # @testset "expdesign" begin test_expdesign.(instances_expdesign_few, T = T, options = options) end
    #
    # @testset "linearopt" begin test_linearopt.(instances_linearopt_few, T = T, options = options) end
    #
    # @testset "matrixcompletion" begin test_matrixcompletion.(instances_matrixcompletion_few, T = T, options = options) end

    @testset "sparsepca" begin test_sparsepca.(instances_sparsepca_few, T = T, options = options) end

    # @testset "polymin" begin test_polymin.(instances_polymin_few, T = T, options = options) end
    #
    # @testset "portfolio" begin test_portfolio.(instances_portfolio_few, T = T, options = options) end
end

# @info("starting linear operators tests")
# @testset "linear operators tests: $T" for T in real_types
#     tol = sqrt(sqrt(eps(T)))
#     options = (atol = 10 * tol, solver = SO.Solver{T}(verbose = true, init_use_iterative = true,
#         preprocess = false, iter_limit = 250, time_limit = 12e2,
#         tol_feas = tol / 10, tol_abs_opt = tol / 10, tol_rel_opt = tol / 10,
#         system_solver = SO.NaiveSystemSolver{T}(use_iterative = true)))
#
#     @testset "densityest" begin test_densityest.(instances_densityest_linops, T = T, options = options) end
#
#     @testset "expdesign" begin test_expdesign.(instances_expdesign_linops, T = T, options = options) end
#
#     @testset "sparsepca" begin test_sparsepca.(instances_sparsepca_linops, T = T, options = options) end
#
#     @testset "polymin" begin test_polymin.(instances_polymin_linops, T = T, options = options) end
#
#     @testset "portfolio" begin test_portfolio.(instances_portfolio_linops, T = T, options = options) end
# end
