#=
Copyright 2019, Chris Coey and contributors
=#

using Test
import Hypatia
const SO = Hypatia.Solvers

examples_dir = joinpath(@__DIR__, "../examples")

include(joinpath(examples_dir, "densityest/native.jl"))
include(joinpath(examples_dir, "envelope/native.jl"))
include(joinpath(examples_dir, "expdesign/native.jl"))
include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "matrixcompletion/native.jl"))
include(joinpath(examples_dir, "matrixregression/native.jl"))
include(joinpath(examples_dir, "maxvolume/native.jl"))
include(joinpath(examples_dir, "polymin/native.jl"))
include(joinpath(examples_dir, "portfolio/native.jl"))
include(joinpath(examples_dir, "sparsepca/native.jl"))

T = Float64

options = (atol = sqrt(sqrt(eps(T))), solver = SO.Solver{T}(
    verbose = true, iter_limit = 250, time_limit = 12e2,
    system_solver = SO.QRCholDenseSystemSolver{T}(),
    ))

@info("starting native examples tests")
@testset "native examples tests" begin
    @testset "densityest" begin test_densityest.(instances_densityest_few, T = T, options = options) end
    @testset "envelope" begin test_envelope.(instances_envelope_few, T = T, options = options) end
    @testset "expdesign" begin test_expdesign.(instances_expdesign_few, T = T, options = options) end
    @testset "linearopt" begin test_linearopt.(instances_linearopt_few, T = T, options = options) end
    @testset "matrixcompletion" begin test_matrixcompletion.(instances_matrixcompletion_few, T = T, options = options) end
    @testset "matrixregression" begin test_matrixregression.(instances_matrixregression_few, R = T, options = options) end # real
    @testset "matrixregression" begin test_matrixregression.(instances_matrixregression_few, R = Complex{T}, options = options) end # complex
    @testset "maxvolume" begin test_maxvolume.(instances_maxvolume_few, T = T, options = options) end
    if T == Float64 # some ModelUtilities functions only work with Float64
        @testset "polymin" begin test_polymin.(instances_polymin_few, T = T, options = options) end
    end
    @testset "portfolio" begin test_portfolio.(instances_portfolio_few, T = T, options = options) end
    @testset "sparsepca" begin test_sparsepca.(instances_sparsepca_few, T = T, options = options) end
end

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
