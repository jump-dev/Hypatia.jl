#=
Copyright 2019, Chris Coey and contributors
=#

using Test
import Hypatia
const MO = Hypatia.Models
const SO = Hypatia.Solvers

examples_dir = joinpath(@__DIR__, "../examples")

include(joinpath(examples_dir, "densityest/native.jl"))
include(joinpath(examples_dir, "envelope/native.jl"))
include(joinpath(examples_dir, "expdesign/native.jl"))
include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "matrixcompletion/native.jl"))
include(joinpath(examples_dir, "polymin/native.jl"))
include(joinpath(examples_dir, "portfolio/native.jl"))
include(joinpath(examples_dir, "sparsepca/native.jl"))

@info("starting native examples tests")

real_types = [
    Float64,
    Float32,
    ]

@testset "native examples: $T" for T in real_types
    test_options = (
        solver_options = (verbose = false,),
        )

    @testset "densityest" begin test_densityest.(instances_densityest_few, T = T, test_options = test_options) end

    @testset "envelope" begin test_envelope.(instances_envelope_few, T = T, test_options = test_options) end

    @testset "expdesign" begin test_expdesign.(instances_expdesign_few, T = T, test_options = test_options) end

    @testset "linearopt" begin test_linearopt.(instances_linearopt_few, T = T, test_options = test_options) end

    @testset "matrixcompletion" begin test_matrixcompletion.(instances_matrixcompletion_few, T = T, test_options = test_options) end

    @testset "sparsepca" begin test_sparsepca.(instances_sparsepca_few, T = T, test_options = test_options) end

    @testset "polymin" begin test_polymin.(instances_polymin_few, T = T, test_options = test_options) end

    @testset "portfolio" begin test_portfolio.(instances_portfolio_few, T = T, test_options = test_options) end
end

@testset "native examples (linear operators) : $T" for T in real_types
    test_options = (
        linear_model = MO.RawLinearModel,
        system_solver = SO.NaiveCombinedHSDSystemSolver,
        linear_model_options = (use_iterative = true,),
        system_solver_options = (use_iterative = true,),
        solver_options = (verbose = false, tol_feas = 1e-5, tol_abs_opt = 1e-5, tol_rel_opt = 1e-5),
        )

    @testset "densityest" begin test_densityest.(instances_densityest_linops, T = T, test_options = test_options) end

    @testset "expdesign" begin test_expdesign.(instances_expdesign_linops, T = T, test_options = test_options) end

    @testset "sparsepca" begin test_sparsepca.(instances_sparsepca_linops, T = T, test_options = test_options) end

    @testset "polymin" begin test_polymin.(instances_polymin_linops, T = T, test_options = test_options) end

    @testset "portfolio" begin test_portfolio.(instances_portfolio_linops, T = T, test_options = test_options) end
end
