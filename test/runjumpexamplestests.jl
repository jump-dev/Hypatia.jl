#=
Copyright 2019, Chris Coey and contributors
=#

using Test
import Hypatia

examples_dir = joinpath(@__DIR__, "../examples")

include(joinpath(examples_dir, "centralpolymat/JuMP.jl"))
include(joinpath(examples_dir, "contraction/JuMP.jl"))
include(joinpath(examples_dir, "densityest/JuMP.jl"))
include(joinpath(examples_dir, "envelope/JuMP.jl"))
include(joinpath(examples_dir, "expdesign/JuMP.jl"))
include(joinpath(examples_dir, "lotkavolterra/JuMP.jl"))
include(joinpath(examples_dir, "muconvexity/JuMP.jl"))
include(joinpath(examples_dir, "polymin/JuMP.jl"))
include(joinpath(examples_dir, "polynorm/JuMP.jl"))
include(joinpath(examples_dir, "regionofattr/JuMP.jl"))
include(joinpath(examples_dir, "secondorderpoly/JuMP.jl"))
include(joinpath(examples_dir, "semidefinitepoly/JuMP.jl"))
include(joinpath(examples_dir, "shapeconregr/JuMP.jl"))

@info("starting JuMP examples tests")
@testset "JuMP examples" begin
    # TODO rewrite like test/native.jl functions, to take T and all options and use build_solve_check
    JuMP_options = (
        verbose = false,
        test_certificates = true,
        max_iters = 250,
        time_limit = 6e2, # 1 minute
        )

    # @testset "centralpolymat" begin test_centralpolymatJuMP(; JuMP_options...,
    #     tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, tol_feas = 1e-7) end
    #
    # @testset "contraction" begin test_contractionJuMP(; JuMP_options...,
    #     tol_rel_opt = 1e-4, tol_abs_opt = 1e-4, tol_feas = 1e-4) end
    #
    # @testset "densityest" begin test_densityestJuMP(; JuMP_options...,
    #     tol_rel_opt = 1e-5, tol_abs_opt = 1e-5, tol_feas = 1e-6) end

    @testset "envelope" begin test_envelopeJuMP(; JuMP_options...) end

    @testset "expdesign" begin test_expdesignJuMP(; JuMP_options...) end

    @testset "lotkavolterra" begin test_lotkavolterraJuMP(; JuMP_options...,
        tol_rel_opt = 1e-5, tol_abs_opt = 1e-6, tol_feas = 1e-6) end

    # @testset "muconvexity" begin test_muconvexityJuMP(; JuMP_options...) end

    @testset "polymin" begin test_polyminJuMP(; JuMP_options...,
        tol_rel_opt = 1e-9, tol_abs_opt = 1e-8, tol_feas = 1e-9) end

    # @testset "polynorm" begin test_polynormJuMP(; JuMP_options...) end

    # @testset "regionofattr" begin test_regionofattrJuMP(; JuMP_options...,
    #     tol_abs_opt = 1e-6, tol_rel_opt = 1e-6, tol_feas = 1e-6) end
    #
    # @testset "secondorderpoly" begin test_secondorderpolyJuMP(; JuMP_options...) end
    #
    # @testset "semidefinitepoly" begin test_semidefinitepolyJuMP(; JuMP_options...,
    #     tol_abs_opt = 1e-7, tol_rel_opt = 1e-7, tol_feas = 1e-7) end
    #
    # @testset "shapeconregr" begin test_shapeconregrJuMP(; JuMP_options...,
    #     tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, tol_feas = 1e-6) end
end
