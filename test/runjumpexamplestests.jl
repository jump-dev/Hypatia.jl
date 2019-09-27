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

options = (
    test_certificates = false,
    verbose = true,
    iter_limit = 250,
    time_limit = 6e2, # 1 minute
    )

@info("starting JuMP examples tests")
@testset "JuMP examples" begin
    # @testset "centralpolymat" begin test_centralpolymatJuMP(; tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, tol_feas = 1e-7, options...) end
    # @testset "contraction" begin test_contractionJuMP(; tol_rel_opt = 1e-4, tol_abs_opt = 1e-4, tol_feas = 1e-4, options...) end
    # @testset "densityest" begin test_densityestJuMP(; tol_rel_opt = 1e-5, tol_abs_opt = 1e-5, tol_feas = 1e-6, options...) end
    @testset "envelope" begin test_envelopeJuMP(; tol_rel_opt = 1e-7, tol_abs_opt = 1e-8, tol_feas = 1e-8, options...) end
    @testset "expdesign" begin test_expdesignJuMP(; tol_rel_opt = 1e-6, tol_abs_opt = 1e-7, tol_feas = 1e-7, options...) end
    @testset "lotkavolterra" begin test_lotkavolterraJuMP(; tol_rel_opt = 1e-5, tol_abs_opt = 1e-6, tol_feas = 1e-6, options...) end
    # @testset "muconvexity" begin test_muconvexityJuMP(; options...) end
    @testset "polymin" begin test_polyminJuMP(; tol_rel_opt = 1e-7, tol_abs_opt = 1e-8, tol_feas = 1e-8, options...) end
    # @testset "polynorm" begin test_polynormJuMP(; options...) end
    # @testset "regionofattr" begin test_regionofattrJuMP(; tol_abs_opt = 1e-6, tol_rel_opt = 1e-6, tol_feas = 1e-6, options...) end
    # @testset "secondorderpoly" begin test_secondorderpolyJuMP(; options...) end
    # @testset "semidefinitepoly" begin test_semidefinitepolyJuMP(; tol_abs_opt = 1e-7, tol_rel_opt = 1e-7, tol_feas = 1e-7, options...) end
    # @testset "shapeconregr" begin test_shapeconregrJuMP(; tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, tol_feas = 1e-6, options...) end
end
