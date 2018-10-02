#=
Copyright 2018, Chris Coey and contributors
=#

using Hypatia
using Test


# native interface tests

# verbflag = false # Hypatia verbose option
#
# # TODO interpolation tests
#
# # load optimizer builder functions from examples folder
# egs_dir = joinpath(@__DIR__, "../examples")
# include(joinpath(egs_dir, "envelope/envelope.jl"))
# include(joinpath(egs_dir, "lp/lp.jl"))
# include(joinpath(egs_dir, "namedpoly/namedpoly.jl"))
#
# # run native and MOI interfaces on examples
# @testset "native interface tests" begin
#     include(joinpath(@__DIR__, "native.jl"))
# end

# MathOptInterface tests

import MathOptInterface
MOI = MathOptInterface
MOIT = MOI.Test
MOIB = MOI.Bridges
MOIU = MOI.Utilities

MOIU.@model(HypatiaModelData,
    (),
    (
        MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,
        # MOI.Interval,
    ),
    (
        MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
        MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
        MOI.PositiveSemidefiniteConeTriangle,
        MOI.ExponentialCone,
        # MOI.PowerCone,
    ),
    (),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    )

optimizer = MOIU.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.Optimizer())

config = MOIT.TestConfig(
    atol=1e-3,
    rtol=1e-3,
    solve=true,
    query=true,
    modify_lhs=true,
    duals=true,
    infeas_certificates=true,
    )

#
# function linear15test(model::MOI.ModelLike, config::MOIT.TestConfig)
#     atol = config.atol
#     rtol = config.rtol
#     # minimize 0
#     # s.t. 0 == 0
#     #      x == 1
#     @test MOI.supports(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
#     @test MOI.supports(model, MOI.ObjectiveSense())
#     @test MOI.supports_constraint(model, MOI.VectorAffineFunction{Float64}, MOI.Zeros)
#
#     MOI.empty!(model)
#     @test MOI.is_empty(model)
#
#     x = MOI.add_variables(model, 1)
#     # Create a VectorAffineFunction with two rows, but only
#     # one term, belonging to the second row. The first row,
#     # which is empty, is essentially a constraint that 0 == 0.
#     c = MOI.add_constraint(model,
#         MOI.VectorAffineFunction(
#             MOI.VectorAffineTerm.(2, MOI.ScalarAffineTerm.([1.0], x)),
#             zeros(2)
#         ),
#         MOI.Zeros(2)
#     )
#
#     MOI.set(model,
#         MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
#         MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0], x), 0.0))
#     MOI.set(model, MOI.ObjectiveSense(), MOI.MinSense)
#
#     if config.solve
#         MOI.optimize!(model)
#
#         @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#         @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
#
#         @test MOI.get(model, MOI.ObjectiveValue()) ≈ 0 atol=atol rtol=rtol
#
#         @test MOI.get(model, MOI.VariablePrimal(), x[1]) ≈ 0 atol=atol rtol=rtol
#     end
# end

function linear11test(model::MOI.ModelLike, config::MOIT.TestConfig)
    atol = config.atol
    rtol = config.rtol
    # simple 2 variable, 1 constraint problem
    # min x + y
    # st   x + y >= 1
    #      x + y >= 2

    @test MOI.supports(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    @test MOI.supports(model, MOI.ObjectiveSense())
    @test MOI.supports_constraint(model, MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64})
    @test MOI.supports_constraint(model, MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})

    MOI.empty!(model)
    @test MOI.is_empty(model)

    v = MOI.add_variables(model, 2)

    c1 = MOI.add_constraint(model, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], v), 0.0), MOI.GreaterThan(1.0))
    c2 = MOI.add_constraint(model, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], v), 0.0), MOI.GreaterThan(2.0))

    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], v), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MinSense)

    if config.solve
        MOI.optimize!(model)

        @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 2.0 atol=atol rtol=rtol
    end

    c3 = MOI.transform(model, c2, MOI.LessThan(2.0))

    @test isa(c3, MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}})
    @test MOI.is_valid(model, c2) == false
    @test MOI.is_valid(model, c3) == true

    if config.solve
        MOI.optimize!(model)

        @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1.0 atol=atol rtol=rtol
    end
end

# linear15test(optimizer, config)
linear11test(optimizer, config)

# @testset "MathOptInterface tests" begin
# @testset "Continuous linear problems" begin
#     MOIT.contlineartest(
#         MOIB.SplitInterval{Float64}(
#             optimizer
#         ),
#         config)
# end
# @testset "Continuous conic problems" begin
#     exclude = ["rootdet", "logdet", "sdp"] # TODO bridges not working? should not need to exclude in future
#     MOIT.contconictest(
#         # MOIB.SquarePSD{Float64}(
#         MOIB.GeoMean{Float64}(
#         # MOIB.LogDet{Float64}(
#         # MOIB.RootDet{Float64}(
#             optimizer
#         ),#))),
#         config, exclude)
# end
end

return nothing
