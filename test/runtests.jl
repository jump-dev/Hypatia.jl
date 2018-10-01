
using Alfonso
using Test


# native interface tests

verbflag = false # Alfonso verbose option

# TODO interpolation tests

# load optimizer builder functions from examples folder
egs_dir = joinpath(@__DIR__, "../examples")
include(joinpath(egs_dir, "envelope/envelope.jl"))
include(joinpath(egs_dir, "lp/lp.jl"))
include(joinpath(egs_dir, "namedpoly/namedpoly.jl"))

# run native and MOI interfaces on examples
include(joinpath(@__DIR__, "nativeexamples.jl"))


# MathOptInterface tests

import MathOptInterface
MOI = MathOptInterface
MOIT = MOI.Test
MOIB = MOI.Bridges
MOIU = MOI.Utilities

MOIU.@model(AlfonsoModelData,
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

optimizer = MOIU.CachingOptimizer(AlfonsoModelData{Float64}(), Alfonso.Optimizer())

config = MOIT.TestConfig(
    atol=1e-4,
    rtol=1e-4,
    solve=true,
    query=true,
    modify_lhs=true,
    duals=true,
    infeas_certificates=true,
    )

@testset "Continuous linear problems" begin
    MOIT.contlineartest(
        MOIB.SplitInterval{Float64}(
            optimizer
        ),
        config)
end

@testset "Continuous conic problems" begin
    exclude = ["rootdet", "logdet", "sdp"] # TODO bridges not working? should not need to exclude in future
    MOIT.contconictest(
        # MOIB.SquarePSD{Float64}(
        MOIB.GeoMean{Float64}(
        # MOIB.LogDet{Float64}(
        # MOIB.RootDet{Float64}(
            optimizer
        ),#))),
        config, exclude)
end


return nothing
