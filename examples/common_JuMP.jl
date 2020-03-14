#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for JuMP examples
=#

include(joinpath(@__DIR__, "common.jl"))

import JuMP
const MOI = JuMP.MOI

MOI.Utilities.@model(ClassicConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.PositiveSemidefiniteConeTriangle, MOI.ExponentialCone, MOI.DualExponentialCone,),
    (MOI.PowerCone, MOI.DualPowerCone,),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

MOI.Utilities.@model(ExpConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.ExponentialCone,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

MOI.Utilities.@model(SOConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

abstract type ExampleInstanceJuMP{T <: Real} <: ExampleInstance{T} end

function test(
    E::Type{<:ExampleInstanceJuMP{Float64}}, # an instance of a JuMP example # TODO support generic reals
    inst_data::Tuple,
    extender = nothing, # MOI.Utilities-defined optimizer with subset of cones if using extended formulation
    solver_options = (); # additional non-default solver options specific to the example
    default_solver_options = (verbose = false,), # default solver options
    rseed::Int = 1,
    )
    # setup instance and model
    Random.seed!(rseed)
    inst = E(inst_data...)
    build_time = @elapsed model = build(inst)
    model_backend = JuMP.backend(model)

    # solve model
    hyp_opt = Hypatia.Optimizer(; default_solver_options..., solver_options...)
    if isnothing(extender)
        # not using MOI extended formulation
        JuMP.set_optimizer(model, () -> hyp_opt)
        JuMP.optimize!(model)
    else
        # use MOI automated extended formulation
        JuMP.set_optimizer(model, extender{Float64})
        MOI.Utilities.attach_optimizer(model_backend)
        MOI.copy_to(hyp_opt, model_backend.optimizer.model)
        JuMP.set_optimizer(model, () -> hyp_opt)
        MOI.optimize!(hyp_opt)
    end

    # run tests for the example
    test_extra(inst, model)

    result = model_backend.optimizer.model.optimizer.result
    return (extender, build_time, result)
end

# fallback: just check optimal status
function test_extra(inst::ExampleInstanceJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end
