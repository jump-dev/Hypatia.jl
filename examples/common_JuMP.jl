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

abstract type ExampleInstanceJuMP{T <: Real} <: ExampleInstance{T} end

function test(
    E::Type{<:ExampleInstanceJuMP{Float64}}, # an instance of a JuMP example # TODO support generic reals
    inst_data::Tuple,
    extend::Bool = false, # whether to use MOI automatic bridging to a `classic' cone formulation
    solver_options = (); # additional non-default solver options specific to the example
    default_solver_options = (verbose = false,), # default solver options
    rseed::Int = 1,
    )
    # setup instance and model
    Random.seed!(rseed)
    inst = E(inst_data...)
    build_time = @elapsed model = build(inst)

    # solve model
    hyp_opt = Hypatia.Optimizer(; default_solver_options..., solver_options...)
    model_backend = JuMP.backend(model)
    if extend
        error("doesn't seem to use EF - fix")
        # use MOI automated extended formulation
        JuMP.set_optimizer(model, ClassicConeOptimizer{Float64})
        MOI.Utilities.attach_optimizer(model_backend)
        MOI.copy_to(hyp_opt, model_backend.optimizer.model)
    end
    JuMP.set_optimizer(model, () -> hyp_opt)
    JuMP.optimize!(model)

    # run tests for the example
    test_extra(inst, model)

    result = model_backend.optimizer.model.optimizer.result
    return (build_time, result)
end

# fallback: just check optimal status
function test_extra(inst::ExampleInstanceJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end
