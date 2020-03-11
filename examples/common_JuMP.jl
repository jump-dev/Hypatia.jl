#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for JuMP examples

a model function returns a tuple of (JuMP model, test helpers)
=#

include(joinpath(@__DIR__, "common.jl"))
import JuMP
const MOI = JuMP.MOI

abstract type ExampleInstanceJuMP{T <: Real} <: ExampleInstance{T} end

function test(
    E::Type{<:ExampleInstanceJuMP{T}}, # an instance of a JuMP example
    inst_data::Tuple,
    extend::Bool = false, # whether to use MOI automatic bridging to a `classic' cone formulation
    solver_options = nothing, # additional non-default solver options specific to the example
    test_options = nothing; # options for the example test function
    default_solver_options = (verbose = true,), # default solver options
    rseed::Int = 1,
    ) where {T <: Real}
    # setup instance and model
    Random.seed!(rseed)
    inst = E(inst_data...)
    model = build(inst)

    # solve model
    hyp_opt = Hypatia.Optimizer(; default_solver_options..., solver_options...)
    if extend
        error("doesn't seem to use EF - fix")
        # use MOI automated extended formulation
        JuMP.set_optimizer(model, ClassicConeOptimizer{Float64})
        model_backend = JuMP.backend(model)
        MOI.Utilities.attach_optimizer(model_backend)
        MOI.copy_to(hyp_opt, model_backend.optimizer.model)
    end
    JuMP.set_optimizer(model, () -> hyp_opt)
    JuMP.optimize!(model)

    # run tests for the example
    test_extra(inst, model, test_options)

    return nothing
end

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
