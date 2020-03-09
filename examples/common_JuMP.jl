#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for JuMP examples

a model function returns a tuple of (JuMP model, test helpers)

an instance consists of a tuple of:
(1) args to the example model function
(2) options for the example test function
(3) solver options
=#

using Test
import Random
using LinearAlgebra
import Hypatia
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities
import JuMP
const MOI = JuMP.MOI

function test_JuMP_instance(
    model_function::Function,
    test_function::Function,
    instance_info::Tuple;
    T::Type{<:Real} = Float64,
    default_solver_options::NamedTuple = NamedTuple(),
    rseed::Int = 1,
    )
    # setup model
    Random.seed!(rseed)
    (model, test_helpers) = model_function(instance_info[1]...)

    # solve model
    JuMP.set_optimizer(model, () -> Hypatia.Optimizer(; default_solver_options..., instance_info[3]...))
    JuMP.optimize!(model)

    # run tests for the example
    test_function(model, test_helpers, instance_info[2])

    return model
end
