#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for native examples

a model function returns a tuple of ((c, A, b, G, h, cones, ...), test helpers)

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

function test_native_instance(
    model_function::Function,
    test_function::Function,
    instance_info::Tuple;
    T::Type{<:Real} = Float64,
    solver_options::NamedTuple = NamedTuple(),
    rseed::Int = 1,
    )
    # setup model
    Random.seed!(rseed)
    (model, test_helpers) = model_function(T, instance_info[1]...)

    # solve model
    result = Hypatia.Solvers.build_solve_check(model...; solver_options..., instance_info[2]...)

    # run tests for the example
    test_function(result, test_helpers, instance_info[3])

    return result
end
