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
import Hypatia.Models
import Hypatia.Solvers
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

solver_for_model(::Models.Model{T}) where {T <: Real} = Solvers.Solver{T}

function test_native_instance(
    model_function::Function,
    test_function::Function,
    instance_info::Tuple;
    rseed::Int = 1,
    default_solver_options::NamedTuple = (verbose = false,),
    test::Bool = false,
    checker_tols...
    )
    # setup model
    Random.seed!(rseed)
    (model, test_helpers) = model_function(instance_info[1]...)

    # solve model
    solver = solver_for_model(model)(; default_solver_options..., instance_info[3]...)
    result = Solvers.solve_check(model; solver = solver, test = test, checker_tols...)

    # run tests for the example
    test_function(result, test_helpers, instance_info[2])

    return result
end
