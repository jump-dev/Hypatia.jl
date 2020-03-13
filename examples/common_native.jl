#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for native examples
=#

include(joinpath(@__DIR__, "common.jl"))

abstract type ExampleInstanceNative{T <: Real} <: ExampleInstance{T} end

function test(
    E::Type{<:ExampleInstanceNative{T}}, # an instance of a native example
    inst_data::Tuple,
    solver_options = (); # additional non-default solver options specific to the example
    default_solver_options = (verbose = false,), # default solver options
    rseed::Int = 1,
    checker_options = (test = false,)
    ) where {T <: Real}
    # setup instance and model
    Random.seed!(rseed)
    inst = E(inst_data...)
    model = build(inst)

    # solve model
    solver = Solvers.Solver{T}(; default_solver_options..., solver_options...)
    result = Solvers.solve_check(model; solver = solver, checker_options...)

    # run tests for the example
    test_extra(inst, result)

    return result
end
