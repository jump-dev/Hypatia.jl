#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for native examples
=#

include(joinpath(@__DIR__, "common.jl"))

abstract type ExampleInstanceNative{T <: Real} <: ExampleInstance{T} end

# run a native instance with a given solver and return solve info
function test(
    E::Type{<:ExampleInstanceNative{T}}, # an instance of a native example
    inst_data::Tuple,
    solver_options = (), # additional non-default solver options specific to the example
    solver::Type{<:Hypatia.Optimizer} = Hypatia.Optimizer;
    default_solver_options = (), # default solver options
    rseed::Int = 1,
    ) where {T <: Real}
    # setup instance and model
    Random.seed!(rseed)
    inst = E(inst_data...)
    build_time = @elapsed model = build(inst)

    # solve model
    solver = Solvers.Solver{T}(; default_solver_options..., solver_options...)
    Solvers.load(solver, model)
    Solvers.solve(solver)
    flush(stdout); flush(stderr)

    # process the solve info and solution
    result = process_result(model, solver)
    flush(stdout); flush(stderr)

    # run tests for the example
    test_extra(inst, result)
    flush(stdout); flush(stderr)

    return (nothing, build_time, result)
end

# fallback: just check optimal status
function test_extra(inst::ExampleInstanceNative, result::NamedTuple)
    @test result.status == :Optimal
end
