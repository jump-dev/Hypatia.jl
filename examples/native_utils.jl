#=
common code for native examples
=#

abstract type ExampleInstanceNative{T <: Real} <: ExampleInstance{T} end

# fallback: just check optimal status
function test_extra(
    inst::ExampleInstanceNative,
    solve_stats::NamedTuple,
    ::NamedTuple,
    )
    @test solve_stats.status == Solvers.Optimal
end

function run_instance(
    ex_type::Type{<:ExampleInstanceNative{T}}, # an instance of a native example
    inst_data::Tuple,
    inst_options::NamedTuple = NamedTuple(),
    solver_type::Type{<:Solvers.Solver} = Solvers.Solver{T};
    default_options::NamedTuple = NamedTuple(),
    test::Bool = true,
    rseed::Int = 1,
    verbose::Bool = true,
    ) where {T <: Real}
    new_options = merge(default_options, inst_options)

    verbose && println("setup model")
    setup_time = @elapsed begin
        Random.seed!(rseed)
        inst = ex_type(inst_data...)
        model = build(inst)
        model_stats = get_model_stats(model)
        solver = Solvers.Solver{T}(; default_options..., inst_options...)
    end
    flush(stdout); flush(stderr)

    verbose && println("solve and check")
    check_time = @elapsed begin
        Solvers.load(solver, model)
        Solvers.solve(solver)
        flush(stdout); flush(stderr)

        (solve_stats, solution) = process_result(model, solver)

        test && test_extra(inst, solve_stats, solution)
    end
    flush(stdout); flush(stderr)

    return (; model_stats..., solve_stats..., setup_time,
        check_time, :script_status => "Success")
end
