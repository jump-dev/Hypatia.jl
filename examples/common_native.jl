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
    solver_options = (); # additional non-default solver options specific to the example
    default_solver_options = (), # default solver options
    rseed::Int = 1,
    checker_options = (test = false,)
    ) where {T <: Real}
    # setup instance and model
    Random.seed!(rseed)
    inst = E(inst_data...)
    build_time = @elapsed model = build(inst)

    # solve model
    solver = Solvers.Solver{T}(; default_solver_options..., solver_options...)
    Solvers.load(solver, model)
    Solvers.solve(solver)

    # process the solve info and solution
    result = process_result(model, solver)

    # run tests for the example
    test_extra(inst, result)

    return (nothing, build_time, result)
end

# fallback: just check optimal status
function test_extra(inst::ExampleInstanceNative, result::NamedTuple)
    @test result.status == :Optimal
end

# return solve information and certificate violations
function process_result(
    model::Models.Model{T},
    solver::Solvers.Solver{T},
    ) where {T <: Real}
    status = Solvers.get_status(solver)
    solve_time = Solvers.get_solve_time(solver)
    num_iters = Solvers.get_num_iters(solver)

    primal_obj = Solvers.get_primal_obj(solver)
    dual_obj = Solvers.get_dual_obj(solver)

    x = Solvers.get_x(solver)
    y = Solvers.get_y(solver)
    s = Solvers.get_s(solver)
    z = Solvers.get_z(solver)

    obj_diff = primal_obj - dual_obj
    compl = dot(s, z)

    (c, A, b, G, h, obj_offset) = (model.c, model.A, model.b, model.G, model.h, model.obj_offset)
    if status == :Optimal
        x_res = G' * z + A' * y + c
        y_res = A * x - b
        z_res = G * x + s - h
        x_res_rel = relative_residual(x_res, c)
        y_res_rel = relative_residual(y_res, b)
        z_res_rel = relative_residual(z_res, h)
        x_viol = norm(x_res_rel, Inf)
        y_viol = norm(y_res_rel, Inf)
        z_viol = norm(z_res_rel, Inf)
    elseif status == :PrimalInfeasible
        if dual_obj < obj_offset
            @warn("dual_obj < obj_offset for primal infeasible case")
        end
        # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
        x_res = G' * z + A' * y
        x_res_rel = relative_residual(x_res, c)
        x_viol = norm(x_res_rel, Inf)
        y_viol = NaN
        z_viol = NaN
    elseif status == :DualInfeasible
        if primal_obj > obj_offset
            @warn("primal_obj > obj_offset for primal infeasible case")
        end
        # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
        y_res = A * x
        z_res = G * x + s
        y_res_rel = relative_residual(y_res, b)
        z_res_rel = relative_residual(z_res, h)
        x_viol = NaN
        y_viol = norm(y_res_rel, Inf)
        z_viol = norm(z_res_rel, Inf)
    elseif status == :IllPosed
        # TODO primal vs dual ill-posed statuses and conditions
    end

    return (status = status,
        solve_time = solve_time, num_iters = num_iters,
        primal_obj = primal_obj, dual_obj = dual_obj,
        x = x, y = y, s = s, z = z,
        obj_diff = obj_diff, compl = compl,
        x_viol = x_viol, y_viol = y_viol, z_viol = z_viol)
end
