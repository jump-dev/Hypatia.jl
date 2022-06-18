#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
common code for examples
=#

abstract type ExampleInstance{T <: Real} end

# this is a workaround for randn's lack of support for BigFloat
Random.randn(R::Type{BigFloat}, dims::Vararg{Int, N} where {N}) = R.(randn(dims...))
function Random.randn(R::Type{Complex{BigFloat}}, dims::Vararg{Int, N} where {N})
    return R.(randn(ComplexF64, dims...))
end

# helper for calculating solution violations
function relative_residual(residual::Vector{T}, constant::Vector{T}) where {T <: Real}
    return norm(residual, Inf) / (1 + norm(constant, Inf))
end

# calculate violations for Hypatia certificate equalities
function certificate_violations(
    status::Solvers.Status,
    primal_obj::T,
    dual_obj::T,
    model::Models.Model{T},
    x::Vector{T},
    y::Vector{T},
    z::Vector{T},
    s::Vector{T},
) where {T <: Real}
    (c, A, b, G, h, obj_offset) =
        (model.c, model.A, model.b, model.G, model.h, model.obj_offset)

    if status == Solvers.PrimalInfeasible
        x_res = G' * z + A' * y
        x_viol = relative_residual(x_res, c)
        y_viol = T(NaN)
        z_viol = T(NaN)
    elseif status == Solvers.DualInfeasible
        y_res = A * x
        z_res = G * x + s
        y_viol = relative_residual(y_res, b)
        z_viol = relative_residual(z_res, h)
        x_viol = T(NaN)
        # TODO elseif status == Solvers.IllPosed (primal vs dual ill-posed)
    else
        if status != Solvers.Optimal
            println(
                "status $status not handled, but computing optimality " *
                "certificate violations anyway",
            )
        end
        x_res = G' * z + A' * y + c
        y_res = A * x - b
        z_res = G * x + s - h
        x_viol = relative_residual(x_res, c)
        y_viol = relative_residual(y_res, b)
        z_viol = relative_residual(z_res, h)
    end

    compl = dot(s, z)
    rel_obj_diff = (primal_obj - dual_obj) / (1 + abs(dual_obj))

    return (x_viol, y_viol, z_viol, compl, rel_obj_diff)
end

# return solve information and certificate violations
function process_result(model::Models.Model{T}, solver::Solvers.Solver{T}) where {T <: Real}
    status = Solvers.get_status(solver)
    solve_time = Solvers.get_solve_time(solver)
    iters = Solvers.get_num_iters(solver)

    primal_obj = Solvers.get_primal_obj(solver)
    dual_obj = Solvers.get_dual_obj(solver)
    z = Solvers.get_z(solver)
    s = Solvers.get_s(solver)
    x = Solvers.get_x(solver)
    y = Solvers.get_y(solver)
    (x_viol, y_viol, z_viol, compl, rel_obj_diff) =
        certificate_violations(status, primal_obj, dual_obj, model, x, y, z, s)

    flush(stdout)
    flush(stderr)

    solve_stats = (;
        status,
        solve_time,
        iters,
        primal_obj,
        dual_obj,
        rel_obj_diff,
        compl,
        x_viol,
        y_viol,
        z_viol,
        :time_rescale => solver.time_rescale,
        :time_initx => solver.time_initx,
        :time_inity => solver.time_inity,
        :time_loadsys => solver.time_loadsys,
        :time_upsys => solver.time_upsys,
        :time_upfact => solver.time_upfact,
        :time_uprhs => solver.time_uprhs,
        :time_getdir => solver.time_getdir,
        :time_search => solver.time_search,
    )
    solution = (; x, y, z, s)

    return (solve_stats, solution)
end

function get_model_stats(model::Models.Model)
    return (;
        :n => model.n,
        :p => model.p,
        :q => model.q,
        :nu => model.nu,
        :cone_types => [string(nameof(c)) for c in unique(typeof.(model.cones))],
        :num_cones => length(model.cones),
        :max_q => maximum(Cones.dimension(c) for c in model.cones),
    )
end
