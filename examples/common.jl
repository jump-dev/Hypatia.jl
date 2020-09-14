#=
common code for examples
=#

using Test
import Random
using LinearAlgebra
import LinearAlgebra.BlasReal

import Hypatia
import Hypatia.ModelUtilities
import Hypatia.Cones
import Hypatia.Models
import Hypatia.Solvers

abstract type ExampleInstance{T <: Real} end

# NOTE this is a workaround for randn's lack of support for BigFloat
Random.randn(R::Type{BigFloat}, dims::Vararg{Int, N} where N) = R.(randn(dims...))
Random.randn(R::Type{Complex{BigFloat}}, dims::Vararg{Int, N} where N) = R.(randn(ComplexF64, dims...))

# helper for calculating solution violations
function relative_residual(residual::Vector{T}, constant::Vector{T}) where {T <: Real}
    @assert length(residual) == length(constant)
    return T[residual[i] / max(one(T), abs(constant[i])) for i in eachindex(constant)]
end

# calculate violations for Hypatia certificate equalities
function certificate_violations(
    status::Symbol,
    model::Models.Model{T},
    x::Vector{T},
    y::Vector{T},
    z::Vector{T},
    s::Vector{T},
    ) where {T <: Real}
    (c, A, b, G, h, obj_offset) = (model.c, model.A, model.b, model.G, model.h, model.obj_offset)

    if status == Solvers.Optimal
        x_res = G' * z + A' * y + c
        y_res = A * x - b
        z_res = G * x + s - h
        x_res_rel = relative_residual(x_res, c)
        y_res_rel = relative_residual(y_res, b)
        z_res_rel = relative_residual(z_res, h)
        x_viol = norm(x_res_rel, Inf)
        y_viol = norm(y_res_rel, Inf)
        z_viol = norm(z_res_rel, Inf)
    elseif status == Solvers.PrimalInfeasible
        # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
        x_res = G' * z + A' * y
        x_res_rel = relative_residual(x_res, c)
        x_viol = norm(x_res_rel, Inf)
        y_viol = NaN
        z_viol = NaN
    elseif status == Solvers.DualInfeasible
        # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
        y_res = A * x
        z_res = G * x + s
        y_res_rel = relative_residual(y_res, b)
        z_res_rel = relative_residual(z_res, h)
        x_viol = NaN
        y_viol = norm(y_res_rel, Inf)
        z_viol = norm(z_res_rel, Inf)
    # TODO elseif status == Solvers.IllPosed # primal vs dual ill-posed statuses and conditions
    else # failure
        x_viol = NaN
        y_viol = NaN
        z_viol = NaN
    end

    return (x_viol, y_viol, z_viol)
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
    obj_diff = primal_obj - dual_obj

    z = Solvers.get_z(solver)
    s = Solvers.get_s(solver)
    x = Solvers.get_x(solver)
    y = Solvers.get_y(solver)
    compl = dot(s, z)
    (x_viol, y_viol, z_viol) = certificate_violations(status, model, x, y, z, s)

    solve_stats = (status, solve_time, num_iters, primal_obj, dual_obj, obj_diff, compl, x_viol, y_viol, z_viol, x, y, z, s)
    flush(stdout); flush(stderr)
    return solve_stats
end
