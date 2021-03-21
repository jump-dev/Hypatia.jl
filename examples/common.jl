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
relative_residual(residual::Vector{T}, constant::Vector{T}) where {T <: Real} = norm(residual, Inf) / (1 + norm(constant, Inf))

# calculate violations for Hypatia certificate equalities
function certificate_violations(
    status::Solvers.Status,
    model::Models.Model{T},
    x::Vector{T},
    y::Vector{T},
    z::Vector{T},
    s::Vector{T},
    ) where {T <: Real}
    (c, A, b, G, h, obj_offset) = (model.c, model.A, model.b, model.G, model.h, model.obj_offset)

    if status == Solvers.PrimalInfeasible
        # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
        x_res = G' * z + A' * y
        x_viol = relative_residual(x_res, c)
        y_viol = T(NaN)
        z_viol = T(NaN)
    elseif status == Solvers.DualInfeasible
        # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
        y_res = A * x
        z_res = G * x + s
        y_viol = relative_residual(y_res, b)
        z_viol = relative_residual(z_res, h)
        x_viol = T(NaN)
    # TODO elseif status == Solvers.IllPosed # primal vs dual ill-posed statuses and conditions
    else
        if status != Solvers.Optimal
            println("status $status not handled, but computing optimality certificate violations anyway")
        end
        x_res = G' * z + A' * y + c
        y_res = A * x - b
        z_res = G * x + s - h
        x_viol = relative_residual(x_res, c)
        y_viol = relative_residual(y_res, b)
        z_viol = relative_residual(z_res, h)
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
    rel_obj_diff = (primal_obj - dual_obj) / (1 + abs(dual_obj))

    z = Solvers.get_z(solver)
    s = Solvers.get_s(solver)
    x = Solvers.get_x(solver)
    y = Solvers.get_y(solver)
    compl = dot(s, z)
    (x_viol, y_viol, z_viol) = certificate_violations(status, model, x, y, z, s)

    time_rescale = solver.time_rescale
    time_initx = solver.time_initx
    time_inity = solver.time_inity
    time_unproc = solver.time_unproc
    time_loadsys = solver.time_loadsys
    time_upsys = solver.time_upsys
    time_upfact = solver.time_upfact
    time_uprhs = solver.time_uprhs
    time_getdir = solver.time_getdir
    time_search = solver.time_search

    solve_stats = (status, solve_time, num_iters, primal_obj, dual_obj, rel_obj_diff, compl, x_viol, y_viol, z_viol,
        time_rescale, time_initx, time_inity, time_unproc, time_loadsys, time_upsys, time_upfact, time_uprhs, time_getdir, time_search,
        x, y, z, s)
    flush(stdout); flush(stderr)
    return solve_stats
end

function get_model_stats(model::Models.Model)
    string_cones = [string(nameof(c)) for c in unique(typeof.(model.cones))]
    num_cones = length(model.cones)
    maxq = maximum(Cones.dimension(c) for c in model.cones)
    # string_cones = [string(nameof(c)) for c in typeof.(model.cones)]
    return (model.n, model.p, model.q, model.nu, string_cones, num_cones, maxq)
end

use_curve_search(::Solvers.Stepper) = true
use_curve_search(stepper::Solvers.PredOrCentStepper) = stepper.use_curve_search
use_correction(::Solvers.Stepper) = true
use_correction(stepper::Solvers.PredOrCentStepper) = stepper.use_correction
shift(::Solvers.Stepper) = 0
shift(stepper::Solvers.CombinedStepper) = stepper.shift_alpha_sched
