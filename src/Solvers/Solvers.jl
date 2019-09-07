#=
Copyright 2018, Chris Coey and contributors

functions and caches for interior point algorithms
=#

module Solvers

using Printf
using LinearAlgebra
import LinearAlgebra.BlasReal
using SparseArrays
import SuiteSparse
import IterativeSolvers
using Test
using TimerOutputs
import Hypatia.Cones
import Hypatia.Models
import Hypatia.HypCholSolveCache
import Hypatia.hyp_chol_solve!
import Hypatia.HypBKSolveCache
import Hypatia.hyp_bk_solve!
import Hypatia.BlockMatrix

abstract type Solver{T <: Real} end

# homogeneous self-dual embedding algorithm
abstract type HSDSystemSolver{T <: Real} end
include("homogeneous_self_dual/solver.jl")
include("homogeneous_self_dual/naive.jl")
include("homogeneous_self_dual/naiveelim.jl")
include("homogeneous_self_dual/symindef.jl")
include("homogeneous_self_dual/qrchol.jl")

get_timer(solver::Solver) = solver.timer

get_status(solver::Solver) = solver.status
get_solve_time(solver::Solver) = solver.solve_time
get_num_iters(solver::Solver) = solver.num_iters

get_primal_obj(solver::Solver) = solver.primal_obj
get_dual_obj(solver::Solver) = solver.dual_obj

get_s(solver::Solver) = copy(solver.point.s)
get_s(solver::Solver, model::Models.Model) = get_s(solver)
get_z(solver::Solver) = copy(solver.point.z)
get_z(solver::Solver, model::Models.Model) = get_z(solver)

get_x(solver::Solver) = copy(solver.point.x)
function get_x(solver::Solver{T}, model::Models.PreprocessedLinearModel{T}) where {T <: Real}
    x = zeros(T, length(model.c_raw))
    x[model.x_keep_idxs] = solver.point.x # unpreprocess solver's solution
    return x
end
get_x(solver::Solver{T}, model::Models.Model{T}) where {T <: Real} = get_x(solver)

get_y(solver::Solver) = copy(solver.point.y)
function get_y(solver::Solver{T}, model::Models.PreprocessedLinearModel{T}) where {T <: Real}
    y = zeros(T, length(model.b_raw))
    y[model.y_keep_idxs] = solver.point.y # unpreprocess solver's solution
    return y
end
get_y(solver::Solver{T}, model::Models.Model{T}) where {T <: Real} = get_y(solver)

# build linear model, solve, optionally test conic certificates, and return certificates and solver and model information
function build_solve_check(
    c::Vector{T},
    A,
    b::Vector{T},
    G,
    h::Vector{T},
    cones::Vector{Cones.Cone{T}};
    obj_offset::T = zero(T),
    linear_model::Type{<:Models.LinearModel} = Models.PreprocessedLinearModel,
    linear_model_options::NamedTuple = NamedTuple(),
    solver::HSDSolver{T} = HSDSolver{T}(),
    test::Bool = true,
    atol::Real = sqrt(sqrt(eps(T))),
    rtol::Real = atol,
    ) where {T <: Real}
    model = linear_model{T}(c, A, b, G, h, cones; obj_offset = obj_offset, linear_model_options...)
    load(solver, model)
    solve(solver)

    status = get_status(solver)
    primal_obj = get_primal_obj(solver)
    dual_obj = get_dual_obj(solver)
    x = get_x(solver, model)
    y = get_y(solver, model)
    s = get_s(solver, model)
    z = get_z(solver, model)

    if test
        (c, A, b, G, h, cones) = Models.get_original_data(model)
        if status == :Optimal
            @test primal_obj ≈ dual_obj atol=atol rtol=rtol
            @test A * x ≈ b atol=atol rtol=rtol
            @test G * x + s ≈ h atol=atol rtol=rtol
            @test G' * z + A' * y ≈ -c atol=atol rtol=rtol
            @test dot(c, x) + model.obj_offset ≈ primal_obj atol=atol^2 rtol=rtol^2
            @test -dot(b, y) - dot(h, z) + model.obj_offset ≈ dual_obj atol=atol^2 rtol=rtol^2
            @test dot(s, z) ≈ zero(T) atol=10atol rtol=10rtol
        elseif status == :PrimalInfeasible
            @test dual_obj > model.obj_offset
            @test -dot(b, y) - dot(h, z) + model.obj_offset ≈ dual_obj atol=atol^2 rtol=rtol^2
            # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
            # @test G' * z ≈ -A' * y atol=atol rtol=rtol
        elseif status == :DualInfeasible
            @test primal_obj < model.obj_offset
            @test dot(c, x) + model.obj_offset ≈ primal_obj atol=atol^2 rtol=rtol^2
            # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
            # @test G * x ≈ -s atol=atol rtol=rtol
            # @test A * x ≈ zeros(T, length(y)) atol=atol rtol=rtol
        elseif status == :IllPosed
            # TODO primal vs dual ill-posed statuses and conditions
        end
    end

    solve_time = get_solve_time(solver)

    return (solver = solver, model = model, solve_time = solve_time, primal_obj = primal_obj, dual_obj = dual_obj, status = status, x = x, y = y, s = s, z = z)
end

end
