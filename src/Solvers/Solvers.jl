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
import Krylov
using Test
using TimerOutputs
import Hypatia.Cones
import Hypatia.Models
import Hypatia.HypReal
import Hypatia.HypLinMap
import Hypatia.hyp_AtA!
import Hypatia.hyp_chol!
import Hypatia.hyp_ldiv_chol_L!
import Hypatia.HypBlockMatrix

import CSV, DataFrames

abstract type Solver{T <: HypReal} end

# homogeneous self-dual embedding algorithm
abstract type HSDStepper{T <: HypReal} end
abstract type CombinedHSDSystemSolver{T <: HypReal} end
include("homogeneous_self_dual/solver.jl")
include("homogeneous_self_dual/combined_step/stepper.jl")
include("homogeneous_self_dual/combined_step/naive.jl")
include("homogeneous_self_dual/combined_step/naiveelim.jl")
include("homogeneous_self_dual/combined_step/symindef.jl")
include("homogeneous_self_dual/combined_step/qrchol.jl")
# include("homogeneous_self_dual/combined_step/cholchol.jl")

# TODO sequential quadratic algorithm for linear, quadratic, and smooth convex models

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
function get_x(solver::Solver{T}, model::Models.PreprocessedLinearModel{T}) where {T <: HypReal}
    x = zeros(T, length(model.c_raw))
    x[model.x_keep_idxs] = solver.point.x # unpreprocess solver's solution
    return x
end
get_x(solver::Solver{T}, model::Models.Model{T}) where {T <: HypReal} = get_x(solver)

get_y(solver::Solver) = copy(solver.point.y)
function get_y(solver::Solver{T}, model::Models.PreprocessedLinearModel{T}) where {T <: HypReal}
    y = zeros(T, length(model.b_raw))
    y[model.y_keep_idxs] = solver.point.y # unpreprocess solver's solution
    return y
end
get_y(solver::Solver{T}, model::Models.Model{T}) where {T <: HypReal} = get_y(solver)

# check conic certificates are valid
# TODO pick default tols based on T
function get_certificates(
    solver::Solver{T},
    model::Models.LinearModel{T};
    test::Bool = true,
    atol = max(1e-5, sqrt(sqrt(eps(T)))),
    rtol = atol,
    ) where {T <: HypReal}
    status = get_status(solver)
    primal_obj = get_primal_obj(solver)
    dual_obj = get_dual_obj(solver)
    x = get_x(solver, model)
    y = get_y(solver, model)
    s = get_s(solver, model)
    z = get_z(solver, model)

    if test
        (c, A, b, G, h, cones, cone_idxs) = Models.get_original_data(model)
        if status == :Optimal
            @test primal_obj ≈ dual_obj atol=atol rtol=rtol
            @test A * x ≈ b atol=atol rtol=rtol
            @test G * x + s ≈ h atol=atol rtol=rtol
            @test G' * z + A' * y ≈ -c atol=atol rtol=rtol
            @test dot(s, z) ≈ zero(T) atol=atol rtol=rtol
            @test dot(c, x) ≈ primal_obj atol=atol^2 rtol=rtol^2
            @test dot(b, y) + dot(h, z) ≈ -dual_obj atol=atol^2 rtol=rtol^2
        elseif status == :PrimalInfeasible
            # @test isnan(primal_obj)
            @test dual_obj > zero(T)
            @test dot(b, y) + dot(h, z) ≈ -dual_obj atol=atol^2 rtol=rtol^2
            # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
            # @test G' * z ≈ -A' * y atol=atol rtol=rtol
        elseif status == :DualInfeasible
            # @test isnan(dual_obj)
            @test primal_obj < zero(T)
            @test dot(c, x) ≈ primal_obj atol=atol^2 rtol=rtol^2
            # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
            # @test G * x ≈ -s atol=atol rtol=rtol
            # @test A * x ≈ zeros(T, length(y)) atol=atol rtol=rtol
        elseif status == :IllPosed
            # TODO primal vs dual ill-posed statuses and conditions
        end
    end

    return (x = x, y = y, s = s, z = z, primal_obj = primal_obj, dual_obj = dual_obj, status = status)
end

# build linear model, solve, check conic certificates, and return certificate data
function build_solve_check(
    c::Vector{T},
    A::HypLinMap{T},
    b::Vector{T},
    G::HypLinMap{T},
    h::Vector{T},
    cones::Vector{Cones.Cone{T}},
    cone_idxs::Vector{UnitRange{Int}};
    test::Bool = true,
    linear_model::Type{<:Models.LinearModel} = Models.PreprocessedLinearModel,
    system_solver::Type{<:CombinedHSDSystemSolver} = Solvers.QRCholCombinedHSDSystemSolver,
    linear_model_options::NamedTuple = NamedTuple(),
    system_solver_options::NamedTuple = NamedTuple(),
    stepper_options::NamedTuple = NamedTuple(),
    solver_options::NamedTuple = NamedTuple(),
    atol::Real = max(1e-5, sqrt(sqrt(eps(T)))),
    rtol::Real = atol,
    ) where {T <: HypReal}
    model = linear_model{T}(c, A, b, G, h, cones, cone_idxs; linear_model_options...)
    stepper = CombinedHSDStepper{T}(model, system_solver = system_solver{T}(model; system_solver_options...); stepper_options...)
    solver = HSDSolver{T}(model, stepper = stepper; solver_options...)
    solve(solver)
    return get_certificates(solver, model, test = test, atol = atol, rtol = rtol)
end

end
