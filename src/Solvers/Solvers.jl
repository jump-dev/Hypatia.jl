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
import Hypatia.HypLUSolveCache
import Hypatia.hyp_lu_solve!
import Hypatia.BlockMatrix

# homogeneous self-dual embedding algorithm
abstract type SystemSolver{T <: Real} end
include("homogeneous_self_dual/solver.jl")
include("homogeneous_self_dual/naive.jl")
include("homogeneous_self_dual/naiveelim.jl")
include("homogeneous_self_dual/symindef.jl")
include("homogeneous_self_dual/qrchol.jl")

# solve, optionally test conic certificates, and return solve information
function solve_check(
    model::Models.Model{T};
    solver::Solver{T} = Solver{T}(),
    test::Bool = true,
    atol::Real = sqrt(sqrt(eps(T))),
    rtol::Real = sqrt(sqrt(eps(T))),
    ) where {T <: Real}
    load(solver, model)
    solve(solver)

    status = get_status(solver)
    primal_obj = get_primal_obj(solver)
    dual_obj = get_dual_obj(solver)
    x = get_x(solver)
    y = get_y(solver)
    s = get_s(solver)
    z = get_z(solver)

    if test
        (c, A, b, G, h, cones, obj_offset) = (model.c, model.A, model.b, model.G, model.h, model.cones, model.obj_offset)
        if status == :Optimal
            @test primal_obj ≈ dual_obj atol=atol rtol=rtol
            @test A * x ≈ b atol=atol rtol=rtol
            @test G * x + s ≈ h atol=atol rtol=rtol
            @test G' * z + A' * y ≈ -c atol=atol rtol=rtol
            @test dot(c, x) + obj_offset ≈ primal_obj atol=atol^2 rtol=rtol^2
            @test -dot(b, y) - dot(h, z) + obj_offset ≈ dual_obj atol=atol^2 rtol=rtol^2
            @test dot(s, z) ≈ zero(T) atol=10atol rtol=10rtol
        elseif status == :PrimalInfeasible
            @test dual_obj > obj_offset
            @test -dot(b, y) - dot(h, z) + obj_offset ≈ dual_obj atol=atol^2 rtol=rtol^2
            # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
            # @test G' * z ≈ -A' * y atol=atol rtol=rtol
        elseif status == :DualInfeasible
            @test primal_obj < obj_offset
            @test dot(c, x) + obj_offset ≈ primal_obj atol=atol^2 rtol=rtol^2
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

# build model and call solve_check
function build_solve_check(
    c::Vector{T},
    A,
    b::Vector{T},
    G,
    h::Vector{T},
    cones::Vector{Cones.Cone{T}};
    obj_offset::T = zero(T),
    other_options...
    ) where {T <: Real}
    model = Models.Model{T}(c, A, b, G, h, cones, obj_offset = obj_offset)
    return solve_check(model; other_options...)
end

end
