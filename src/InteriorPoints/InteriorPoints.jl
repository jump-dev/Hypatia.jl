#=
Copyright 2018, Chris Coey and contributors

functions and caches for interior point algorithms
=#

module InteriorPoints

using Printf
using LinearAlgebra
using SparseArrays
import Hypatia.Cones
import Hypatia.Models

abstract type IPMSolver end

abstract type InteriorPoint end

include("homogselfdual.jl")

get_status(solver::IPMSolver) = solver.status
get_solve_time(solver::IPMSolver) = solver.solve_time
get_num_iters(solver::IPMSolver) = solver.num_iters

get_x(solver::IPMSolver) = copy(solver.point.tx)
get_s(solver::IPMSolver) = copy(solver.point.ts)
get_y(solver::IPMSolver) = copy(solver.point.ty)
get_z(solver::IPMSolver) = copy(solver.point.tz)

get_primal_obj(solver::IPMSolver) = dot(solver.model.c, solver.point.tx)
get_dual_obj(solver::IPMSolver) = -dot(solver.model.b, solver.point.ty) - dot(solver.model.h, solver.point.tz)

end
