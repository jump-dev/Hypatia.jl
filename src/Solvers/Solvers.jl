#=
Copyright 2018, Chris Coey and contributors

functions and caches for interior point algorithms
=#

module Solvers

using Printf
using LinearAlgebra
using SparseArrays

import Hypatia.Cones
import Hypatia.Models

abstract type Solver end

# homogeneous self-dual embedding algorithm
abstract type HSDStepper end
include("homogeneous_self_dual/solver.jl")
include("homogeneous_self_dual/steppers.jl")
include("homogeneous_self_dual/combined_step/stepper.jl")
include("homogeneous_self_dual/combined_step/naive.jl")
# include("homogeneous_self_dual/combined_step/cholchol.jl")

# TODO sequential quadratic algorithm for linear and smooth_convex models
# include("sequential_quadratic/solver.jl")

get_status(solver::Solver) = solver.status
get_solve_time(solver::Solver) = solver.solve_time
get_num_iters(solver::Solver) = solver.num_iters

get_x(solver::Solver) = copy(solver.point.x)
get_s(solver::Solver) = copy(solver.point.s)
get_y(solver::Solver) = copy(solver.point.y)
get_z(solver::Solver) = copy(solver.point.z)

get_primal_obj(solver::Solver) = solver.primal_obj
get_dual_obj(solver::Solver) = solver.dual_obj

end
