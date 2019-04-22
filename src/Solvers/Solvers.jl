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
abstract type CombinedHSDSystemSolver end
include("homogeneous_self_dual/solver.jl")
include("homogeneous_self_dual/combined_step/stepper.jl")
include("homogeneous_self_dual/combined_step/naive.jl")
include("homogeneous_self_dual/combined_step/naiveelim.jl")
include("homogeneous_self_dual/combined_step/symindef.jl")
include("homogeneous_self_dual/combined_step/qrchol.jl")
# include("homogeneous_self_dual/combined_step/cholchol.jl")

# TODO sequential quadratic algorithm for linear, quadratic, and smooth convex models

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
function get_x(solver::Solver, model::Models.PreprocessedLinearModel)
    x = zeros(model.n_raw)
    x[model.x_keep_idxs] = get_x(solver) # unpreprocess solver's solution
    return x
end
get_x(solver::Solver, model::Models.Model) = get_x(solver)

get_y(solver::Solver) = copy(solver.point.y)
function get_y(solver::Solver, model::Models.PreprocessedLinearModel)
    y = zeros(model.p_raw)
    y[model.y_keep_idxs] = get_y(solver) # unpreprocess solver's solution
    return y
end
get_y(solver::Solver, model::Models.Model) = get_y(solver)

end
