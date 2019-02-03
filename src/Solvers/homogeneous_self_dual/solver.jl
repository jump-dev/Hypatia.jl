#=
Copyright 2018, Chris Coey and contributors

interior point type and functions for algorithms based on homogeneous self dual embedding

TODO make internal statuses types
=#

mutable struct HSDSolver <: Solver
    model::Models.Linear

    step_cache::HSDStepper

    verbose::Bool
    tol_rel_opt::Float64
    tol_abs_opt::Float64
    tol_feas::Float64
    max_iters::Int
    time_limit::Float64

    x_conv_tol::Float64
    y_conv_tol::Float64
    z_conv_tol::Float64

    point::Models.Point
    tau::Float64
    kap::Float64
    mu::Float64

    x_residual::Vector{Float64}
    y_residual::Vector{Float64}
    z_residual::Vector{Float64}
    x_norm_res_t::Float64
    y_norm_res_t::Float64
    z_norm_res_t::Float64
    x_norm_res::Float64
    y_norm_res::Float64
    z_norm_res::Float64

    obj_primal_t::Float64
    obj_dual_t::Float64
    obj_primal::Float64
    obj_dual::Float64
    gap::Float64
    rel_gap::Float64

    # solve info
    status::Symbol
    num_iters::Int
    solve_time::Float64
    primal_obj::Float64
    dual_obj::Float64

    function HSDSolver(
        model::Models.Linear,
        # step_cache::HSDStepper,
        # point::Models.Point,
        ;
        verbose::Bool = true,
        tol_rel_opt = 1e-6,
        tol_abs_opt = 1e-7,
        tol_feas = 1e-7,
        max_iters::Int = 100,
        time_limit::Float64 = 3e2,
        )
        solver = new()

        solver.model = model

        solver.stepper = CombinedCholCholStepper(model)

        solver.verbose = verbose
        solver.tol_rel_opt = tol_rel_opt
        solver.tol_abs_opt = tol_abs_opt
        solver.tol_feas = tol_feas
        solver.max_iters = max_iters
        solver.time_limit = time_limit

        solver.x_conv_tol = inv(max(1.0, norm(model.c)))
        solver.y_conv_tol = inv(max(1.0, norm(model.b)))
        solver.z_conv_tol = inv(max(1.0, norm(model.h)))

        solver.point = Models.find_initial_point(model)
        solver.tau = 1.0
        solver.kap = 1.0
        solver.mu = NaN

        solver.x_residual = similar(model.c)
        solver.y_residual = similar(model.b)
        solver.z_residual = similar(model.h)
        solver.x_norm_res_t = NaN
        solver.y_norm_res_t = NaN
        solver.z_norm_res_t = NaN
        solver.x_norm_res = NaN
        solver.y_norm_res = NaN
        solver.z_norm_res = NaN

        solver.obj_primal_t = NaN
        solver.obj_dual_t = NaN
        solver.obj_primal = NaN
        solver.obj_dual = NaN
        solver.gap = NaN
        solver.rel_gap = NaN

        solver.status = :SolveNotCalled
        solver.num_iters = 0
        solver.solve_time = NaN
        solver.primal_obj = NaN
        solver.dual_obj = NaN

        return solver
    end
end

get_tau(solver::HSDSolver) = solver.tau
get_kappa(solver::HSDSolver) = solver.kap
get_mu(solver::HSDSolver) = solver.mu

# TODO maybe use iteration interface rather than while loop
function solve(solver::HSDSolver)
    solver.status = :SolveCalled
    start_time = time()

    calc_mu(solver)
    if isnan(solver.mu) || abs(1.0 - solver.mu) > 1e-6
        error("initial mu is $(solver.mu) (should be 1.0)")
    end

    solver.verbose && @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s\n",
        "iter", "p_obj", "d_obj", "abs_gap", "rel_gap", "p_inf", "d_inf", "tau", "kap", "mu")

    while true
        calc_residual(solver)

        if check_convergence(solver)
            break
        end

        if solver.num_iters == solver.max_iters
            solver.verbose && println("iteration limit reached; terminating")
            solver.status = :IterationLimit
            break
        end

        if time() - start_time >= solver.time_limit
            solver.verbose && println("time limit reached; terminating")
            solver.status = :TimeLimit
            break
        end

        # TODO may use different function, or function could change during some iteration eg if numerical difficulties
        combined_predict_correct(solver)

        solver.num_iters += 1
    end

    # calculate result and iteration statistics and finish
    point = solver.point
    point.x ./= solver.tau
    point.y ./= solver.tau
    point.z ./= solver.tau
    point.s ./= solver.tau

    solver.solve_time = time() - start_time

    if solver.verbose
        println("\nstatus is $(solver.status) after $(solver.num_iters) iterations and $(trunc(solver.solve_time, digits=3)) seconds\n")
    end

    return
end

function check_convergence(solver::HSDSolver)
    model = solver.model
    point = solver.point

    norm_res_primal = max(solver.y_norm_res * solver.y_conv_tol, solver.z_norm_res * solver.z_conv_tol)
    norm_res_dual = solver.x_norm_res * solver.x_conv_tol

    solver.obj_primal_t = dot(model.c, point.x)
    solver.obj_dual_t = -dot(model.b, point.y) - dot(model.h, point.z)
    solver.obj_primal = solver.obj_primal_t / solver.tau
    solver.obj_dual = solver.obj_dual_t / solver.tau
    solver.gap = dot(point.z, point.s) # TODO maybe should adapt original Alfonso condition instead of using this CVXOPT condition
    if solver.obj_primal < 0.0
        solver.rel_gap = solver.gap / -solver.obj_primal
    elseif solver.obj_dual > 0.0
        solver.rel_gap = solver.gap / solver.obj_dual
    else
        solver.rel_gap = NaN
    end

    # print iteration statistics
    if solver.verbose
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            solver.num_iters, solver.obj_primal, solver.obj_dual, solver.gap, solver.rel_gap,
            norm_res_primal, norm_res_dual, solver.tau, solver.kap, solver.mu
            )
        flush(stdout)
    end

    # check convergence criteria
    # TODO nearly primal or dual infeasible or nearly optimal cases?
    if norm_res_primal <= solver.tol_feas && norm_res_dual <= solver.tol_feas &&
        (solver.gap <= solver.tol_abs_opt || (!isnan(solver.rel_gap) && solver.rel_gap <= solver.tol_rel_opt))
        solver.verbose && println("optimal solution found; terminating")
        solver.status = :Optimal
        return true
    end
    if solver.obj_dual_t > 0.0
        infres_pr = solver.x_norm_res_t * solver.x_conv_tol / solver.obj_dual_t
        if infres_pr <= solver.tol_feas
            solver.verbose && println("primal infeasibility detected; terminating")
            solver.status = :PrimalInfeasible
            return true
        end
    end
    if solver.obj_primal_t < 0.0
        infres_du = -max(solver.y_norm_res_t * solver.y_conv_tol, solver.z_norm_res_t * solver.z_conv_tol) / solver.obj_primal_t
        if infres_du <= solver.tol_feas
            solver.verbose && println("dual infeasibility detected; terminating")
            solver.status = :DualInfeasible
            return true
        end
    end
    if solver.mu <= solver.tol_feas * 1e-2 && solver.tau <= solver.tol_feas * 1e-2 * min(1.0, solver.kap)
        solver.verbose && println("ill-posedness detected; terminating")
        solver.status = :IllPosed
        return true
    end

    return false
end


function calc_residual(solver::HSDSolver)
    model = solver.model
    point = solver.point

    # x_residual = -A'*y - G'*z - c*tau
    x_residual = solver.x_residual
    x_residual .= -model.A' * point.y - model.G' * point.z # TODO remove allocs
    solver.x_norm_res_t = norm(x_residual)
    @. x_residual -= model.c * solver.tau
    solver.x_norm_res = norm(x_residual) / solver.tau

    # y_residual = A*x - b*tau
    y_residual = solver.y_residual
    mul!(y_residual, model.A, point.x)
    solver.y_norm_res_t = norm(y_residual)
    @. y_residual -= model.b * solver.tau
    solver.y_norm_res = norm(y_residual) / solver.tau

    # z_residual = s + G*x - h*tau
    z_residual = solver.z_residual
    mul!(z_residual, model.G, point.x)
    @. z_residual += point.s
    solver.z_norm_res_t = norm(z_residual)
    @. z_residual -= model.h * solver.tau
    solver.z_norm_res = norm(z_residual) / solver.tau

    return
end

function calc_mu(solver::HSDSolver)
    solver.mu = (dot(solver.point.z, solver.point.s) + solver.tau * solver.kap) /
        (1.0 + solver.model.nu)
    return solver.mu
end



# # get neighborhood parameters depending on magnitude of barrier parameter and maximum number of correction steps
# # TODO calculate values from the formulae given in Papp & Yildiz "On A Homogeneous Interior-Point Algorithm for Non-Symmetric Convex Conic Optimization"
# function getbetaeta(maxcorrsteps::Int, bnu::Float64)
#     if maxcorrsteps <= 2
#         if bnu < 10.0
#             return (0.1810, 0.0733, 0.0225)
#         elseif bnu < 100.0
#             return (0.2054, 0.0806, 0.0263)
#         else
#             return (0.2190, 0.0836, 0.0288)
#         end
#     elseif maxcorrsteps <= 4
#         if bnu < 10.0
#             return (0.2084, 0.0502, 0.0328)
#         elseif bnu < 100.0
#             return (0.2356, 0.0544, 0.0380)
#         else
#             return (0.2506, 0.0558, 0.0411)
#         end
#     else
#         if bnu < 10.0
#             return (0.2387, 0.0305, 0.0429)
#         elseif bnu < 100.0
#             return (0.2683, 0.0327, 0.0489)
#         else
#             return (0.2844, 0.0332, 0.0525)
#         end
#     end
# end
