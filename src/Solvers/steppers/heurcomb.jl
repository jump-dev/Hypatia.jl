#=
combined directions stepper
=#

mutable struct HeurCombStepper{T <: Real} <: Stepper{T}
    gamma_fun::Function
    prev_pred_alpha::T
    prev_alpha::T
    prev_gamma::T

    rhs::Point{T}
    dir::Point{T}
    res::Point{T}
    dir_temp::Vector{T}
    dir_cent::Vector{T}
    dir_centcorr::Vector{T}
    dir_pred::Vector{T}
    dir_predcorr::Vector{T}

    step_searcher::StepSearcher{T}

    function HeurCombStepper{T}(;
        gamma_fun::Function = (a::T -> (1 - a)),
        # gamma_fun::Function = (a -> abs2(1 - a)),
        ) where {T <: Real}
        stepper = new{T}()
        stepper.gamma_fun = gamma_fun
        return stepper
    end
end

# create the stepper cache
function load(stepper::HeurCombStepper{T}, solver::Solver{T}) where {T <: Real}
    stepper.prev_pred_alpha = one(T)
    stepper.prev_gamma = one(T)
    stepper.prev_alpha = one(T)

    model = solver.model
    stepper.rhs = Point(model)
    stepper.dir = Point(model)
    stepper.res = Point(model)
    dim = length(stepper.rhs.vec)
    stepper.dir_temp = zeros(T, dim)
    stepper.dir_cent = zeros(T, dim)
    stepper.dir_centcorr = zeros(T, dim)
    stepper.dir_pred = zeros(T, dim)
    stepper.dir_predcorr = zeros(T, dim)

    stepper.step_searcher = StepSearcher{T}(model)

    return stepper
end

# original combined stepper
function step(stepper::HeurCombStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    model = solver.model
    rhs = stepper.rhs
    dir = stepper.dir
    dir_cent = stepper.dir_cent
    dir_centcorr = stepper.dir_centcorr
    dir_pred = stepper.dir_pred
    dir_predcorr = stepper.dir_predcorr

    # update linear system solver factorization
    update_lhs(solver.system_solver, solver)

    # calculate centering direction and correction
    update_rhs_cent(solver, rhs)
    get_directions(stepper, solver, false, iter_ref_steps = 3)
    copyto!(dir_cent, dir.vec)
    update_rhs_centcorr(solver, rhs, dir)
    get_directions(stepper, solver, false, iter_ref_steps = 3)
    copyto!(dir_centcorr, dir.vec)

    # calculate affine/prediction direction and correction
    update_rhs_pred(solver, rhs)
    get_directions(stepper, solver, true, iter_ref_steps = 3)
    copyto!(dir_pred, dir.vec)
    update_rhs_predcorr(solver, rhs, dir)
    get_directions(stepper, solver, true, iter_ref_steps = 3)
    copyto!(dir_predcorr, dir.vec)

    # calculate centering factor gamma by finding distance pred_alpha for stepping in pred direction
    copyto!(dir.vec, dir_pred)
    # TODO try max_nbhd = Inf, but careful of cones with no dual feas check
    stepper.prev_pred_alpha = pred_alpha = find_max_alpha(point, dir, stepper.step_searcher, model, prev_alpha = stepper.prev_pred_alpha, min_alpha = T(1e-2))
    stepper.prev_gamma = gamma = stepper.gamma_fun(pred_alpha)

    # calculate combined direction and keep in dir
    gamma_alpha = gamma * pred_alpha
    gamma1 = 1 - gamma
    gamma1_alpha = gamma1 * pred_alpha
    @. dir.vec = gamma * dir_cent + gamma_alpha * dir_centcorr + gamma1 * dir_pred + gamma1_alpha * dir_predcorr

    # find distance alpha for stepping in combined direction
    alpha = find_max_alpha(point, dir, stepper.step_searcher, model, prev_alpha = stepper.prev_alpha, min_alpha = T(1e-3))

    if iszero(alpha)
        # could not step far in combined direction, so attempt a pure centering step
        solver.verbose && println("performing centering step")
        @. dir.vec = dir_cent + dir_centcorr

        # find distance alpha for stepping in centering direction
        alpha = find_max_alpha(point, dir, stepper.step_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3))

        if iszero(alpha)
            copyto!(dir.vec, dir_cent)
            alpha = find_max_alpha(point, dir, stepper.step_searcher, model, prev_alpha = one(T), min_alpha = T(1e-6))
            if iszero(alpha)
                @warn("numerical failure: could not step in centering direction; terminating")
                solver.status = NumericalFailure
                return false
            end
        end
    end
    stepper.prev_alpha = alpha

    # step
    @. point.vec += alpha * dir.vec
    calc_mu(solver)

    return true
end

expect_improvement(::HeurCombStepper) = true

function print_iteration_stats(stepper::HeurCombStepper{T}, solver::Solver{T}) where {T <: Real}
    if iszero(solver.num_iters)
        if iszero(solver.model.p)
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s\n",
                "iter", "p_obj", "d_obj", "abs_gap",
                "x_feas", "z_feas", "tau", "kap", "mu",
                "gamma", "alpha",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap,
                solver.x_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu
                )
        else
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
                "iter", "p_obj", "d_obj", "abs_gap",
                "x_feas", "y_feas", "z_feas", "tau", "kap", "mu",
                "gamma", "alpha",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu
                )
        end
    else
        if iszero(solver.model.p)
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap,
                solver.x_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu,
                stepper.prev_gamma, stepper.prev_alpha,
                )
        else
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu,
                stepper.prev_gamma, stepper.prev_alpha,
                )
        end
    end
    flush(stdout)
    return
end
