#=
combined directions stepper
=#

mutable struct HeurCombStepper{T <: Real} <: Stepper{T}
    prev_pred_alpha::T
    prev_alpha::T
    prev_gamma::T

    rhs::Point{T}
    dir::Point{T}
    res::Point{T}
    dir_temp::Vector{T}

    line_searcher::LineSearcher{T}

    HeurCombStepper{T}() where {T <: Real} = new{T}()
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
    stepper.dir_temp = zeros(T, length(stepper.rhs.vec))

    stepper.line_searcher = LineSearcher{T}(model)

    return stepper
end

# original combined stepper
function step(stepper::HeurCombStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    model = solver.model

    # update linear system solver factorization and helpers
    Cones.grad.(model.cones)
    update_lhs(solver.system_solver, solver)

    # calculate centering direction and keep in dir_cent
    update_rhs_cent(solver, stepper.rhs)
    get_directions(stepper, solver, false, iter_ref_steps = 3)
    dir_cent = copy(stepper.dir.vec) # TODO
    update_rhs_centcorr(solver, stepper.rhs, stepper.dir)
    get_directions(stepper, solver, false, iter_ref_steps = 3)
    dir_centcorr = copy(stepper.dir.vec) # TODO
    # copyto!(stepper.dir_cent, stepper.dir)

    # calculate affine/prediction direction and keep in dir
    update_rhs_pred(solver, stepper.rhs)
    get_directions(stepper, solver, true, iter_ref_steps = 3)
    dir_pred = copy(stepper.dir.vec) # TODO
    update_rhs_predcorr(solver, stepper.rhs, stepper.dir)
    get_directions(stepper, solver, true, iter_ref_steps = 3)
    dir_predcorr = copy(stepper.dir.vec) # TODO

    # calculate centering factor gamma by finding distance pred_alpha for stepping in pred direction
    copyto!(stepper.dir.vec, dir_pred)
    stepper.prev_pred_alpha = pred_alpha = find_max_alpha(point, stepper.dir, stepper.line_searcher, model, prev_alpha = stepper.prev_pred_alpha, min_alpha = T(1e-2), max_nbhd = one(T)) # TODO try max_nbhd = Inf, but careful of cones with no dual feas check

    # TODO allow different function (heuristic) as option?
    # stepper.prev_gamma = gamma = abs2(1 - pred_alpha)
    stepper.prev_gamma = gamma = 1 - pred_alpha

    # calculate combined direction and keep in dir
    # axpby!(gamma, stepper.dir_cent, 1 - gamma, stepper.dir)
    @. stepper.dir.vec = gamma * (dir_cent + pred_alpha * dir_centcorr) + (1 - gamma) * (dir_pred + pred_alpha * dir_predcorr) # TODO

    # find distance alpha for stepping in combined direction
    alpha = find_max_alpha(point, stepper.dir, stepper.line_searcher, model, prev_alpha = stepper.prev_alpha, min_alpha = T(1e-3))

    if iszero(alpha)
        # could not step far in combined direction, so attempt a pure centering step
        solver.verbose && println("performing centering step")
        # copyto!(stepper.dir, stepper.dir_cent)
        @. stepper.dir.vec = dir_cent + dir_centcorr

        # find distance alpha for stepping in centering direction
        alpha = find_max_alpha(point, stepper.dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-6))

        if iszero(alpha)
            copyto!(stepper.dir.vec, dir_cent)
            alpha = find_max_alpha(point, stepper.dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-6))

            if iszero(alpha)
                @warn("numerical failure: could not step in centering direction; terminating")
                solver.status = NumericalFailure
                return false
            end
        end
    end
    stepper.prev_alpha = alpha

    # step
    @. point.vec += alpha * stepper.dir.vec
    calc_mu(solver)

    return true
end

function print_iteration_stats(stepper::HeurCombStepper{T}, solver::Solver{T}) where {T <: Real}
    if iszero(solver.num_iters)
        if iszero(solver.model.p)
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
                "iter", "p_obj", "d_obj", "rel_gap", "abs_gap",
                "x_feas", "z_feas", "tau", "kap", "mu",
                "gamma", "alpha",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu
                )
        else
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
                "iter", "p_obj", "d_obj", "rel_gap", "abs_gap",
                "x_feas", "y_feas", "z_feas", "tau", "kap", "mu",
                "gamma", "alpha",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu
                )
        end
    else
        if iszero(solver.model.p)
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu,
                stepper.prev_gamma, stepper.prev_alpha,
                )
        else
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu,
                stepper.prev_gamma, stepper.prev_alpha,
                )
        end
    end
    flush(stdout)
    return
end
