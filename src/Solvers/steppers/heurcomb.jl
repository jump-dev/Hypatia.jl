#=
combined predict and center stepper
=#

mutable struct HeurCombStepper{T <: Real} <: Stepper{T}
    use_correction::Bool # TODO
    prev_pred_alpha::T
    prev_alpha::T
    prev_gamma::T
    rhs::Point{T}
    dir::Point{T}
    res::Point{T} # TODO rename if used for cand and res
    dir_cent::Point{T}
    dir_centcorr::Point{T}
    dir_pred::Point{T}
    dir_predcorr::Point{T}
    dir_temp::Vector{T}
    step_searcher::StepSearcher{T}

    function HeurCombStepper{T}(;
        use_correction::Bool = true,
        ) where {T <: Real}
        stepper = new{T}()
        stepper.use_correction = use_correction
        return stepper
    end
end

# create the stepper cache
function load(stepper::HeurCombStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    if stepper.use_correction && !any(Cones.use_correction, model.cones)
        # model has no cones that use correction
        stepper.use_correction = false
    end

    stepper.prev_pred_alpha = one(T)
    stepper.prev_gamma = one(T)
    stepper.prev_alpha = one(T)
    stepper.rhs = Point(model)
    stepper.dir = Point(model)
    stepper.res = Point(model)
    stepper.dir_cent = Point(model, cones_only = true)
    stepper.dir_pred = Point(model, cones_only = true)
    if stepper.use_correction
        stepper.dir_centcorr = Point(model, cones_only = true)
        stepper.dir_predcorr = Point(model, cones_only = true)
    end
    stepper.dir_temp = zeros(T, length(stepper.rhs.vec))
    stepper.step_searcher = StepSearcher{T}(model)

    return stepper
end


# TODO make correction optional
# TODO try using single curve search

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
    copyto!(dir_cent.vec, dir.vec)
    update_rhs_centcorr(solver, rhs, dir)
    get_directions(stepper, solver, false, iter_ref_steps = 3)
    copyto!(dir_centcorr.vec, dir.vec)

    # calculate affine/prediction direction and correction
    update_rhs_pred(solver, rhs)
    get_directions(stepper, solver, true, iter_ref_steps = 3)
    copyto!(dir_pred.vec, dir.vec)
    update_rhs_predcorr(solver, rhs, dir)
    get_directions(stepper, solver, true, iter_ref_steps = 3)
    copyto!(dir_predcorr.vec, dir.vec)

    alpha = zero(T)
    if stepper.use_correction
        # TODO move corr here

        # do curve search with correction
        alpha = search_alpha(true, point, model, stepper, min_alpha = T(1e-2)) # TODO tune min_alpha
        if iszero(alpha)
            # try not using correction
            @warn("very small alpha in curve search; trying without correction")
        else
            # step
            gamma = (1 - alpha)
            @. point.vec += alpha * dir_pred.vec + abs2(alpha) * dir_predcorr.vec + gamma * dir_cent.vec + abs2(gamma) * dir_centcorr.vec
        end
    end

    if iszero(alpha)
        # do line search in uncorrected centering direction
        alpha = search_alpha(false, point, model, stepper)
        if iszero(alpha)
            @warn("very small alpha in line search; terminating")
            solver.status = NumericalFailure
            return false
        end
        # step
        @. point.vec += alpha * dir_centcorr.vec

        # calculate centering factor gamma by finding distance pred_alpha for stepping in pred direction
        # # TODO try max_nbhd = Inf, but careful of cones with no dual feas check
        # stepper.prev_pred_alpha = pred_alpha = search_alpha_line(point, dir, stepper.step_searcher, model, prev_alpha = stepper.prev_pred_alpha, min_alpha = T(1e-2))
        #
        # # TODO tune function for gamma
        # gamma = 1 - pred_alpha
        # # gamma = abs2(1 - pred_alpha)
        # stepper.prev_gamma = gamma
        #
        # # calculate combined direction and keep in dir
        # gamma_alpha = gamma * pred_alpha
        # gamma1 = 1 - gamma
        # gamma1_alpha = gamma1 * pred_alpha
        # @. dir.vec = gamma * dir_cent + gamma_alpha * dir_centcorr + gamma1 * dir_pred + gamma1_alpha * dir_predcorr
        #
        # # find distance alpha for stepping in combined direction
        # alpha = search_alpha_line(point, dir, stepper.step_searcher, model, prev_alpha = stepper.prev_alpha, min_alpha = T(1e-3))
        #
        # if iszero(alpha)
        #     # could not step far in combined direction, so attempt a pure centering step
        #     solver.verbose && println("performing centering step")
        #     @. dir.vec = dir_cent + dir_centcorr
        #
        #     # find distance alpha for stepping in centering direction
        #     alpha = search_alpha_line(point, dir, stepper.step_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3))
        #
        #     if iszero(alpha)
        #         copyto!(dir.vec, dir_cent)
        #         alpha = search_alpha_line(point, dir, stepper.step_searcher, model, prev_alpha = one(T), min_alpha = T(1e-6))
        #         if iszero(alpha)
        #             @warn("numerical failure: could not step in centering direction; terminating")
        #             solver.status = NumericalFailure
        #             return false
        #         end
        #     end
        # end
        # stepper.prev_alpha = alpha
        #
        # # step
        # @. point.vec += alpha * dir.vec
    end

    stepper.prev_alpha = alpha
    calc_mu(solver)


    # # calculate centering factor gamma by finding distance pred_alpha for stepping in pred direction
    # copyto!(dir.vec, dir_pred)
    # # TODO try max_nbhd = Inf, but careful of cones with no dual feas check
    # stepper.prev_pred_alpha = pred_alpha = search_alpha_line(point, dir, stepper.step_searcher, model, prev_alpha = stepper.prev_pred_alpha, min_alpha = T(1e-2))
    #
    # # TODO tune function for gamma
    # gamma = 1 - pred_alpha
    # # gamma = abs2(1 - pred_alpha)
    # stepper.prev_gamma = gamma
    #
    # # calculate combined direction and keep in dir
    # gamma_alpha = gamma * pred_alpha
    # gamma1 = 1 - gamma
    # gamma1_alpha = gamma1 * pred_alpha
    # @. dir.vec = gamma * dir_cent + gamma_alpha * dir_centcorr + gamma1 * dir_pred + gamma1_alpha * dir_predcorr
    #
    # # find distance alpha for stepping in combined direction
    # alpha = search_alpha_line(point, dir, stepper.step_searcher, model, prev_alpha = stepper.prev_alpha, min_alpha = T(1e-3))
    #
    # if iszero(alpha)
    #     # could not step far in combined direction, so attempt a pure centering step
    #     solver.verbose && println("performing centering step")
    #     @. dir.vec = dir_cent + dir_centcorr
    #
    #     # find distance alpha for stepping in centering direction
    #     alpha = search_alpha_line(point, dir, stepper.step_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3))
    #
    #     if iszero(alpha)
    #         copyto!(dir.vec, dir_cent)
    #         alpha = search_alpha_line(point, dir, stepper.step_searcher, model, prev_alpha = one(T), min_alpha = T(1e-6))
    #         if iszero(alpha)
    #             @warn("numerical failure: could not step in centering direction; terminating")
    #             solver.status = NumericalFailure
    #             return false
    #         end
    #     end
    # end
    # stepper.prev_alpha = alpha
    #
    # # step
    # @. point.vec += alpha * dir.vec
    # calc_mu(solver)

    return true
end

expect_improvement(::HeurCombStepper) = true

function update_cone_points(
    add_correction::Bool,
    alpha::T,
    point::Point{T},
    stepper::HeurCombStepper{T}
    ) where {T <: Real}
    cand = stepper.res # TODO rename
    dir_cent = stepper.dir_cent
    dir_centcorr = stepper.dir_centcorr
    dir_pred = stepper.dir_pred
    dir_predcorr = stepper.dir_predcorr

    # TODO check tau, kap, then update in one line entire z,t,s,k with a view in point

    # TODO not all
    if add_correction
        gamma = (1 - alpha)
        @. cand.vec = point.vec + alpha * dir_pred.vec + abs2(alpha) * dir_predcorr.vec + gamma * dir_cent.vec + abs2(gamma) * dir_centcorr.vec
    else
        # TODO
        # maybe just center only
        @. cand.vec = point.vec + alpha * dir_cent.vec
    end
    tau_c = cand.tau[1]
    kap_c = cand.kap[1]
    (min(tau_c, kap_c, tau_c * kap_c) < eps(T)) && return false

    # tau_c = cand.tau[1] = point.tau[1] + alpha * dir_nocorr.tau[1] + alpha_sqr * dir_corr.tau[1]
    # kap_c = cand.kap[1] = point.kap[1] + alpha * dir_nocorr.kap[1] + alpha_sqr * dir_corr.kap[1]
    # (min(tau_c, kap_c, tau_c * kap_c) < eps(T)) && return false
    #
    # @. cand.z = point.z + alpha * dir_nocorr.z + alpha_sqr * dir_corr.z
    # @. cand.s = point.s + alpha * dir_nocorr.s + alpha_sqr * dir_corr.s



    # TODO
    # if add_correction
        # dir_corr = stepper.dir_corr
        # alpha_sqr = abs2(alpha)
        #
        # tau_c = cand.tau[1] = point.tau[1] + alpha * dir_nocorr.tau[1] + alpha_sqr * dir_corr.tau[1]
        # kap_c = cand.kap[1] = point.kap[1] + alpha * dir_nocorr.kap[1] + alpha_sqr * dir_corr.kap[1]
        # (min(tau_c, kap_c, tau_c * kap_c) < eps(T)) && return false
        #
        # @. cand.z = point.z + alpha * dir_nocorr.z + alpha_sqr * dir_corr.z
        # @. cand.s = point.s + alpha * dir_nocorr.s + alpha_sqr * dir_corr.s
    # else
    #     tau_c = cand.tau[1] = point.tau[1] + alpha * dir_nocorr.tau[1]
    #     kap_c = cand.kap[1] = point.kap[1] + alpha * dir_nocorr.kap[1]
    #     (min(tau_c, kap_c, tau_c * kap_c) < eps(T)) && return false
    #
    #     @. cand.z = point.z + alpha * dir_nocorr.z
    #     @. cand.s = point.s + alpha * dir_nocorr.s
    # end

    return true
end

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
