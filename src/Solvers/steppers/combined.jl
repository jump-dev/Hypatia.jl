#=
combined predict and center stepper
=#

mutable struct CombinedStepper{T <: Real} <: Stepper{T}
    use_correction::Bool # TODO remove if unused
    prev_alpha::T
    rhs::Point{T}
    dir::Point{T}
    res::Point{T} # TODO rename if used for cand and res
    dir_cent::Point{T}
    dir_centcorr::Point{T}
    dir_pred::Point{T}
    dir_predcorr::Point{T}
    dir_temp::Vector{T}
    step_searcher::StepSearcher{T}
    cent_only::Bool
    uncorr_only::Bool

    function CombinedStepper{T}(;
        use_correction::Bool = true,
        ) where {T <: Real}
        stepper = new{T}()
        stepper.use_correction = use_correction
        return stepper
    end
end

function load(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    if stepper.use_correction && !any(Cones.use_correction, model.cones)
        # model has no cones that use correction
        stepper.use_correction = false
    end

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
    stepper.uncorr_only = stepper.cent_only = false

    return stepper
end

function step(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
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

    # if stepper.use_correction # TODO? or not

    stepper.uncorr_only = stepper.cent_only = false
    alpha = search_alpha(point, model, stepper)

    if iszero(alpha)
        # TODO try combined without corrections
        println("trying combined without correction")
        stepper.uncorr_only = true
        alpha = search_alpha(point, model, stepper)

        if iszero(alpha)
            # try centering direction
            println("trying centering without correction")
            stepper.cent_only = true
            alpha = search_alpha(point, model, stepper)

            if iszero(alpha)
                @warn("cannot step in centering direction")
                solver.status = NumericalFailure
                return false
            end

            # step
            @. point.vec += alpha * dir_cent.vec
        else
            # step
            @. point.vec += alpha * dir_pred.vec + (1 - alpha) * dir_cent.vec
        end
    else
        # step
        cent = 1 - alpha
        @. point.vec += alpha * (dir_pred.vec + alpha * dir_predcorr.vec) + cent * (dir_cent.vec + cent * dir_centcorr.vec)
    end

    stepper.prev_alpha = alpha
    calc_mu(solver)

    return true
end

expect_improvement(stepper::CombinedStepper) = true

function update_cone_points(
    alpha::T,
    point::Point{T},
    stepper::CombinedStepper{T}
    ) where {T <: Real}
    cand = stepper.res # TODO rename
    dir_cent = stepper.dir_cent
    dir_centcorr = stepper.dir_centcorr
    dir_pred = stepper.dir_pred
    dir_predcorr = stepper.dir_predcorr

    # TODO check tau, kap, then update in one line entire z,t,s,k with a view in point
    # TODO only s,z,t,k
    if stepper.cent_only
        @assert stepper.uncorr_only # TODO
        @. cand.vec = point.vec + alpha * dir_cent.vec
    elseif stepper.uncorr_only
        @. cand.vec = point.vec + alpha * dir_pred.vec + (1 - alpha) * dir_cent.vec
    else
        cent = 1 - alpha
        @. cand.vec = point.vec + alpha * (dir_pred.vec + alpha * dir_predcorr.vec) + cent * (dir_cent.vec + cent * dir_centcorr.vec)
    end

    tau_c = cand.tau[1]
    kap_c = cand.kap[1]
    (min(tau_c, kap_c, tau_c * kap_c) < eps(T)) && return false

    return true
end

# TODO refac
function print_iteration_stats(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    if iszero(solver.num_iters)
        if iszero(solver.model.p)
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %5s %9s\n",
                "iter", "p_obj", "d_obj", "abs_gap",
                "x_feas", "z_feas", "tau", "kap", "mu",
                "step", "alpha",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap,
                solver.x_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu
                )
        else
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %5s %9s\n",
                "iter", "p_obj", "d_obj", "abs_gap",
                "x_feas", "y_feas", "z_feas", "tau", "kap", "mu",
                "step", "alpha",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu
                )
        end
    else
        step = (stepper.cent_only ? "cent" : (stepper.uncorr_only ? "comb" : "corr"))
        if iszero(solver.model.p)
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %5s %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap,
                solver.x_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu,
                step, stepper.prev_alpha,
                )
        else
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %5s %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.point.tau[1], solver.point.kap[1], solver.mu,
                step, stepper.prev_alpha,
                )
        end
    end
    flush(stdout)
    return
end
