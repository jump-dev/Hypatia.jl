#=
predict or center stepper
=#

mutable struct PredOrCentStepper{T <: Real} <: Stepper{T}
    use_correction::Bool
    use_curve_search::Bool
    prev_alpha::T
    cent_count::Int

    rhs::Point{T}
    dir::Point{T}
    temp::Point{T}
    dir_nocorr::Point{T}
    dir_corr::Point{T}
    dir_temp::Vector{T}

    step_searcher::StepSearcher{T}
    uncorr_only::Bool
    uncorr_alpha::T

    function PredOrCentStepper{T}(;
        use_correction::Bool = true,
        use_curve_search::Bool = true,
        ) where {T <: Real}
        stepper = new{T}()
        if use_curve_search
            # can only use curve search if using correction
            @assert use_correction
        end
        stepper.use_correction = use_correction
        stepper.use_curve_search = use_curve_search
        return stepper
    end
end

function load(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    if stepper.use_correction && !any(Cones.use_correction, model.cones)
        # model has no cones that use correction
        stepper.use_correction = false
    end

    stepper.prev_alpha = one(T)
    stepper.cent_count = 0

    stepper.rhs = Point(model)
    stepper.dir = Point(model)
    stepper.temp = Point(model)
    stepper.dir_nocorr = Point(model, ztsk_only = true) # TODO probably don't need this AND dir
    if stepper.use_correction
        stepper.dir_corr = Point(model, ztsk_only = true)
    end
    stepper.dir_temp = zeros(T, length(stepper.rhs.vec))

    stepper.step_searcher = StepSearcher{T}(model)
    stepper.uncorr_only = false
    stepper.uncorr_alpha = 0

    return stepper
end

function step(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    model = solver.model
    rhs = stepper.rhs
    dir = stepper.dir
    dir_nocorr = stepper.dir_nocorr
    cones = model.cones

    # update linear system solver factorization
    solver.time_upsys += @elapsed update_lhs(solver.syssolver, solver)

    # decide whether to predict or center
    is_pred = ((stepper.cent_count > 3) ||
        all(Cones.in_neighborhood.(cones, sqrt(solver.mu), T(0.05)))) # TODO tune, make option
    stepper.cent_count = (is_pred ? 0 : stepper.cent_count + 1)
    rhs_fun_nocorr = (is_pred ? update_rhs_pred : update_rhs_cent)

    # get uncorrected direction
    solver.time_uprhs += @elapsed rhs_fun_nocorr(solver, rhs)
    solver.time_getdir += @elapsed get_directions(stepper, solver, is_pred)
    copyto!(dir_nocorr.vec, dir.vec) # TODO maybe instead of copying, pass in the dir point we want into the directions function
    try_nocorr = true

    if stepper.use_correction
        # get correction direction
        rhs_fun_corr = (is_pred ? update_rhs_predcorr : update_rhs_centcorr)
        dir_corr = stepper.dir_corr
        solver.time_uprhs += @elapsed rhs_fun_corr(solver, rhs, dir)
        solver.time_getdir += @elapsed get_directions(stepper, solver, is_pred)
        copyto!(dir_corr.vec, dir.vec)

        if stepper.use_curve_search
            # do single curve search with correction
            stepper.uncorr_only = false
            solver.time_search += @elapsed alpha = search_alpha(point, model,
                stepper)

            if iszero(alpha)
                # try not using correction
                @warn("very small alpha in curve search; trying without correction")
            else
                # step
                update_stepper_points(alpha, point, stepper, false)
                stepper.prev_alpha = alpha
                return true
            end
        else
            # do two line searches, first for uncorrected alpha, then for corrected alpha
            try_nocorr = false
            stepper.uncorr_only = true
            solver.time_search += @elapsed alpha = search_alpha(point, model,
                stepper)
            stepper.uncorr_alpha = alpha
            uncorr_sched = stepper.step_searcher.prev_sched

            if !iszero(alpha)
                stepper.uncorr_only = false
                solver.time_search += @elapsed alpha = search_alpha(point,
                    model, stepper)

                if iszero(alpha)
                    # use uncorrected direction: start at alpha found
                    # during uncorrected direction search
                    stepper.uncorr_only = true
                    solver.time_search += @elapsed alpha = search_alpha(point,
                        model, stepper, sched = uncorr_sched)
                    # check alpha didn't decrease more
                    @assert stepper.step_searcher.prev_sched == uncorr_sched
                end

                update_stepper_points(alpha, point, stepper, false)
                stepper.prev_alpha = alpha
                return true
            end
        end
    end

    if try_nocorr
        # do line search in uncorrected direction
        stepper.uncorr_only = true
        solver.time_search += @elapsed alpha = search_alpha(point, model, stepper)
    end

    if iszero(alpha)
        @warn("very small alpha in line search; terminating")
        solver.status = NumericalFailure
        stepper.prev_alpha = alpha
        return false
    end

    # step
    update_stepper_points(alpha, point, stepper, false)
    stepper.prev_alpha = alpha

    return true
end

expect_improvement(stepper::PredOrCentStepper) = iszero(stepper.cent_count)

function update_stepper_points(
    alpha::T,
    point::Point{T},
    stepper::PredOrCentStepper{T},
    ztsk_only::Bool,
    ) where {T <: Real}
    if ztsk_only
        cand = stepper.temp.ztsk
        copyto!(cand, point.ztsk)
        dir_nocorr = stepper.dir_nocorr.ztsk
    else
        cand = point.vec
        dir_nocorr = stepper.dir_nocorr.vec
    end

    @. cand += alpha * dir_nocorr
    if !stepper.uncorr_only
        dir_corr = (ztsk_only ? stepper.dir_corr.ztsk : stepper.dir_corr.vec)
        corr_factor = (stepper.use_curve_search ? abs2(alpha) :
            alpha * stepper.uncorr_alpha)
        @. cand += corr_factor * dir_corr
    end

    return
end

print_header_more(stepper::PredOrCentStepper, solver::Solver) =
    @printf("%5s %9s", "step", "alpha")

function print_iteration_more(stepper::PredOrCentStepper, solver::Solver)
    step = (iszero(stepper.cent_count) ? "pred" : "cent")
    @printf("%5s %9.2e", step, stepper.prev_alpha)
    return
end
