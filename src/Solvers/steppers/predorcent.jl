#=
predict or center stepper
=#

mutable struct PredOrCentStepper{T <: Real} <: Stepper{T}
    use_adjustment::Bool
    use_curve_search::Bool
    max_cent_steps::Int
    pred_prox_bound::T
    use_pred_sum_prox::Bool
    searcher_options

    prev_alpha::T
    cent_count::Int
    rhs::Point{T}
    dir::Point{T}
    temp::Point{T}
    dir_noadj::Point{T}
    dir_adj::Point{T}
    dir_temp::Vector{T}

    searcher::StepSearcher{T}
    unadj_only::Bool
    unadj_alpha::T

    function PredOrCentStepper{T}(;
        use_adjustment::Bool = true,
        use_curve_search::Bool = use_adjustment,
        max_cent_steps::Int = 4,
        pred_prox_bound::T = T(0.0332), # from Alfonso solver
        use_pred_sum_prox::Bool = false,
        searcher_options...
        ) where {T <: Real}
        stepper = new{T}()
        if use_curve_search
            # can only use curve search if using adjustment
            @assert use_adjustment
        end
        stepper.use_adjustment = use_adjustment
        stepper.use_curve_search = use_curve_search
        stepper.max_cent_steps = max_cent_steps
        stepper.pred_prox_bound = pred_prox_bound
        stepper.use_pred_sum_prox = use_pred_sum_prox
        stepper.searcher_options = searcher_options
        return stepper
    end
end

function load(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    if stepper.use_adjustment && !any(Cones.use_dder3, model.cones)
        # model has no cones that use third order deriv
        stepper.use_adjustment = false
    end

    stepper.prev_alpha = one(T)
    stepper.cent_count = 0
    stepper.rhs = Point(model)
    stepper.dir = Point(model)
    stepper.temp = Point(model)
    stepper.dir_noadj = Point(model, ztsk_only = true)
    if stepper.use_adjustment
        stepper.dir_adj = Point(model, ztsk_only = true)
    end
    stepper.dir_temp = zeros(T, length(stepper.rhs.vec))

    stepper.searcher = StepSearcher{T}(model; stepper.searcher_options...)
    stepper.unadj_only = false
    stepper.unadj_alpha = 0

    return stepper
end

function step(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    model = solver.model
    rhs = stepper.rhs
    dir = stepper.dir
    dir_noadj = stepper.dir_noadj
    cones = model.cones

    # update linear system solver factorization
    solver.time_upsys += @elapsed update_lhs(solver.syssolver, solver)

    # decide whether to predict or center
    is_pred = choose_pred_cent(stepper, solver)
    stepper.cent_count = (is_pred ? 0 : stepper.cent_count + 1)
    rhs_fun_noadj = (is_pred ? update_rhs_pred : update_rhs_cent)

    # get unadjusted direction
    solver.time_uprhs += @elapsed rhs_fun_noadj(solver, rhs)
    solver.time_getdir += @elapsed get_directions(stepper, solver)
    copyto!(dir_noadj.vec, dir.vec)
    try_noadj = true

    if stepper.use_adjustment
        # get adjustment direction
        rhs_fun_adj = (is_pred ? update_rhs_predadj : update_rhs_centadj)
        dir_adj = stepper.dir_adj
        solver.time_uprhs += @elapsed rhs_fun_adj(solver, rhs, dir)
        solver.time_getdir += @elapsed get_directions(stepper, solver)
        copyto!(dir_adj.vec, dir.vec)

        if stepper.use_curve_search
            # do single curve search with adjustment
            stepper.unadj_only = false
            solver.time_search += @elapsed alpha = search_alpha(point, model,
                stepper)

            if iszero(alpha)
                # try not using adjustment
                @warn("very small alpha in curve search; trying without adjustment")
            else
                # step
                update_stepper_points(alpha, point, stepper, false)
                stepper.prev_alpha = alpha
                return true
            end
        else
            # do two line searches, first for unadjusted alpha, then for corrected alpha
            try_noadj = false
            stepper.unadj_only = true
            solver.time_search += @elapsed alpha = search_alpha(point, model,
                stepper)
            stepper.unadj_alpha = alpha
            unadj_sched = stepper.searcher.prev_sched

            if !iszero(alpha)
                stepper.unadj_only = false
                solver.time_search += @elapsed alpha = search_alpha(point,
                    model, stepper)

                if iszero(alpha)
                    # use unadjusted direction: start at alpha found
                    # during unadjusted direction search
                    stepper.unadj_only = true
                    solver.time_search += @elapsed alpha = search_alpha(point,
                        model, stepper, sched = unadj_sched)
                    # check alpha didn't decrease more
                    @assert stepper.searcher.prev_sched == unadj_sched
                end

                update_stepper_points(alpha, point, stepper, false)
                stepper.prev_alpha = alpha
                return true
            end
        end
    end

    if try_noadj
        # do line search in unadjusted direction
        stepper.unadj_only = true
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

# decide whether to predict or center
function choose_pred_cent(stepper::PredOrCentStepper, solver::Solver)
    if stepper.cent_count >= stepper.max_cent_steps
        return true
    else
        rtmu = sqrt(solver.mu)
        use_sum = stepper.use_pred_sum_prox
        proxs = (Cones.get_proximity(cone_k, rtmu, use_sum) for
            cone_k in solver.model.cones)
        prox = (use_sum ? sum(proxs) : maximum(proxs))
        return (!isnan(prox) && prox < stepper.pred_prox_bound)
    end
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
        dir_noadj = stepper.dir_noadj.ztsk
    else
        cand = point.vec
        dir_noadj = stepper.dir_noadj.vec
    end

    @. cand += alpha * dir_noadj
    if !stepper.unadj_only
        dir_adj = (ztsk_only ? stepper.dir_adj.ztsk : stepper.dir_adj.vec)
        adj_factor = (stepper.use_curve_search ? abs2(alpha) :
            alpha * stepper.unadj_alpha)
        @. cand += adj_factor * dir_adj
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
