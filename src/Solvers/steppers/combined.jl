#=
combined predict and center stepper
=#

mutable struct CombinedStepper{T <: Real} <: Stepper{T}
    prev_alpha::T
    shift_alpha_sched::Int
    rhs::Point{T}
    dir::Point{T}
    temp::Point{T}
    dir_cent::Point{T}
    dir_centcorr::Point{T}
    dir_pred::Point{T}
    dir_predcorr::Point{T}
    dir_temp::Vector{T}
    step_searcher::StepSearcher{T}
    cent_only::Bool
    uncorr_only::Bool

    function CombinedStepper{T}(
        shift_alpha_sched::Int = 4, # TODO tune, maybe use heuristic based on how fast alpha search is compared to a full IPM iteration
        ) where {T <: Real}
        stepper = new{T}()
        stepper.shift_alpha_sched = shift_alpha_sched
        return stepper
    end
end

function load(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model

    stepper.prev_alpha = one(T)
    stepper.rhs = Point(model)
    stepper.dir = Point(model)
    stepper.temp = Point(model)
    stepper.dir_cent = Point(model, ztsk_only = true)
    stepper.dir_pred = Point(model, ztsk_only = true)
    stepper.dir_centcorr = Point(model, ztsk_only = true)
    stepper.dir_predcorr = Point(model, ztsk_only = true)
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
    get_directions(stepper, solver, false)
    copyto!(dir_cent.vec, dir.vec)
    update_rhs_centcorr(solver, rhs, dir)
    get_directions(stepper, solver, false)
    copyto!(dir_centcorr.vec, dir.vec)

    # calculate affine/prediction direction and correction
    update_rhs_pred(solver, rhs)
    get_directions(stepper, solver, true)
    copyto!(dir_pred.vec, dir.vec)
    update_rhs_predcorr(solver, rhs, dir)
    get_directions(stepper, solver, true)
    copyto!(dir_predcorr.vec, dir.vec)

    # search with combined directions and corrections
    stepper.uncorr_only = stepper.cent_only = false
    alpha = search_alpha(point, model, stepper)

    if iszero(alpha)
        # recover
        println("trying combined without correction")
        stepper.uncorr_only = true
        alpha = search_alpha(point, model, stepper)

        if iszero(alpha)
            println("trying centering with correction")
            stepper.cent_only = true
            stepper.uncorr_only = false
            alpha = search_alpha(point, model, stepper)

            if iszero(alpha)
                println("trying centering without correction")
                stepper.uncorr_only = true
                alpha = search_alpha(point, model, stepper)

                if iszero(alpha)
                    @warn("cannot step in centering direction")
                    solver.status = NumericalFailure
                    stepper.prev_alpha = alpha
                    return false
                end
            end
        end
    end

    # step
    update_cone_points(alpha, point, stepper, false)
    stepper.prev_alpha = alpha

    return true
end

expect_improvement(stepper::CombinedStepper) = true

function update_cone_points(
    alpha::T,
    point::Point{T},
    stepper::CombinedStepper{T},
    ztsk_only::Bool,
    ) where {T <: Real}
    if ztsk_only
        cand = stepper.temp.ztsk
        copyto!(cand, point.ztsk)
        dir_cent = stepper.dir_cent.ztsk
        dir_pred = stepper.dir_pred.ztsk
    else
        cand = point.vec
        dir_cent = stepper.dir_cent.vec
        dir_pred = stepper.dir_pred.vec
    end

    if stepper.uncorr_only
        # no correction
        if stepper.cent_only
            # centering
            @. cand += alpha * dir_cent
        else
            # combined
            alpha_m1 = 1 - alpha
            @. cand += alpha * dir_pred + alpha_m1 * dir_cent
        end
    else
        # correction
        dir_centcorr = (ztsk_only ? stepper.dir_centcorr.ztsk : stepper.dir_centcorr.vec)
        alpha_sqr = abs2(alpha)
        if stepper.cent_only
            # centering
            @. cand += alpha * dir_cent + alpha_sqr * dir_centcorr
        else
            # combined
            dir_predcorr = (ztsk_only ? stepper.dir_predcorr.ztsk : stepper.dir_predcorr.vec)
            alpha_m1 = 1 - alpha
            alpha_m1sqr = abs2(alpha_m1)
            @. cand += alpha * dir_pred + alpha_sqr * dir_predcorr + alpha_m1 * dir_cent + alpha_m1sqr * dir_centcorr
        end
    end

    return
end

function start_sched(stepper::CombinedStepper, step_searcher::StepSearcher)
    (stepper.shift_alpha_sched <= 0) && return 1
    return max(1, step_searcher.prev_sched - stepper.shift_alpha_sched)
end

print_header_more(stepper::CombinedStepper, solver::Solver) = @printf("%5s %9s", "step", "alpha")

function print_iteration_more(stepper::CombinedStepper, solver::Solver)
    if stepper.cent_only
        step = (stepper.uncorr_only ? "cent" : "ce-c")
    else
        step = (stepper.uncorr_only ? "comb" : "co-c")
    end
    @printf("%5s %9.2e", step, stepper.prev_alpha)
    return
end
