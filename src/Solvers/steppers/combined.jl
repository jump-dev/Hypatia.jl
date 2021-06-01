#=
combined predict and center stepper
=#

mutable struct CombinedStepper{T <: Real} <: Stepper{T}
    shift_sched::Int
    searcher_options

    prev_alpha::T
    rhs::Point{T}
    dir::Point{T}
    temp::Point{T}
    dir_cent::Point{T}
    dir_centadj::Point{T}
    dir_pred::Point{T}
    dir_predadj::Point{T}
    dir_temp::Vector{T}

    searcher::StepSearcher{T}
    cent_only::Bool
    unadj_only::Bool

    function CombinedStepper{T}(;
        shift_sched::Int = 0,
        searcher_options...
        ) where {T <: Real}
        stepper = new{T}()
        stepper.shift_sched = shift_sched
        stepper.searcher_options = searcher_options
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
    stepper.dir_centadj = Point(model, ztsk_only = true)
    stepper.dir_predadj = Point(model, ztsk_only = true)
    stepper.dir_temp = zeros(T, length(stepper.rhs.vec))

    stepper.searcher = StepSearcher{T}(model; stepper.searcher_options...)
    stepper.unadj_only = stepper.cent_only = false

    return stepper
end

function step(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    model = solver.model
    rhs = stepper.rhs
    dir = stepper.dir
    dir_cent = stepper.dir_cent
    dir_centadj = stepper.dir_centadj
    dir_pred = stepper.dir_pred
    dir_predadj = stepper.dir_predadj

    # update linear system solver factorization
    solver.time_upsys += @elapsed update_lhs(solver.syssolver, solver)

    # calculate centering direction and adjustment
    solver.time_uprhs += @elapsed update_rhs_cent(solver, rhs)
    solver.time_getdir += @elapsed get_directions(stepper, solver)
    copyto!(dir_cent.vec, dir.vec)
    solver.time_uprhs += @elapsed update_rhs_centadj(solver, rhs, dir)
    solver.time_getdir += @elapsed get_directions(stepper, solver)
    copyto!(dir_centadj.vec, dir.vec)

    # calculate affine/prediction direction and adjustment
    solver.time_uprhs += @elapsed update_rhs_pred(solver, rhs)
    solver.time_getdir += @elapsed get_directions(stepper, solver)
    copyto!(dir_pred.vec, dir.vec)
    solver.time_uprhs += @elapsed update_rhs_predadj(solver, rhs, dir)
    solver.time_getdir += @elapsed get_directions(stepper, solver)
    copyto!(dir_predadj.vec, dir.vec)

    # search with combined directions and adjustments
    stepper.unadj_only = stepper.cent_only = false
    solver.time_search += @elapsed alpha = search_alpha(point, model, stepper)

    if iszero(alpha)
        # recover
        solver.verbose && println("trying combined without adjustment")
        stepper.unadj_only = true
        solver.time_search += @elapsed alpha = search_alpha(point, model, stepper)

        if iszero(alpha)
            solver.verbose && println("trying centering with adjustment")
            stepper.cent_only = true
            stepper.unadj_only = false
            solver.time_search += @elapsed alpha =
                search_alpha(point, model, stepper)

            if iszero(alpha)
                solver.verbose && println("trying centering without adjustment")
                stepper.unadj_only = true
                solver.time_search += @elapsed alpha =
                    search_alpha(point, model, stepper)

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
    update_stepper_points(alpha, point, stepper, false)
    stepper.prev_alpha = alpha

    return true
end

expect_improvement(stepper::CombinedStepper) = true

function update_stepper_points(
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

    if stepper.unadj_only
        # no adjustment
        if stepper.cent_only
            # centering
            @. cand += alpha * dir_cent
        else
            # combined
            alpha_m1 = 1 - alpha
            @. cand += alpha * dir_pred + alpha_m1 * dir_cent
        end
    else
        # adjustment
        dir_centadj = (ztsk_only ? stepper.dir_centadj.ztsk :
            stepper.dir_centadj.vec)
        alpha_sqr = abs2(alpha)
        if stepper.cent_only
            # centering
            @. cand += alpha * dir_cent + alpha_sqr * dir_centadj
        else
            # combined
            dir_predadj = (ztsk_only ? stepper.dir_predadj.ztsk :
                stepper.dir_predadj.vec)
            alpha_m1 = 1 - alpha
            alpha_m1sqr = abs2(alpha_m1)
            @. cand += alpha * dir_pred + alpha_sqr * dir_predadj +
                alpha_m1 * dir_cent + alpha_m1sqr * dir_centadj
        end
    end

    return
end

function start_sched(stepper::CombinedStepper, searcher::StepSearcher)
    (stepper.shift_sched <= 0) && return 1
    return max(1, searcher.prev_sched - stepper.shift_sched)
end

print_header_more(stepper::CombinedStepper, solver::Solver) =
    @printf("%5s %9s", "step", "alpha")

function print_iteration_more(stepper::CombinedStepper, solver::Solver)
    if stepper.cent_only
        step = (stepper.unadj_only ? "cent" : "ce-a")
    else
        step = (stepper.unadj_only ? "comb" : "co-a")
    end
    @printf("%5s %9.2e", step, stepper.prev_alpha)
    return
end
