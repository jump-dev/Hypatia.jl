#=
predict or center stepper
=#

mutable struct PredOrCentStepper{T <: Real} <: Stepper{T}
    use_correction::Bool
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

    function PredOrCentStepper{T}(;
        use_correction::Bool = true,
        ) where {T <: Real}
        stepper = new{T}()
        stepper.use_correction = use_correction
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

    return stepper
end

function step(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    model = solver.model
    rhs = stepper.rhs
    dir = stepper.dir
    dir_nocorr = stepper.dir_nocorr

    # update linear system solver factorization
    update_lhs(solver.system_solver, solver)

    # decide whether to predict or center
    is_pred = (stepper.prev_alpha > T(0.1)) && ((stepper.cent_count > 3) || all(Cones.in_neighborhood.(model.cones, sqrt(solver.mu), T(0.05)))) # TODO tune, make option
    stepper.cent_count = (is_pred ? 0 : stepper.cent_count + 1)
    rhs_fun_nocorr = (is_pred ? update_rhs_pred : update_rhs_cent)

    # get uncorrected direction
    rhs_fun_nocorr(solver, rhs)
    get_directions(stepper, solver, is_pred)
    copyto!(dir_nocorr.vec, dir.vec) # TODO maybe instead of copying, pass in the dir point we want into the directions function
    alpha = zero(T)

    if stepper.use_correction
        # get correction direction
        rhs_fun_corr = (is_pred ? update_rhs_predcorr : update_rhs_centcorr)
        dir_corr = stepper.dir_corr
        rhs_fun_corr(solver, rhs, dir)
        get_directions(stepper, solver, is_pred)
        copyto!(dir_corr.vec, dir.vec)

        # do curve search with correction
        stepper.uncorr_only = false
        alpha = search_alpha(point, model, stepper, min_alpha = T(1e-2)) # TODO tune min_alpha
        if iszero(alpha)
            # try not using correction
            @warn("very small alpha in curve search; trying without correction")
        else
            # step
            @. point.vec += alpha * dir_nocorr.vec + abs2(alpha) * dir_corr.vec
        end
    end

    if iszero(alpha)
        # do line search in uncorrected direction
        stepper.uncorr_only = true
        alpha = search_alpha(point, model, stepper, min_alpha = T(1e-2))
        if iszero(alpha) && is_pred
            # do centering step instead
            @warn("very small alpha in line search; trying centering")
            update_rhs_cent(solver, rhs)
            get_directions(stepper, solver, is_pred)
            copyto!(dir_nocorr.vec, dir.vec)
            stepper.cent_count = 1

            alpha = search_alpha(point, model, stepper)
        end

        if iszero(alpha)
            @warn("very small alpha in line search; terminating")
            solver.status = NumericalFailure
            return false
        end

        # step
        @. point.vec += alpha * dir_nocorr.vec
    end

    stepper.prev_alpha = alpha
    calc_mu(solver)

    return true
end

expect_improvement(stepper::PredOrCentStepper) = iszero(stepper.cent_count)

function update_cone_points(
    alpha::T,
    point::Point{T},
    stepper::PredOrCentStepper{T}
    ) where {T <: Real}
    cand = stepper.temp
    dir_nocorr = stepper.dir_nocorr

    if stepper.uncorr_only
        @. cand.ztsk = point.ztsk + alpha * dir_nocorr.ztsk
    else
        dir_corr = stepper.dir_corr
        alpha_sqr = abs2(alpha)
        @. cand.ztsk = point.ztsk + alpha * dir_nocorr.ztsk + alpha_sqr * dir_corr.ztsk
    end

    return
end

print_header_more(stepper::PredOrCentStepper, solver::Solver) = @printf("%5s %9s", "step", "alpha")

function print_iteration_more(stepper::PredOrCentStepper, solver::Solver)
    step = (iszero(stepper.cent_count) ? "pred" : "cent")
    @printf("%5s %9.2e", step, stepper.prev_alpha)
    return
end
