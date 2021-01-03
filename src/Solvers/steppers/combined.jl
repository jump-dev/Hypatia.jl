#=
combined predict and center stepper
=#

mutable struct CombinedStepper{T <: Real} <: Stepper{T}
    use_correction::Bool # TODO remove if unused
    prev_alpha::T
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
    # TODO allow no correction
    # if stepper.use_correction && !any(Cones.use_correction, model.cones)
    #     # model has no cones that use correction
    #     stepper.use_correction = false
    # end
    @assert stepper.use_correction

    stepper.prev_alpha = one(T)
    stepper.rhs = Point(model)
    stepper.dir = Point(model)
    stepper.temp = Point(model)
    stepper.dir_cent = Point(model, ztsk_only = true)
    stepper.dir_pred = Point(model, ztsk_only = true)
    if stepper.use_correction
        stepper.dir_centcorr = Point(model, ztsk_only = true)
        stepper.dir_predcorr = Point(model, ztsk_only = true)
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

    # if stepper.use_correction # TODO?

    stepper.uncorr_only = stepper.cent_only = false
    alpha = search_alpha(point, model, stepper)

    if iszero(alpha)
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
                return false
            end

            # step
            @. point.vec += alpha * dir_cent.vec
        else
            # step
            alpha_m1 = 1 - alpha
            @. point.vec += alpha * dir_pred.vec + alpha_m1 * dir_cent.vec
        end
    else
        # step
        alpha_sqr = abs2(alpha)
        alpha_m1 = 1 - alpha
        alpha_m1sqr = abs2(alpha_m1)
        @. point.vec += alpha * dir_pred.vec + alpha_sqr * dir_predcorr.vec + alpha_m1 * dir_cent.vec + alpha_m1sqr * dir_centcorr.vec
    end

    stepper.prev_alpha = alpha

    return true
end

expect_improvement(stepper::CombinedStepper) = true

function update_cone_points(
    alpha::T,
    point::Point{T},
    stepper::CombinedStepper{T}
    ) where {T <: Real}
    cand = stepper.temp
    dir_cent = stepper.dir_cent
    dir_pred = stepper.dir_pred

    if stepper.cent_only
        @assert stepper.uncorr_only
        @. cand.ztsk = point.ztsk + alpha * dir_cent.ztsk
    elseif stepper.uncorr_only
        alpha_m1 = 1 - alpha
        @. cand.ztsk = point.ztsk + alpha * dir_pred.ztsk + alpha_m1 * dir_cent.ztsk
    else
        dir_centcorr = stepper.dir_centcorr
        dir_predcorr = stepper.dir_predcorr
        alpha_sqr = abs2(alpha)
        alpha_m1 = 1 - alpha
        alpha_m1sqr = abs2(alpha_m1)
        @. cand.ztsk = point.ztsk + alpha * dir_pred.ztsk + alpha_sqr * dir_predcorr.ztsk + alpha_m1 * dir_cent.ztsk + alpha_m1sqr * dir_centcorr.ztsk
    end

    return
end

print_header_more(stepper::CombinedStepper, solver::Solver) = @printf("%5s %9s", "step", "alpha")

function print_iteration_more(stepper::CombinedStepper, solver::Solver)
    step = (stepper.cent_only ? "cent" : (stepper.uncorr_only ? "comb" : "corr"))
    @printf("%5s %9.2e", step, stepper.prev_alpha)
    return
end
