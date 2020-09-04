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
    dir_cent::Vector{T}
    tau_row::Int
    kap_row::Int

    # TODO make a line search cache
    z_ls::Vector{T}
    s_ls::Vector{T}
    primal_views_ls::Vector
    dual_views_ls::Vector
    cone_times::Vector{Float64}
    cone_order::Vector{Int}

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
    q = model.q
    stepper.tau_row = model.n + model.p + q + 1
    stepper.kap_row = stepper.tau_row + q + 1
    stepper.dir_temp = zeros(T, stepper.kap_row)
    stepper.dir_cent = zeros(T, stepper.kap_row)

    cones = model.cones
    stepper.z_ls = zeros(T, q)
    stepper.s_ls = zeros(T, q)
    stepper.primal_views_ls = [view(Cones.use_dual_barrier(cone) ? stepper.z_ls : stepper.s_ls, idxs) for (cone, idxs) in zip(cones, model.cone_idxs)]
    stepper.dual_views_ls = [view(Cones.use_dual_barrier(cone) ? stepper.s_ls : stepper.z_ls, idxs) for (cone, idxs) in zip(cones, model.cone_idxs)]
    stepper.cone_times = zeros(length(cones))
    stepper.cone_order = collect(1:length(cones))

    return stepper
end

# original combined stepper
function step(stepper::HeurCombStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    timer = solver.timer

    # # TODO remove the need for this updating here - should be done in line search (some instances failing without it though)
    # cones = solver.model.cones
    # rtmu = sqrt(solver.mu)
    # irtmu = inv(rtmu)
    # Cones.load_point.(cones, point.primal_views, irtmu)
    # Cones.load_dual_point.(cones, point.dual_views)
    # Cones.reset_data.(cones)
    # @assert all(Cones.is_feas.(cones))
    # Cones.grad.(cones)
    # Cones.hess.(cones)
    # # @assert all(Cones.in_neighborhood.(cones, rtmu, T(0.7)))

    # update linear system solver factorization and helpers
    Cones.grad.(solver.model.cones)
    @timeit timer "update_lhs" update_lhs(solver.system_solver, solver)

    # calculate centering direction and keep in dir_cent
    @timeit timer "rhs_cent" update_rhs_cent(solver, stepper.rhs)
    @timeit timer "dir_cent" get_directions(stepper, solver, false, iter_ref_steps = 3)
    dir_cent = copy(stepper.dir) # TODO
    @timeit timer "rhs_centcorr" update_rhs_centcorr(solver, stepper.rhs, stepper.dir)
    @timeit timer "dir_centcorr" get_directions(stepper, solver, false, iter_ref_steps = 3)
    dir_centcorr = copy(stepper.dir) # TODO
    # copyto!(stepper.dir_cent, stepper.dir)

    # calculate affine/prediction direction and keep in dir
    @timeit timer "rhs_pred" update_rhs_pred(solver, stepper.rhs)
    @timeit timer "dir_pred" get_directions(stepper, solver, true, iter_ref_steps = 3)
    dir_pred = copy(stepper.dir) # TODO
    @timeit timer "rhs_predcorr" update_rhs_predcorr(solver, stepper.rhs, stepper.dir)
    @timeit timer "dir_predcorr" get_directions(stepper, solver, true, iter_ref_steps = 3)
    dir_predcorr = copy(stepper.dir) # TODO

    # calculate centering factor gamma by finding distance pred_alpha for stepping in pred direction
    copyto!(stepper.dir, dir_pred)
    @timeit timer "alpha_pred" stepper.prev_pred_alpha = pred_alpha = find_max_alpha(stepper, solver, prev_alpha = stepper.prev_pred_alpha, min_alpha = T(1e-2), max_nbhd = one(T)) # TODO try max_nbhd = Inf, but careful of cones with no dual feas check

    # TODO allow different function (heuristic) as option?
    # stepper.prev_gamma = gamma = abs2(1 - pred_alpha)
    stepper.prev_gamma = gamma = 1 - pred_alpha

    # calculate combined direction and keep in dir
    # axpby!(gamma, stepper.dir_cent, 1 - gamma, stepper.dir)
    @. stepper.dir = gamma * (dir_cent + pred_alpha * dir_centcorr) + (1 - gamma) * (dir_pred + pred_alpha * dir_predcorr) # TODO

    # find distance alpha for stepping in combined direction
    @timeit timer "alpha_comb" alpha = find_max_alpha(stepper, solver, prev_alpha = stepper.prev_alpha, min_alpha = T(1e-3))

    if iszero(alpha)
        # could not step far in combined direction, so attempt a pure centering step
        solver.verbose && println("performing centering step")
        # copyto!(stepper.dir, stepper.dir_cent)
        @. stepper.dir = dir_cent + dir_centcorr

        # find distance alpha for stepping in centering direction
        @timeit timer "alpha_cent" alpha = find_max_alpha(stepper, solver, prev_alpha = one(T), min_alpha = T(1e-6))

        if iszero(alpha)
            copyto!(stepper.dir, dir_cent)
            @timeit timer "alpha_cent2" alpha = find_max_alpha(stepper, solver, prev_alpha = one(T), min_alpha = T(1e-6))

            if iszero(alpha)
                @warn("numerical failure: could not step in centering direction; terminating")
                solver.status = :NumericalFailure
                return false
            end
        end
    end
    stepper.prev_alpha = alpha

    # step distance alpha in combined direction
    @. point.x += alpha * stepper.dir.x
    @. point.y += alpha * stepper.dir.y
    @. point.z += alpha * stepper.dir.z
    @. point.s += alpha * stepper.dir.s
    solver.point.tau += alpha * stepper.dir[stepper.tau_row]
    solver.point.kap += alpha * stepper.dir[stepper.kap_row]
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
