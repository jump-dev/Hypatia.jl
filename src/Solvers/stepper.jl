#=
Copyright 2019, Chris Coey and contributors

interior point stepping routines for algorithms based on homogeneous self dual embedding
=#

mutable struct CombinedStepper{T <: Real} <: Stepper{T}
    prev_pred_alpha::T
    prev_alpha::T
    prev_gamma::T
    rhs::Vector{T}
    x_rhs
    y_rhs
    z_rhs
    s_rhs
    s_rhs_k::Vector
    dir::Vector{T}
    x_dir
    y_dir
    z_dir
    dual_dir_k::Vector
    s_dir
    primal_dir_k::Vector
    dir_temp::Vector{T}
    dir_cent::Vector{T}
    res::Vector{T}
    x_res
    y_res
    z_res
    s_res
    s_res_k::Vector
    tau_row::Int
    kap_row::Int
    z_ls::Vector{T}
    s_ls::Vector{T}
    primal_views_ls::Vector
    dual_views_ls::Vector
    cone_times::Vector{Float64}
    cone_order::Vector{Int}

    CombinedStepper{T}() where {T <: Real} = new{T}()
end

# create the stepper cache
function load(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    stepper.prev_pred_alpha = one(T)
    stepper.prev_gamma = one(T)
    stepper.prev_alpha = one(T)

    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs

    dim = n + p + 2q + 2
    rhs = zeros(T, dim)
    dir = zeros(T, dim)
    res = zeros(T, dim)
    stepper.rhs = rhs
    stepper.dir = dir
    stepper.dir_temp = zeros(T, dim)
    stepper.dir_cent = zeros(T, dim)
    stepper.res = res

    rows = 1:n
    stepper.x_rhs = view(rhs, rows)
    stepper.x_dir = view(dir, rows)
    stepper.x_res = view(res, rows)

    rows = n .+ (1:p)
    stepper.y_rhs = view(rhs, rows)
    stepper.y_dir = view(dir, rows)
    stepper.y_res = view(res, rows)

    rows = (n + p) .+ (1:q)
    stepper.z_rhs = view(rhs, rows)
    stepper.z_dir = view(dir, rows)
    stepper.z_res = view(res, rows)

    tau_row = n + p + q + 1
    stepper.tau_row = tau_row

    rows = tau_row .+ (1:q)
    stepper.s_rhs = view(rhs, rows)
    stepper.s_rhs_k = [view(rhs, tau_row .+ idxs_k) for idxs_k in cone_idxs]
    stepper.s_dir = view(dir, rows)
    stepper.s_res = view(res, rows)
    stepper.s_res_k = [view(res, tau_row .+ idxs_k) for idxs_k in cone_idxs]

    stepper.primal_dir_k = similar(stepper.s_res_k)
    stepper.dual_dir_k = similar(stepper.s_res_k)
    for (k, idxs_k) in enumerate(cone_idxs)
        s_k = view(dir, tau_row .+ idxs_k)
        z_k = view(dir, (n + p) .+ idxs_k)
        (stepper.primal_dir_k[k], stepper.dual_dir_k[k]) = (Cones.use_dual_barrier(cones[k]) ? (z_k, s_k) : (s_k, z_k))
    end

    stepper.kap_row = dim

    stepper.z_ls = zeros(T, q)
    stepper.s_ls = zeros(T, q)
    stepper.primal_views_ls = [view(Cones.use_dual_barrier(model.cones[k]) ? stepper.z_ls : stepper.s_ls, model.cone_idxs[k]) for k in eachindex(model.cones)]
    stepper.dual_views_ls = [view(Cones.use_dual_barrier(model.cones[k]) ? stepper.s_ls : stepper.z_ls, model.cone_idxs[k]) for k in eachindex(model.cones)]

    stepper.cone_times = zeros(Float64, length(solver.model.cones))
    stepper.cone_order = collect(1:length(solver.model.cones))

    return stepper
end

# # original combined stepper
# function step(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
#     point = solver.point
#     timer = solver.timer
#
#     # TODO remove the need for this updating here - should be done in line search (some instances failing without it though)
#     rtmu = sqrt(solver.mu)
#     irtmu = inv(rtmu)
#     Cones.load_point.(cones, point.primal_views, irtmu)
#     Cones.load_dual_point.(cones, point.dual_views)
#     Cones.reset_data.(cones)
#     @assert all(Cones.is_feas.(cones))
#     Cones.grad.(cones)
#     Cones.hess.(cones)
#     # @assert all(Cones.in_neighborhood.(cones, rtmu, T(0.7)))

#     # update linear system solver factorization and helpers
#     Cones.grad.(solver.model.cones)
#     @timeit timer "update_lhs" update_lhs(solver.system_solver, solver)
#
#     # calculate centering direction and keep in dir_cent
#     @timeit timer "rhs_cent" update_rhs_cent(stepper, solver)
#     @timeit timer "dir_cent" get_directions(stepper, solver, false, iter_ref_steps = 3)
#     dir_cent = copy(stepper.dir) # TODO
#     @timeit timer "rhs_centcorr" update_rhs_centcorr(stepper, solver)
#     @timeit timer "dir_centcorr" get_directions(stepper, solver, false, iter_ref_steps = 3)
#     dir_centcorr = copy(stepper.dir) # TODO
#     # copyto!(stepper.dir_cent, stepper.dir)
#
#     # calculate affine/prediction direction and keep in dir
#     @timeit timer "rhs_pred" update_rhs_pred(stepper, solver)
#     @timeit timer "dir_pred" get_directions(stepper, solver, true, iter_ref_steps = 3)
#     dir_pred = copy(stepper.dir) # TODO
#     @timeit timer "rhs_predcorr" update_rhs_predcorr(stepper, solver)
#     @timeit timer "dir_predcorr" get_directions(stepper, solver, true, iter_ref_steps = 3)
#     dir_predcorr = copy(stepper.dir) # TODO
#
#     # calculate centering factor gamma by finding distance pred_alpha for stepping in pred direction
#     copyto!(stepper.dir, dir_pred)
#     @timeit timer "alpha_pred" stepper.prev_pred_alpha = pred_alpha = find_max_alpha(stepper, solver, prev_alpha = stepper.prev_pred_alpha, min_alpha = T(1e-2), max_nbhd = one(T)) # TODO try max_nbhd = Inf, but careful of cones with no dual feas check
#
#     # TODO allow different function (heuristic) as option?
#     # stepper.prev_gamma = gamma = abs2(1 - pred_alpha)
#     stepper.prev_gamma = gamma = 1 - pred_alpha
#
#     # calculate combined direction and keep in dir
#     # axpby!(gamma, stepper.dir_cent, 1 - gamma, stepper.dir)
#     @. stepper.dir = gamma * (dir_cent + pred_alpha * dir_centcorr) + (1 - gamma) * (dir_pred + pred_alpha * dir_predcorr) # TODO
#
#     # find distance alpha for stepping in combined direction
#     @timeit timer "alpha_comb" alpha = find_max_alpha(stepper, solver, prev_alpha = stepper.prev_alpha, min_alpha = T(1e-3))
#
#     if iszero(alpha)
#         # could not step far in combined direction, so attempt a pure centering step
#         solver.verbose && println("performing centering step")
#         # copyto!(stepper.dir, stepper.dir_cent)
#         @. stepper.dir = dir_cent + dir_centcorr
#
#         # find distance alpha for stepping in centering direction
#         @timeit timer "alpha_cent" alpha = find_max_alpha(stepper, solver, prev_alpha = one(T), min_alpha = T(1e-6))
#
#         if iszero(alpha)
#             copyto!(stepper.dir, dir_cent)
#             @timeit timer "alpha_cent2" alpha = find_max_alpha(stepper, solver, prev_alpha = one(T), min_alpha = T(1e-6))
#
#             if iszero(alpha)
#                 @warn("numerical failure: could not step in centering direction; terminating")
#                 solver.status = :NumericalFailure
#                 return false
#             end
#         end
#     end
#     stepper.prev_alpha = alpha
#
#     # step distance alpha in combined direction
#     @. point.x += alpha * stepper.x_dir
#     @. point.y += alpha * stepper.y_dir
#     @. point.z += alpha * stepper.z_dir
#     @. point.s += alpha * stepper.s_dir
#     solver.tau += alpha * stepper.dir[stepper.tau_row]
#     solver.kap += alpha * stepper.dir[stepper.kap_row]
#     calc_mu(solver)
#
#     if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
#         @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
#         solver.status = :NumericalFailure
#         return false
#     end
#
#     return true
# end

# stepper using line search between cent and pred points
function step(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    timer = solver.timer

    # # TODO remove the need for this updating here - should be done in line search (some instances failing without it though)
    # rtmu = sqrt(solver.mu)
    # irtmu = inv(rtmu)
    # cones = solver.model.cones
    # Cones.load_point.(cones, point.primal_views, irtmu)
    # Cones.load_dual_point.(cones, point.dual_views)
    # Cones.reset_data.(cones)
    # @assert all(Cones.is_feas.(cones))
    # Cones.grad.(cones)
    # Cones.hess.(cones)
    # # # @assert all(Cones.in_neighborhood.(cones, rtmu, T(0.7)))

    # update linear system solver factorization and helpers
    Cones.grad.(solver.model.cones)
    @timeit timer "update_lhs" update_lhs(solver.system_solver, solver)

    z_dir = stepper.z_dir
    s_dir = stepper.s_dir

    # calculate centering direction and keep in dir_cent
    @timeit timer "rhs_cent" update_rhs_cent(stepper, solver)
    @timeit timer "dir_cent" get_directions(stepper, solver, false, iter_ref_steps = 3)
    dir_cent = copy(stepper.dir) # TODO
    tau_cent = stepper.dir[stepper.tau_row]
    kap_cent = stepper.dir[stepper.kap_row]
    z_cent = copy(z_dir)
    s_cent = copy(s_dir)

    @timeit timer "rhs_centcorr" update_rhs_centcorr(stepper, solver)
    @timeit timer "dir_centcorr" get_directions(stepper, solver, false, iter_ref_steps = 3)
    dir_centcorr = copy(stepper.dir) # TODO
    tau_centcorr = stepper.dir[stepper.tau_row]
    kap_centcorr = stepper.dir[stepper.kap_row]
    z_centcorr = copy(z_dir)
    s_centcorr = copy(s_dir)

    # calculate affine/prediction direction and keep in dir
    @timeit timer "rhs_pred" update_rhs_pred(stepper, solver)
    @timeit timer "dir_pred" get_directions(stepper, solver, true, iter_ref_steps = 3)
    dir_pred = copy(stepper.dir) # TODO
    tau_pred = stepper.dir[stepper.tau_row]
    kap_pred = stepper.dir[stepper.kap_row]
    z_pred = copy(z_dir)
    s_pred = copy(s_dir)

    @timeit timer "rhs_predcorr" update_rhs_predcorr(stepper, solver)
    @timeit timer "dir_predcorr" get_directions(stepper, solver, true, iter_ref_steps = 3)
    dir_predcorr = copy(stepper.dir) # TODO
    tau_predcorr = stepper.dir[stepper.tau_row]
    kap_predcorr = stepper.dir[stepper.kap_row]
    z_predcorr = copy(z_dir)
    s_predcorr = copy(s_dir)

    # TODO check cent point (step 1) is acceptable
    @. stepper.dir = dir_cent + dir_centcorr
    alpha = find_max_alpha(stepper, solver, prev_alpha = one(T), min_alpha = T(0.99))
    # TODO cleanup
    if iszero(alpha)
        @warn("could not do full step in centering-correction direction")
        dir_centcorr .= 0
        stepper.dir .= 0
        tau_centcorr = stepper.dir[stepper.tau_row]
        kap_centcorr = stepper.dir[stepper.kap_row]
        z_centcorr = copy(z_dir)
        s_centcorr = copy(s_dir)
    end

    # TODO use a smarter line search, eg bisection
    # TODO start with beta = 0.9999 for pred factor, and decrease until point satisfies nbhd
    cones = solver.model.cones
    cone_times = stepper.cone_times
    cone_order = stepper.cone_order
    z = solver.point.z
    s = solver.point.s
    tau = solver.tau
    kap = solver.kap
    # z_dir = stepper.z_dir
    # s_dir = stepper.s_dir
    # tau_dir = stepper.dir[stepper.tau_row]
    # kap_dir = stepper.dir[stepper.kap_row]
    z_ls = stepper.z_ls
    s_ls = stepper.s_ls
    primals_ls = stepper.primal_views_ls
    duals_ls = stepper.dual_views_ls
    sz_ks = zeros(T, length(cone_order)) # TODO prealloc
    tau_ls = zero(T)
    kap_ls = zero(T)

    nup1 = solver.model.nu + 1
    max_nbhd = T(0.99)
    # max_nbhd = one(T)
    min_nbhd = T(0.01)

    # beta_schedule = T[0.9999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    # beta = zero(T)
    beta = max(T(0.1), min(stepper.prev_gamma * T(1.4), one(T))) # TODO option for parameter
    beta_decrease = T(0.95)
    beta *= T(0.9999)
    beta /= beta_decrease

    iter_ls = 0
    in_nbhd = false

    # while iter_ls < length(beta_schedule)
    while beta > 0
        in_nbhd = false
        iter_ls += 1

        # beta = beta_schedule[iter_ls]
        beta *= beta_decrease
        if beta < T(0.01)
            beta = zero(T) # pure centering
        end
        betam1 = 1 - beta

        # TODO shouldn't need to reduce corr on cent?
        tau_ls = tau + betam1 * (tau_cent + betam1 * tau_centcorr) + beta * (tau_pred + beta * tau_predcorr)
        kap_ls = kap + betam1 * (kap_cent + betam1 * kap_centcorr) + beta * (kap_pred + beta * kap_predcorr)
        taukap_ls = tau_ls * kap_ls
        (tau_ls < eps(T) || kap_ls < eps(T) || taukap_ls < eps(T)) && continue

        # order the cones by how long it takes to check neighborhood condition and iterate in that order, to improve efficiency
        sortperm!(cone_order, cone_times, initialized = true) # NOTE stochastic

        @. z_ls = z + betam1 * (z_cent + betam1 * z_centcorr) + beta * (z_pred + beta * z_predcorr)
        @. s_ls = s + betam1 * (s_cent + betam1 * s_centcorr) + beta * (s_pred + beta * s_predcorr)

        for k in cone_order
            sz_ks[k] = dot(primals_ls[k], duals_ls[k])
        end
        any(<(eps(T)), sz_ks) && continue

        mu_ls = (sum(sz_ks) + taukap_ls) / nup1
        (mu_ls < eps(T)) && continue

        # TODO experiment with SY nbhd for tau-kappa
        (abs(taukap_ls - mu_ls) > max_nbhd * mu_ls) && continue

        min_nbhd_mu = min_nbhd * mu_ls
        (taukap_ls < min_nbhd_mu) && continue
        any(sz_ks[k] < min_nbhd_mu * Cones.get_nu(cones[k]) for k in cone_order) && continue

        rtmu = sqrt(mu_ls)
        irtmu = inv(rtmu)
        in_nbhd = true
        for k in cone_order
            cone_k = cones[k]
            time_k = time_ns()

            Cones.load_point(cone_k, primals_ls[k], irtmu)
            Cones.load_dual_point(cone_k, duals_ls[k])
            Cones.reset_data(cone_k)

            in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && Cones.in_neighborhood(cone_k, rtmu, max_nbhd))
            # in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && (isinf(max_nbhd) || Cones.in_neighborhood(cone_k, rtmu, max_nbhd)))
            # TODO is_dual_feas function should fall back to a nbhd-like check (for ray maybe) if not using nbhd check
            # in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k))

            cone_times[k] = time_ns() - time_k
            if !in_nbhd_k
                in_nbhd = false
                break
            end
        end
        in_nbhd && break
        iszero(beta) && break
    end
    # @show iter_ls

    # TODO if zero not feasible, do backwards line search
    if !in_nbhd
        @show beta

        copyto!(stepper.dir, dir_cent)

        alpha = find_max_alpha(stepper, solver, prev_alpha = one(T), min_alpha = T(1e-3))
        if iszero(alpha)
            @warn("numerical failure: could not step in centering direction; terminating")
            solver.status = :NumericalFailure
            return false
        end
        stepper.prev_alpha = alpha

        # step distance alpha in combined direction
        @. point.x += alpha * stepper.x_dir
        @. point.y += alpha * stepper.y_dir
        @. point.z += alpha * stepper.z_dir
        @. point.s += alpha * stepper.s_dir
        solver.tau += alpha * stepper.dir[stepper.tau_row]
        solver.kap += alpha * stepper.dir[stepper.kap_row]
    else
        stepper.prev_gamma = gamma = beta # TODO

        # step to combined point
        copyto!(point.z, z_ls)
        copyto!(point.s, s_ls)
        solver.tau = tau_ls
        solver.kap = kap_ls

        # TODO improve
        betam1 = 1 - beta
        @. stepper.dir = betam1 * (dir_cent + betam1 * dir_centcorr) # TODO shouldn't need to reduce corr on cent
        # @. stepper.dir = betam1 * (dir_cent + dir_centcorr)
        @. point.x += stepper.x_dir
        @. point.y += stepper.y_dir
        @. stepper.dir = beta * (dir_pred + beta * dir_predcorr)
        @. point.x += stepper.x_dir
        @. point.y += stepper.y_dir
    end

    calc_mu(solver)

    if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
        @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
        solver.status = :NumericalFailure
        return false
    end

    return true
end

# # stepper alternating predict / center steps
# function step(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
#     cones = solver.model.cones
#     point = solver.point
#     timer = solver.timer
#
#     # TODO remove the need for this updating here - should be done in line search (some instances failing without it though)
#     rtmu = sqrt(solver.mu)
#     irtmu = inv(rtmu)
#     Cones.load_point.(cones, point.primal_views, irtmu)
#     Cones.load_dual_point.(cones, point.dual_views)
#     Cones.reset_data.(cones)
#     @assert all(Cones.is_feas.(cones))
#     Cones.grad.(cones)
#     Cones.hess.(cones)
#
#     update_lhs(solver.system_solver, solver)
#
#     # use_corr = true
#     use_corr = false
#
#     # TODO if use NT, only need nonsymm cones in nbhd
#     if all(Cones.in_neighborhood.(cones, rtmu, T(0.05)))
#         # predict
#         update_rhs_pred(stepper, solver)
#         get_directions(stepper, solver, true, iter_ref_steps = 3)
#         dir_pred = copy(stepper.dir) # TODO
#         if use_corr
#             update_rhs_predcorr(stepper, solver)
#             get_directions(stepper, solver, true, iter_ref_steps = 3)
#             dir_predcorr = copy(stepper.dir) # TODO
#         end
#         pred = true
#         stepper.prev_gamma = zero(T) # TODO print like "pred" in column, or "cent" otherwise
#     else
#         # center
#         update_rhs_cent(stepper, solver)
#         get_directions(stepper, solver, false, iter_ref_steps = 3)
#         dir_cent = copy(stepper.dir) # TODO
#         if use_corr
#             update_rhs_centcorr(stepper, solver)
#             get_directions(stepper, solver, false, iter_ref_steps = 3)
#             dir_centcorr = copy(stepper.dir) # TODO
#         end
#         pred = false
#         stepper.prev_gamma = one(T)
#     end
#
#     # alpha step length
#     alpha = find_max_alpha(stepper, solver, prev_alpha = stepper.prev_alpha, min_alpha = T(1e-3), max_nbhd = T(0.99))
#     # @show alpha
#     !pred && alpha < 0.98 && println(alpha)
#     if iszero(alpha)
#         @warn("very small alpha")
#         solver.status = :NumericalFailure
#         return false
#     end
#     stepper.prev_alpha = alpha
#     if pred
#         stepper.prev_pred_alpha = alpha
#     end
#
#     # step
#     @. point.x += alpha * stepper.x_dir
#     @. point.y += alpha * stepper.y_dir
#     @. point.z += alpha * stepper.z_dir
#     @. point.s += alpha * stepper.s_dir
#     solver.tau += alpha * stepper.dir[stepper.tau_row]
#     solver.kap += alpha * stepper.dir[stepper.kap_row]
#     calc_mu(solver)
#
#     if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
#         @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
#         solver.status = :NumericalFailure
#         return false
#     end
#
#     return true
# end

# update the RHS for pred direction
function update_rhs_pred(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs

    # x, y, z, tau
    stepper.x_rhs .= solver.x_residual
    stepper.y_rhs .= solver.y_residual
    stepper.z_rhs .= solver.z_residual
    rhs[stepper.tau_row] = solver.kap + solver.primal_obj_t - solver.dual_obj_t

    # s
    for k in eachindex(solver.model.cones)
        @. stepper.s_rhs_k[k] = -solver.point.dual_views[k]
    end

    # kap
    rhs[end] = -solver.kap

    return rhs
end

# update the prediction RHS with a correction
function update_rhs_predcorr(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs
    rhs .= 0

    # s
    irtrtmu = inv(sqrt(sqrt(solver.mu))) # TODO or mu^-0.25
    for (k, cone_k) in enumerate(solver.model.cones)
        Cones.use_correction(cone_k) || continue
        # TODO avoid allocs
        prim_dir_k = stepper.primal_dir_k[k]
        H_prim_dir_k = Cones.hess_prod!(similar(prim_dir_k), prim_dir_k, cone_k)
        prim_k_scal = irtrtmu * prim_dir_k
        corr_k = Cones.correction(cone_k, prim_k_scal)
        corr_point = dot(corr_k, cone_k.point)
        @assert !isnan(corr_point)
        corr_viol = abs(corr_point - irtrtmu * dot(prim_k_scal, H_prim_dir_k)) / abs(corr_point + 10eps(T))
        @assert !isnan(corr_viol)
        # if corr_point < eps(T)
        #     @show "pred ", corr_point
        # end
        if corr_viol < 0.001
            @. stepper.s_rhs_k[k] += H_prim_dir_k + corr_k
        # else
        #     println("skip pred-corr: $corr_viol")
        end
    end

    # TODO NT way:
    rhs[end] -= stepper.dir[stepper.tau_row] * stepper.dir[stepper.kap_row] / solver.tau
    # TODO SY way:
    # tau_dir_tau = stepper.dir[stepper.tau_row] / solver.tau
    # rhs[end] += solver.mu / solver.tau * tau_dir_tau * (1 + tau_dir_tau)

    return rhs
end

# update the RHS for centering direction
function update_rhs_cent(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs

    # x, y, z, tau
    stepper.rhs[1:stepper.tau_row] .= 0

    # s
    rtmu = sqrt(solver.mu)
    for (k, cone_k) in enumerate(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. stepper.s_rhs_k[k] = -duals_k - rtmu * grad_k
    end

    # kap
    rhs[end] = -solver.kap + solver.mu / solver.tau

    return rhs
end

# update the centering RHS with a correction
function update_rhs_centcorr(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs
    rhs .= 0

    # s
    irtrtmu = inv(sqrt(sqrt(solver.mu)))
    for (k, cone_k) in enumerate(solver.model.cones)
        Cones.use_correction(cone_k) || continue
        # TODO avoid allocs
        prim_dir_k = stepper.primal_dir_k[k]
        prim_k_scal = irtrtmu * prim_dir_k
        H_prim_dir_k_scal = Cones.hess_prod!(similar(prim_dir_k), prim_k_scal, cone_k)
        corr_k = Cones.correction(cone_k, prim_k_scal)
        corr_point = dot(corr_k, cone_k.point)
        @assert !isnan(corr_point)
        corr_viol = abs(corr_point - dot(prim_k_scal, H_prim_dir_k_scal)) / abs(corr_point + 10eps(T))
        @assert !isnan(corr_viol)
        # if corr_point < eps(T)
        #     @show "cent ", corr_point
        # end
        if corr_viol < 0.001
            stepper.s_rhs_k[k] .+= corr_k
        # else
        #     println("skip cent-corr: $corr_viol")
        end
    end

    # TODO NT way:
    # rhs[end] -= stepper.dir[stepper.tau_row] * stepper.dir[stepper.kap_row] / solver.tau
    # TODO SY way:
    tau_dir_tau = stepper.dir[stepper.tau_row] / solver.tau
    rhs[end] += solver.mu / solver.tau * tau_dir_tau * tau_dir_tau

    return rhs
end

# calculate direction given rhs, and apply iterative refinement
function get_directions(stepper::CombinedStepper{T}, solver::Solver{T}, use_nt::Bool; iter_ref_steps::Int = 0) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    dir_temp = stepper.dir_temp
    res = stepper.res
    system_solver = solver.system_solver
    timer = solver.timer

    tau_scal = (use_nt ? solver.kap : solver.mu / solver.tau) / solver.tau

    solve_system(system_solver, solver, dir, rhs, tau_scal)

    # use iterative refinement
    copyto!(dir_temp, dir)
    res = apply_lhs(stepper, solver, tau_scal) # modifies res
    res .-= rhs
    norm_inf = norm(res, Inf)
    norm_2 = norm(res, 2)
    # @show res

    for i in 1:iter_ref_steps
        # @show norm_inf
        if norm_inf < 100 * eps(T) # TODO change tolerance dynamically
            break
        end
        solve_system(system_solver, solver, dir, res, tau_scal)
        axpby!(true, dir_temp, -1, dir)
        res = apply_lhs(stepper, solver, tau_scal) # modifies res
        res .-= rhs
        # @show res

        norm_inf_new = norm(res, Inf)
        norm_2_new = norm(res, 2)
        if norm_inf_new > norm_inf || norm_2_new > norm_2
            # residual has not improved
            copyto!(dir, dir_temp)
            break
        end

        # residual has improved, so use the iterative refinement
        # TODO only print if using debug mode
        # solver.verbose && @printf("iter ref round %d norms: inf %9.2e to %9.2e, two %9.2e to %9.2e\n", i, norm_inf, norm_inf_new, norm_2, norm_2_new)
        copyto!(dir_temp, dir)
        norm_inf = norm_inf_new
        norm_2 = norm_2_new
    end

    @assert !isnan(norm_inf)
    if norm_inf > 1e-4
        println("residual on direction too large: $norm_inf")
    end

    return dir
end

# calculate residual on 6x6 linear system
function apply_lhs(stepper::CombinedStepper{T}, solver::Solver{T}, tau_scal::T) where {T <: Real}
    model = solver.model
    tau_dir = stepper.dir[stepper.tau_row]
    kap_dir = stepper.dir[stepper.kap_row]

    # A'*y + G'*z + c*tau
    copyto!(stepper.x_res, model.c)
    mul!(stepper.x_res, model.G', stepper.z_dir, true, tau_dir)
    # -G*x + h*tau - s
    @. stepper.z_res = model.h * tau_dir - stepper.s_dir
    mul!(stepper.z_res, model.G, stepper.x_dir, -1, true)
    # -c'*x - b'*y - h'*z - kap
    stepper.res[stepper.tau_row] = -dot(model.c, stepper.x_dir) - dot(model.h, stepper.z_dir) - kap_dir
    # if p = 0, ignore A, b, y
    if !iszero(model.p)
        # A'*y + G'*z + c*tau
        mul!(stepper.x_res, model.A', stepper.y_dir, true, true)
        # -A*x + b*tau
        copyto!(stepper.y_res, model.b)
        mul!(stepper.y_res, model.A, stepper.x_dir, -1, tau_dir)
        # -c'*x - b'*y - h'*z - kap
        stepper.res[stepper.tau_row] -= dot(model.b, stepper.y_dir)
    end

    # s
    for (k, cone_k) in enumerate(model.cones)
        # (du bar) mu*H_k*z_k + s_k
        # (pr bar) z_k + mu*H_k*s_k
        s_res_k = stepper.s_res_k[k]
        Cones.hess_prod!(s_res_k, stepper.primal_dir_k[k], cone_k)
        @. s_res_k += stepper.dual_dir_k[k]
    end

    stepper.res[stepper.kap_row] = tau_scal * tau_dir + kap_dir

    return stepper.res
end

# # backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
# function find_max_alpha(
#     stepper::CombinedStepper{T},
#     solver::Solver{T};
#     prev_alpha::T,
#     min_alpha::T,
#     ) where {T <: Real}
#     cones = solver.model.cones
#     cone_times = stepper.cone_times
#     cone_order = stepper.cone_order
#     z = solver.point.z
#     s = solver.point.s
#     tau = solver.tau
#     kap = solver.kap
#     z_dir = stepper.z_dir
#     s_dir = stepper.s_dir
#     tau_dir = stepper.dir[stepper.tau_row]
#     kap_dir = stepper.dir[stepper.kap_row]
#     z_ls = stepper.z_ls
#     s_ls = stepper.s_ls
#     primals_ls = stepper.primal_views_ls
#     duals_ls = stepper.dual_views_ls
#     timer = solver.timer
#
#     alpha = max(T(0.1), min(prev_alpha * T(1.4), one(T))) # TODO option for parameter
#     if tau_dir < zero(T)
#         alpha = min(alpha, -tau / tau_dir)
#     end
#     if kap_dir < zero(T)
#         alpha = min(alpha, -kap / kap_dir)
#     end
#     alpha *= T(0.9999)
#
#     nup1 = solver.model.nu + 1
#     while true
#         in_nbhd = true
#
#         @. z_ls = z + alpha * z_dir
#         @. s_ls = s + alpha * s_dir
#         dot_s_z = zero(T)
#         for k in cone_order
#             dot_s_z_k = dot(primals_ls[k], duals_ls[k])
#             if dot_s_z_k < eps(T)
#                 in_nbhd = false
#                 break
#             end
#             dot_s_z += dot_s_z_k
#         end
#
#         if in_nbhd
#             taukap_temp = (tau + alpha * tau_dir) * (kap + alpha * kap_dir)
#             mu_temp = (dot_s_z + taukap_temp) / nup1
#
#             if mu_temp > eps(T) && abs(taukap_temp - mu_temp) < mu_temp * solver.max_nbhd
#                 # order the cones by how long it takes to check neighborhood condition and iterate in that order, to improve efficiency
#                 sortperm!(cone_order, cone_times, initialized = true)
#
#                 irt_mu_temp = inv(sqrt(mu_temp))
#                 for k in cone_order
#                     cone_k = cones[k]
#                     time_k = time_ns()
#                     Cones.load_point(cone_k, primals_ls[k], irt_mu_temp)
#                     Cones.load_dual_point(cone_k, duals_ls[k])
#                     Cones.reset_data(cone_k)
#                     in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && Cones.in_neighborhood(cone_k, mu_temp))
#                     cone_times[k] = time_ns() - time_k
#
#                     if !in_nbhd_k
#                         in_nbhd = false
#                         break
#                     end
#                 end
#
#                 if in_nbhd
#                     break
#                 end
#             end
#         end
#
#         if alpha < min_alpha
#             # alpha is very small so finish
#             alpha = zero(T)
#             break
#         end
#
#         # iterate is outside the neighborhood: decrease alpha
#         alpha *= T(0.9) # TODO option for parameter
#     end
#
#     return alpha
# end

# backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
function find_max_alpha(
    stepper::CombinedStepper{T},
    solver::Solver{T}; # TODO remove if not using
    prev_alpha::T,
    min_alpha::T,
    min_nbhd::T = T(0.01),
    # max_nbhd::T = one(T),
    max_nbhd::T = T(0.99),
    ) where {T <: Real}
    cones = solver.model.cones
    cone_times = stepper.cone_times
    cone_order = stepper.cone_order
    z = solver.point.z
    s = solver.point.s
    tau = solver.tau
    kap = solver.kap
    z_dir = stepper.z_dir
    s_dir = stepper.s_dir
    tau_dir = stepper.dir[stepper.tau_row]
    kap_dir = stepper.dir[stepper.kap_row]
    z_ls = stepper.z_ls
    s_ls = stepper.s_ls
    primals_ls = stepper.primal_views_ls
    duals_ls = stepper.dual_views_ls

    alpha_reduce = T(0.95) # TODO tune, maybe try smaller for pred_alpha since heuristic
    nup1 = solver.model.nu + 1
    sz_ks = zeros(T, length(cone_order)) # TODO prealloc

    # TODO experiment with starting alpha (<1)
    # alpha = one(T)
    alpha = max(T(0.1), min(prev_alpha * T(1.4), one(T))) # TODO option for parameter

    if tau_dir < zero(T)
        alpha = min(alpha, -tau / tau_dir)
    end
    if kap_dir < zero(T)
        alpha = min(alpha, -kap / kap_dir)
    end
    alpha *= T(0.9999)

    alpha /= alpha_reduce
    # TODO for feas, as soon as cone is feas, don't test feas again, since line search is backwards
    while true
        if alpha < min_alpha
            # alpha is very small so finish
            alpha = zero(T)
            break
        end
        alpha *= alpha_reduce

        taukap_ls = (tau + alpha * tau_dir) * (kap + alpha * kap_dir)
        (taukap_ls < eps(T)) && continue

        # order the cones by how long it takes to check neighborhood condition and iterate in that order, to improve efficiency
        sortperm!(cone_order, cone_times, initialized = true) # NOTE stochastic

        @. z_ls = z + alpha * z_dir
        @. s_ls = s + alpha * s_dir

        for k in cone_order
            sz_ks[k] = dot(primals_ls[k], duals_ls[k])
        end
        any(<(eps(T)), sz_ks) && continue

        mu_ls = (sum(sz_ks) + taukap_ls) / nup1
        (mu_ls < eps(T)) && continue

        min_nbhd_mu = min_nbhd * mu_ls
        (taukap_ls < min_nbhd_mu) && continue
        any(sz_ks[k] < min_nbhd_mu * Cones.get_nu(cones[k]) for k in cone_order) && continue

        # TODO experiment with SY nbhd for tau-kappa
        isfinite(max_nbhd) && (abs(taukap_ls - mu_ls) > max_nbhd * mu_ls) && continue

        rtmu = sqrt(mu_ls)
        irtmu = inv(rtmu)
        in_nbhd = true
        for k in cone_order
            cone_k = cones[k]
            time_k = time_ns()

            Cones.load_point(cone_k, primals_ls[k], irtmu)
            Cones.load_dual_point(cone_k, duals_ls[k])
            Cones.reset_data(cone_k)

            in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && Cones.in_neighborhood(cone_k, rtmu, max_nbhd))
            # in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && (isinf(max_nbhd) || Cones.in_neighborhood(cone_k, rtmu, max_nbhd)))
            # TODO is_dual_feas function should fall back to a nbhd-like check (for ray maybe) if not using nbhd check
            # in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k))

            cone_times[k] = time_ns() - time_k
            if !in_nbhd_k
                in_nbhd = false
                break
            end
        end
        in_nbhd && break
    end

    return alpha
end

function print_iteration_stats(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    if iszero(solver.num_iters)
        if iszero(solver.model.p)
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
                "iter", "p_obj", "d_obj", "rel_gap", "abs_gap",
                "x_feas", "z_feas", "tau", "kap", "mu",
                "gamma", "alpha",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.z_feas, solver.tau, solver.kap, solver.mu
                )
        else
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
                "iter", "p_obj", "d_obj", "rel_gap", "abs_gap",
                "x_feas", "y_feas", "z_feas", "tau", "kap", "mu",
                "gamma", "alpha",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu
                )
        end
    else
        if iszero(solver.model.p)
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.z_feas, solver.tau, solver.kap, solver.mu,
                stepper.prev_gamma, stepper.prev_alpha,
                )
        else
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu,
                stepper.prev_gamma, stepper.prev_alpha,
                )
        end
    end
    flush(stdout)
    return
end
