#=
Copyright 2019, Chris Coey and contributors

interior point stepping routines for algorithms based on homogeneous self dual embedding
=#

mutable struct CombinedStepper{T <: Real} <: Stepper{T}
    prev_pred_prim_dist::T
    prev_pred_dual_dist::T
    prev_comb_prim_dist::T
    prev_comb_dual_dist::T
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
    dir_corr::Vector{T}
    res::Vector{T}
    x_res
    y_res
    z_res
    s_res
    s_res_k::Vector
    tau_row::Int
    kap_row::Int
    z_linesearch::Vector{T}
    s_linesearch::Vector{T}
    primal_views_linesearch::Vector
    dual_views_linesearch::Vector
    cone_times::Vector{Float64}
    cone_order::Vector{Int}

    corr_tk::T

    CombinedStepper{T}() where {T <: Real} = new{T}()
end

# create the stepper cache
function load(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    stepper.prev_pred_prim_dist = one(T)
    stepper.prev_pred_dual_dist = one(T)
    stepper.prev_comb_prim_dist = one(T)
    stepper.prev_comb_dual_dist = one(T)
    stepper.prev_gamma = one(T)

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
    stepper.dir_corr = zeros(T, dim)
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

    stepper.z_linesearch = zeros(T, q)
    stepper.s_linesearch = zeros(T, q)
    stepper.primal_views_linesearch = [view(Cones.use_dual_barrier(model.cones[k]) ? stepper.z_linesearch : stepper.s_linesearch, model.cone_idxs[k]) for k in eachindex(model.cones)]
    stepper.dual_views_linesearch = [view(Cones.use_dual_barrier(model.cones[k]) ? stepper.s_linesearch : stepper.z_linesearch, model.cone_idxs[k]) for k in eachindex(model.cones)]

    stepper.cone_times = zeros(Float64, length(solver.model.cones))
    stepper.cone_order = collect(1:length(solver.model.cones))

    return stepper
end

function step(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    cones = solver.model.cones
    point = solver.point
    timer = solver.timer

    rtmu = sqrt(solver.mu)
    irtmu = inv(rtmu)
    @assert irtmu >= one(T)
    Cones.load_point.(cones, point.primal_views)
    Cones.rescale_point.(cones, irtmu)
    Cones.load_dual_point.(cones, point.dual_views)
    Cones.reset_data.(cones)
    @assert all(Cones.is_feas.(cones))
    Cones.grad.(cones)
    Cones.hess.(cones)

    # update linear system solver factorization and helpers
    update_lhs(solver.system_solver, solver)

    # calculate affine/prediction direction and keep in dir
    update_rhs_aff(stepper, solver)
    get_directions(stepper, solver, iter_ref_steps = 3)

    # update_rhs_pred(stepper, solver, stepper.prev_pred_prim_dist, stepper.prev_pred_dual_dist) # TODO maybe prev dists not ideal
    alpha = min(stepper.prev_pred_prim_dist, stepper.prev_pred_dual_dist)
    update_rhs_pred(stepper, solver, alpha, alpha) # TODO maybe prev dists not ideal
    get_directions(stepper, solver, iter_ref_steps = 3)

    # get alpha for affine direction
    # stepper.prev_aff_alpha = aff_alpha = find_max_dists(stepper, solver, true, prev_alpha = stepper.prev_aff_alpha, min_dist = T(1e-2))
    (pred_prim_dist, pred_dual_dist) = find_max_dists(stepper, solver, false, min_dist = T(1e-2))
    stepper.prev_pred_prim_dist = pred_prim_dist
    stepper.prev_pred_dual_dist = pred_dual_dist

    # calculate correction factor gamma
    # stepper.prev_gamma = gamma = (one(T) - aff_alpha) * min(abs2(one(T) - aff_alpha), T(0.25))
    # stepper.prev_gamma = gamma = (one(T) - aff_alpha)^2
    # stepper.prev_gamma = gamma = max(0.1, (1 - pred_prim_dist) * (1 - pred_dual_dist))
    # stepper.prev_gamma = gamma = (1 - pred_prim_dist) * (1 - pred_dual_dist)
    # stepper.prev_gamma = gamma = (1 - min(pred_prim_dist, pred_dual_dist))^2

    alpha = min(stepper.prev_pred_prim_dist, stepper.prev_pred_dual_dist)
    pred_prim_dist = pred_dual_dist = alpha
    stepper.prev_gamma = gamma = (one(T) - alpha)^2

    # TODO have to reload point after affine alpha line search
    Cones.load_point.(cones, point.primal_views)
    Cones.rescale_point.(cones, irtmu)
    Cones.load_dual_point.(cones, point.dual_views)
    Cones.reset_data.(cones)
    @assert all(Cones.is_feas.(cones))
    Cones.grad.(cones)
    Cones.hess.(cones)

    # calculate final direction and keep in dir
    update_rhs_final(stepper, solver, gamma, pred_prim_dist, pred_dual_dist)
    get_directions(stepper, solver, iter_ref_steps = 5)

    # find distance alpha for stepping in combined direction
    (comb_prim_dist, comb_dual_dist) = find_max_dists(stepper, solver, false, min_dist = T(1e-3))
    stepper.prev_comb_prim_dist = comb_prim_dist
    stepper.prev_comb_dual_dist = comb_dual_dist
    if iszero(comb_prim_dist) || iszero(comb_dual_dist)
        @warn("very small dist: primal $comb_prim_dist, dual $comb_dual_dist")
        solver.status = :NumericalFailure
        return false
    end

    alpha = min(comb_prim_dist, comb_dual_dist)
    comb_prim_dist = comb_dual_dist = alpha

    # # TODO have to reload point after affine alpha line search
    # Cones.load_point.(cones, point.primal_views)
    # Cones.rescale_point.(cones, irtmu)
    # Cones.load_dual_point.(cones, point.dual_views)
    # Cones.reset_data.(cones)
    # @assert all(Cones.is_feas.(cones))
    # Cones.grad.(cones)
    # Cones.hess.(cones)

    # step distance alpha in combined direction
    @. point.x += comb_prim_dist * stepper.x_dir
    @. point.y += comb_dual_dist * stepper.y_dir
    @. point.z += comb_dual_dist * stepper.z_dir
    @. point.s += comb_prim_dist * stepper.s_dir
    solver.tau += comb_dual_dist * stepper.dir[stepper.tau_row]
    solver.kap += comb_prim_dist * stepper.dir[stepper.kap_row]
    calc_mu(solver)

    if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
        @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
        solver.status = :NumericalFailure
        return false
    end

    return true # step succeeded
end

# update the RHS for affine direction
function update_rhs_aff(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs

    # x, y, z, tau
    if solver.x_feas > 10 * max(solver.gap, solver.y_feas, solver.z_feas)
        stepper.x_rhs .= solver.x_residual
        stepper.y_rhs .= 0
        stepper.z_rhs .= 0
        rhs[stepper.tau_row] = 0
    else
        stepper.x_rhs .= solver.x_residual
        stepper.y_rhs .= solver.y_residual
        stepper.z_rhs .= solver.z_residual
        rhs[stepper.tau_row] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    end

    # s
    for k in eachindex(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        @. stepper.s_rhs_k[k] = -duals_k
    end

    # NT: kap
    rhs[end] = -solver.kap

    return rhs
end

# update the RHS for affine-corr direction
function update_rhs_pred(
    stepper::CombinedStepper{T},
    solver::Solver{T},
    prim_dist::T,
    dual_dist::T,
    ) where {T <: Real}
    rhs = stepper.rhs

    # x, y, z, tau
    if solver.x_feas > 10 * max(solver.gap, solver.y_feas, solver.z_feas)
        stepper.x_rhs .= solver.x_residual
        stepper.y_rhs .= 0
        stepper.z_rhs .= 0
        rhs[stepper.tau_row] = 0
    else
        stepper.x_rhs .= solver.x_residual
        stepper.y_rhs .= solver.y_residual
        stepper.z_rhs .= solver.z_residual
        rhs[stepper.tau_row] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    end

    # s
    irtmu = inv(sqrt(solver.mu))
    for (k, cone_k) in enumerate(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        @. stepper.s_rhs_k[k] = -duals_k
        if Cones.use_correction(cone_k)
            # (reuses affine direction)
            # TODO check math here for case of cone.use_dual true - should s and z be swapped then?
            scal = (cone_k isa Cones.HypoPerLog ? irtmu : one(T))
            corr_k = Cones.correction(cone_k, stepper.primal_dir_k[k], stepper.dual_dir_k[k])
            stepper.s_rhs_k[k] .-= scal * prim_dist * dual_dist * corr_k
        end
    end

    # NT: kap
    # corr_tk = stepper.dir[stepper.tau_row] * stepper.dir[stepper.kap_row] / solver.tau
    stepper.corr_tk = stepper.dir[stepper.tau_row] * stepper.dir[stepper.kap_row] / solver.tau
    rhs[end] = -solver.kap - prim_dist * dual_dist * stepper.corr_tk

    return rhs
end

# update the RHS for combined direction
function update_rhs_final(
    stepper::CombinedStepper{T},
    solver::Solver{T},
    gamma::T,
    prim_dist::T,
    dual_dist::T,
    ) where {T <: Real}
    rhs = stepper.rhs

    # x, y, z, tau
    if solver.x_feas > 10 * max(solver.gap, solver.y_feas, solver.z_feas)
        stepper.x_rhs .= solver.x_residual * (1 - gamma)
        stepper.y_rhs .= 0
        stepper.z_rhs .= 0
        rhs[stepper.tau_row] = 0
        println("only x")
    else
        stepper.x_rhs .= solver.x_residual * (1 - gamma)
        stepper.y_rhs .= solver.y_residual * (1 - gamma)
        stepper.z_rhs .= solver.z_residual * (1 - gamma)
        rhs[stepper.tau_row] = (solver.kap + solver.primal_obj_t - solver.dual_obj_t) * (1 - gamma)
    end
    #
    # # x, y, z, tau
    # stepper.x_rhs .= solver.x_residual * sqrt(1 - gamma)
    # stepper.y_rhs .= solver.y_residual * (1 - gamma)
    # stepper.z_rhs .= solver.z_residual * (1 - gamma)
    # rhs[stepper.tau_row] = (solver.kap + solver.primal_obj_t - solver.dual_obj_t) * (1 - gamma)

    # s
    rtmu = sqrt(solver.mu)
    irtmu = inv(rtmu)
    for (k, cone_k) in enumerate(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        scal = (cone_k isa Cones.HypoPerLog ? rtmu : solver.mu)
        @. stepper.s_rhs_k[k] = -duals_k - (scal * grad_k) * gamma
        if Cones.use_correction(cone_k)
            # (reuses affine direction)
            # TODO check math here for case of cone.use_dual true - should s and z be swapped then?
            # stepper.s_rhs_k[k] .-= cone_k.correction
            scal = (cone_k isa Cones.HypoPerLog ? irtmu : one(T))
            corr_k = Cones.correction(cone_k, stepper.primal_dir_k[k], stepper.dual_dir_k[k])
            # corr_k = cone_k.correction
            stepper.s_rhs_k[k] .-= scal * prim_dist * dual_dist * corr_k
            # stepper.s_rhs_k[k] .-= scal * cone_k.correction * aff_alpha^2 # TODO this is heuristicy currently and just tries to reduce the amount of correction depending on how far we actually can step. redo in math, use linearity of the third-order corrector in s_dir and z_dir
        end
    end

    # NT: kap (corrector reuses kappa/tau affine directions)
    # rhs[end] = -solver.kap + (solver.mu / solver.tau) * gamma - stepper.tkcorr
    # rhs[end] = -solver.kap + (solver.mu / solver.tau) * gamma - stepper.tkcorr * aff_alpha^2
    corr_tk = stepper.dir[stepper.tau_row] * stepper.dir[stepper.kap_row] / solver.tau
    # corr_tk = stepper.corr_tk
    rhs[end] = -solver.kap + (solver.mu / solver.tau) * gamma - prim_dist * dual_dist * corr_tk

    return rhs
end

# calculate direction given rhs, and apply iterative refinement
function get_directions(stepper::CombinedStepper{T}, solver::Solver{T}; iter_ref_steps::Int = 0) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    dir_temp = stepper.dir_temp
    res = stepper.res
    system_solver = solver.system_solver
    timer = solver.timer

    @timeit timer "solve_system" solve_system(system_solver, solver, dir, rhs)

    # use iterative refinement
    @timeit timer "iter_ref" begin
        copyto!(dir_temp, dir)
        @timeit timer "apply_lhs" res = apply_lhs(stepper, solver) # modifies res
        res .-= rhs
        norm_inf = norm(res, Inf)
        norm_2 = norm(res, 2)
        @assert !isnan(norm_inf)

        for i in 1:iter_ref_steps
            # @show norm_inf
            if norm_inf < 100 * eps(T) # TODO change tolerance dynamically
                break
            end
            @timeit timer "solve_system" solve_system(system_solver, solver, dir, res)
            axpby!(true, dir_temp, -1, dir)
            res = apply_lhs(stepper, solver) # modifies res
            res .-= rhs

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
        if norm_inf > 1e-5
            println("residual on direction too large: $norm_inf")
        end
    end

    return dir
end

# calculate residual on 6x6 linear system
function apply_lhs(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
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
        # (pr bar) z_k + mu*H_k*s_k
        # (du bar) mu*H_k*z_k + s_k
        # TODO handle dual barrier
        s_res_k = stepper.s_res_k[k]
        # @show k
        # @show norm(stepper.primal_dir_k[k])
        Cones.scal_hess_prod!(s_res_k, stepper.primal_dir_k[k], cone_k, solver.mu)
        # @show norm(s_res_k)
        # if Cones.use_scaling(cone_k)
        #     scal_hess = Cones.scal_hess(cone_k, solver.mu)
        #     mul!(s_res_k, scal_hess, stepper.primal_dir_k[k])
        # else
        #     Cones.hess_prod!(s_res_k, stepper.primal_dir_k[k], cone_k)
        #     lmul!(solver.mu, s_res_k)
        # end
        @. s_res_k += stepper.dual_dir_k[k]
    end

    # NT: kapbar / taubar * tau + kap
    stepper.res[stepper.kap_row] = solver.kap / solver.tau * tau_dir + kap_dir

    return stepper.res
end



# # backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
# function find_max_dists(
#     stepper::CombinedStepper{T},
#     solver::Solver{T},
#     affine_phase::Bool;
#     prev_alpha::T,
#     min_dist::T,
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
#     z_linesearch = stepper.z_linesearch
#     s_linesearch = stepper.s_linesearch
#     primals_linesearch = stepper.primal_views_linesearch
#     duals_linesearch = stepper.dual_views_linesearch
#     timer = solver.timer
#
#     alpha = one(T)
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
#         @. z_linesearch = z + alpha * z_dir
#         @. s_linesearch = s + alpha * s_dir
#         dot_s_z = zero(T)
#         for k in cone_order
#             dot_s_z_k = dot(primals_linesearch[k], duals_linesearch[k])
#             # TODO change back
#             if dot_s_z_k < eps(Float64)
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
#             rtmu = sqrt(mu_temp)
#             irtmu = inv(mu_temp)
#
#             # TODO for feas, as soon as cone is feas, don't test feas again, since line search is backwards
#
#             if mu_temp > eps(Float64) && (affine_phase || (taukap_temp > mu_temp * 0.5)) #&& abs(taukap_temp - mu_temp) < mu_temp * 0.1)) # TODO redundant
#             # if mu_temp > eps(T) && taukap_temp > mu_temp * 1e-4 # solver.max_nbhd
#                 # order the cones by how long it takes to check neighborhood condition and iterate in that order, to improve efficiency
#                 # sortperm!(cone_order, cone_times, initialized = true)
#
#                 for k in cone_order
#                     cone_k = cones[k]
#                     time_k = time_ns()
#                     Cones.load_point(cone_k, primals_linesearch[k])
#                     Cones.rescale_point.(cones, irtmu)
#                     Cones.load_dual_point(cone_k, duals_linesearch[k])
#                     Cones.reset_data(cone_k)
#
#                     # @show Cones.is_feas(cone_k), Cones.is_dual_feas(cone_k)
#                     # fsble_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && Cones.in_neighborhood_sy(cone_k, mu_temp))
#                     fsble_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k))
#                     if fsble_k
#                         if affine_phase
#                             in_nbhd_k = true
#                         else
#                             in_nbhd_k = (dot(primals_linesearch[k], duals_linesearch[k]) > 0.5 * mu_temp * Cones.get_nu(cone_k))
#                             # in_nbhd_k = Cones.in_neighborhood(cone_k, mu_temp)
#                             # in_nbhd_k = Cones.in_neighborhood_sy(cone_k, mu_temp)
#                         end
#                         # in_nbhd_k = (affine_phase ? true : Cones.in_neighborhood_sy(cone_k, mu_temp) && Cones.in_neighborhood(cone_k, mu_temp))
#                         # in_nbhd_k = (affine_phase ? true : Cones.in_neighborhood_sy(cone_k, mu_temp))
#                         #
#                         # in_nbhd_k = true
#                         # in_nbhd_k = !affine_phase || (dot(primals_linesearch[k], duals_linesearch[k]) / Cones.get_nu(cone_k) > mu_temp)
#                     else
#                         in_nbhd_k = false
#                     end
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
#         if alpha < min_dist
#             # alpha is very small so finish
#             alpha = zero(T)
#             break
#         end
#
#         # iterate is outside the neighborhood: decrease alpha
#         alpha *= T(0.95)
#     end
#
#     return alpha
# end


# backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
function find_max_dists(
    stepper::CombinedStepper{T},
    solver::Solver{T},
    affine_phase::Bool;
    min_dist::T,
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
    z_linesearch = stepper.z_linesearch
    s_linesearch = stepper.s_linesearch
    primals_linesearch = stepper.primal_views_linesearch
    duals_linesearch = stepper.dual_views_linesearch
    timer = solver.timer

    # TODO refac p/d
    prim_dist = dual_dist = one(T)
    if kap_dir < zero(T)
        prim_dist = min(prim_dist, -kap / kap_dir)
    end
    if tau_dir < zero(T)
        dual_dist = min(dual_dist, -tau / tau_dir)
    end
    prim_dist *= T(0.9999)
    dual_dist *= T(0.9999)

    prim_dist = dual_dist = min(prim_dist, dual_dist)
    # primal
    while true
        @. s_linesearch = s + prim_dist * s_dir

        all_feas = true
        for k in cone_order
            cone_k = cones[k]
            if Cones.use_dual_barrier(cone_k)
                Cones.load_dual_point(cone_k, duals_linesearch[k])
                Cones.reset_data(cone_k)
                feas_k = Cones.is_dual_feas(cone_k)
            else
                Cones.load_point(cone_k, primals_linesearch[k])
                Cones.reset_data(cone_k)
                feas_k = Cones.is_feas(cone_k)
            end

            if !feas_k
                all_feas = false
                break
            end
        end
        all_feas && break

        if prim_dist < min_dist
            prim_dist = zero(T)
            break
        end

        prim_dist *= T(0.95)
    end

    # dual
    while true
        @. z_linesearch = z + dual_dist * z_dir

        all_feas = true
        for k in cone_order
            cone_k = cones[k]
            if Cones.use_dual_barrier(cone_k)
                Cones.load_point(cone_k, primals_linesearch[k])
                Cones.reset_data(cone_k)
                feas_k = Cones.is_feas(cone_k)
            else
                Cones.load_dual_point(cone_k, duals_linesearch[k])
                Cones.reset_data(cone_k)
                feas_k = Cones.is_dual_feas(cone_k)
            end

            if !feas_k
                all_feas = false
                break
            end
        end
        all_feas && break

        if dual_dist < min_dist
            dual_dist = zero(T)
            break
        end

        dual_dist *= T(0.95)
    end

    if affine_phase
        return (prim_dist, dual_dist)
        # return min(prim_dist, dual_dist)
    end

    prim_dist = dual_dist = min(prim_dist, dual_dist)

    # TODO optionally: now starting from prim and dual dists, shrink both by equal ratios to satisfy nbhd condition
    # @show prim_dist
    # @show dual_dist
    nup1 = solver.model.nu + 1
    while true
        @. s_linesearch = s + prim_dist * s_dir
        @. z_linesearch = z + dual_dist * z_dir

        sz_ls = dot(primals_linesearch, duals_linesearch)
        sz_ls < 0 && println("sz in line search is $sz_ls")
        tk_ls = (tau + dual_dist * tau_dir) * (kap + prim_dist * kap_dir)
        mu_ls = (sz_ls + tk_ls) / nup1

        if (min(sz_ls, tk_ls, mu_ls) > 10eps(T)) && (tk_ls > mu_ls * 0.5)
            all_in_nbhd = true
            for k in cone_order
                cone_k = cones[k]
                sz_ls_k = dot(primals_linesearch[k], duals_linesearch[k])
                sz_ls_k < 0 && println("sz for cone in line search is $sz_ls_k")
                if (sz_ls_k < 10eps(T)) || (sz_ls_k < 0.5 * mu_ls * Cones.get_nu(cone_k))
                    all_in_nbhd = false
                    break
                end
                sz_ls += sz_ls_k
            end
            all_in_nbhd && break
        end

        if prim_dist < min_dist
            prim_dist = zero(T)
            break
        end
        if dual_dist < min_dist
            dual_dist = zero(T)
            break
        end

        prim_dist *= T(0.95)
        dual_dist *= T(0.95)
        # @. s_linesearch = s + prim_dist * s_dir
        # @. z_linesearch = z + dual_dist * z_dir
    end
    # @show prim_dist
    # @show dual_dist
    # println()

    prim_dist *= T(0.9999)
    dual_dist *= T(0.9999)

    return (prim_dist, dual_dist)
    # return min(prim_dist, dual_dist)
end

function print_iteration_stats(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    if iszero(solver.num_iters)
        if iszero(solver.model.p)
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
                "iter", "p_obj", "d_obj", "rel_gap", "abs_gap",
                "x_feas", "z_feas", "tau", "kap", "mu",
                "gamma", "d_prim", "d_dual",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.z_feas, solver.tau, solver.kap, solver.mu
                )
        else
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
                "iter", "p_obj", "d_obj", "rel_gap", "abs_gap",
                "x_feas", "y_feas", "z_feas", "tau", "kap", "mu",
                "gamma", "d_prim", "d_dual",
                )
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu
                )
        end
    else
        if iszero(solver.model.p)
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.z_feas, solver.tau, solver.kap, solver.mu,
                stepper.prev_gamma, stepper.prev_comb_prim_dist, stepper.prev_comb_dual_dist
                )
        else
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                solver.num_iters, solver.primal_obj, solver.dual_obj, solver.rel_gap, solver.gap,
                solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu,
                stepper.prev_gamma, stepper.prev_comb_prim_dist, stepper.prev_comb_dual_dist
                )
        end
    end
    flush(stdout)
    return
end
