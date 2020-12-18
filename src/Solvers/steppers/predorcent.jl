#=
predict or center stepper
=#

mutable struct PredOrCentStepper{T <: Real} <: Stepper{T}
    use_correction::Bool
    prev_pred_alpha::T
    prev_alpha::T
    prev_is_pred::Bool
    cent_count::Int

    rhs::Point{T}
    dir::Point{T}
    res::Point{T}
    dir_temp::Vector{T}

    line_searcher::LineSearcher{T}

    function PredOrCentStepper{T}(;
        use_correction::Bool = true,
        ) where {T <: Real}
        stepper = new{T}()
        stepper.use_correction = use_correction
        return stepper
    end
end

# create the stepper cache
function load(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
    stepper.prev_pred_alpha = one(T)
    stepper.prev_is_pred = false
    stepper.prev_alpha = one(T)
    stepper.cent_count = 0

    model = solver.model
    stepper.rhs = Point(model)
    stepper.dir = Point(model)
    stepper.res = Point(model)
    stepper.dir_temp = zeros(T, length(stepper.rhs.vec))

    stepper.line_searcher = LineSearcher{T}(model)

    return stepper
end

function step(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    model = solver.model
    rhs = stepper.rhs
    dir = stepper.dir

    dir_nocorr = similar(dir.vec) # TODO
    dir_corr = similar(dir.vec) # TODO

    # update linear system solver factorization
    update_lhs(solver.system_solver, solver)

    stepper.prev_is_pred = (stepper.cent_count > 3) || all(Cones.in_neighborhood.(model.cones, sqrt(solver.mu), T(0.05)))
    # TODO refac?
    # TODO remove add option from update rhs funcs
    if stepper.prev_is_pred
        # predict
        stepper.cent_count = 0

        # update_rhs_pred(solver, rhs)
        # get_directions(stepper, solver, true, iter_ref_steps = 3)
        # if stepper.use_correction
        #     update_rhs_predcorr(solver, rhs, dir)
        #     get_directions(stepper, solver, true, iter_ref_steps = 3)
        # end

        # TODO make correction optional

        # get pred and predcorr directions
        update_rhs_pred(solver, rhs)
        get_directions(stepper, solver, true, iter_ref_steps = 3)
        copyto!(dir_nocorr, dir.vec)
        update_rhs_predcorr(solver, rhs, dir, add = false)
        get_directions(stepper, solver, true, iter_ref_steps = 3)
        copyto!(dir_corr, dir.vec)

        # get prediction alpha
        copyto!(dir.vec, dir_nocorr)
        # TODO try max_nbhd = Inf, but careful of cones with no dual feas check
        alpha_nocorr = find_max_alpha(point, dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3)) # TODO change prev_alpha
        # @show alpha_nocorr

        # get corrected prediction direction
        @. dir.vec = dir_nocorr + alpha_nocorr * dir_corr
    else
        # center
        stepper.cent_count += 1

        # update_rhs_cent(solver, rhs)
        # get_directions(stepper, solver, false, iter_ref_steps = 3)
        # if stepper.use_correction
        #     update_rhs_centcorr(solver, rhs, dir)
        #     get_directions(stepper, solver, false, iter_ref_steps = 3)
        # end

        # TODO make correction optional

        # get cent and centcorr directions
        update_rhs_cent(solver, rhs)
        get_directions(stepper, solver, false, iter_ref_steps = 3)
        copyto!(dir_nocorr, dir.vec)
        update_rhs_centcorr(solver, rhs, dir, add = false)
        get_directions(stepper, solver, false, iter_ref_steps = 3)
        copyto!(dir_corr, dir.vec)

        # get centering alpha
        copyto!(dir.vec, dir_nocorr)
        # TODO try max_nbhd = Inf, but careful of cones with no dual feas check
        alpha_nocorr = find_max_alpha(point, dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3)) # TODO change prev_alpha
        # @show alpha_nocorr

        # get corrected centering direction
        @. dir.vec = dir_nocorr + alpha_nocorr * dir_corr
    end

    # alpha step length
    stepper.prev_alpha = alpha = find_max_alpha(point, dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3), max_nbhd = T(0.99)) # TODO change prev_alpha

    if !stepper.prev_is_pred && alpha < 0.99
        @warn("didn't step 1 for centering")
    end
    if iszero(alpha)
        # TODO attempt recovery
        @warn("very small alpha")
        solver.status = NumericalFailure
        return false
    end
    if stepper.prev_is_pred
        stepper.prev_pred_alpha = alpha
    end

    # step
    @. point.vec += alpha * dir.vec
    calc_mu(solver)

    return true
end

expect_improvement(stepper::PredOrCentStepper) = stepper.prev_is_pred

function print_iteration_stats(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
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
        step = (stepper.prev_is_pred ? "pred" : "cent")
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

# # stepper using line search between cent and pred points
# function step(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
#     point = solver.point
#
#     # # TODO remove the need for this updating here - should be done in line search (some instances failing without it though)
#     # rtmu = sqrt(solver.mu)
#     # irtmu = inv(rtmu)
#     # cones = solver.model.cones
#     # Cones.load_point.(cones, point.primal_views, irtmu)
#     # Cones.load_dual_point.(cones, point.dual_views)
#     # Cones.reset_data.(cones)
#     # @assert all(Cones.is_feas.(cones))
#     # Cones.grad.(cones)
#     # Cones.hess.(cones)
#     # # # @assert all(Cones.in_neighborhood.(cones, rtmu, T(0.7)))
#
#     # update linear system solver factorization and helpers
#     Cones.grad.(solver.model.cones)
#     update_lhs(solver.system_solver, solver)
#
#     z_dir = stepper.dir.z
#     s_dir = stepper.dir.s
#
#     # calculate centering direction and keep in dir_cent
#     update_rhs_cent(stepper, solver)
#     get_directions(stepper, solver, false, iter_ref_steps = 3)
#     dir_cent = copy(stepper.dir) # TODO
#     tau_cent = stepper.dir[stepper.tau_row]
#     kap_cent = stepper.dir[stepper.kap_row]
#     z_cent = copy(z_dir)
#     s_cent = copy(s_dir)
#
#     update_rhs_centcorr(stepper, solver)
#     get_directions(stepper, solver, false, iter_ref_steps = 3)
#     dir_centcorr = copy(stepper.dir) # TODO
#     tau_centcorr = stepper.dir[stepper.tau_row]
#     kap_centcorr = stepper.dir[stepper.kap_row]
#     z_centcorr = copy(z_dir)
#     s_centcorr = copy(s_dir)
#
#     # calculate affine/prediction direction and keep in dir
#     update_rhs_pred(stepper, solver)
#     get_directions(stepper, solver, true, iter_ref_steps = 3)
#     dir_pred = copy(stepper.dir) # TODO
#     tau_pred = stepper.dir[stepper.tau_row]
#     kap_pred = stepper.dir[stepper.kap_row]
#     z_pred = copy(z_dir)
#     s_pred = copy(s_dir)
#
#     update_rhs_predcorr(stepper, solver)
#     get_directions(stepper, solver, true, iter_ref_steps = 3)
#     dir_predcorr = copy(stepper.dir) # TODO
#     tau_predcorr = stepper.dir[stepper.tau_row]
#     kap_predcorr = stepper.dir[stepper.kap_row]
#     z_predcorr = copy(z_dir)
#     s_predcorr = copy(s_dir)
#
#     # TODO check cent point (step 1) is acceptable
#     @. stepper.dir = dir_cent + dir_centcorr
#     alpha = find_max_alpha(stepper, solver, prev_alpha = one(T), min_alpha = T(0.1)) # TODO only check end point alpha = 1 maybe
#     # TODO cleanup
#     if alpha < 0.99
#         # @show alpha
#         @warn("could not do full step in centering-correction direction ($alpha)")
#         dir_centcorr .= 0
#         stepper.dir .= 0
#         tau_centcorr = stepper.dir[stepper.tau_row]
#         kap_centcorr = stepper.dir[stepper.kap_row]
#         z_centcorr = copy(z_dir)
#         s_centcorr = copy(s_dir)
#
#         @. stepper.dir = dir_cent + dir_centcorr
#         alpha = find_max_alpha(stepper, solver, prev_alpha = one(T), min_alpha = T(0.1))
#         @show alpha
#     end
#
#     # TODO use a smarter line search, eg bisection
#     # TODO start with beta = 0.9999 for pred factor, and decrease until point satisfies nbhd
#     cones = solver.model.cones
#     cone_times = stepper.cone_times
#     cone_order = stepper.cone_order
#     z = solver.point.z
#     s = solver.point.s
#     tau = solver.point.tau
#     kap = solver.point.kap
#     # z_dir = stepper.dir.z
#     # s_dir = stepper.dir.s
#     # tau_dir = stepper.dir[stepper.tau_row]
#     # kap_dir = stepper.dir[stepper.kap_row]
#     z_ls = stepper.z_ls
#     s_ls = stepper.s_ls
#     primals_ls = stepper.primal_views_ls
#     duals_ls = stepper.dual_views_ls
#     sz_ks = zeros(T, length(cone_order)) # TODO prealloc
#     tau_ls = zero(T)
#     kap_ls = zero(T)
#
#     nup1 = solver.model.nu + 1
#     # max_nbhd = T(0.99)
#     # max_nbhd = T(0.9)
#     max_nbhd = one(T)
#     min_nbhd = T(0.01)
#
#     # beta_schedule = T[0.9999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
#     # beta = zero(T)
#     beta = max(T(0.1), min(stepper.prev_gamma * T(1.4), one(T))) # TODO option for parameter
#     beta_decrease = T(0.95)
#     beta *= T(0.9999)
#     beta /= beta_decrease
#
#     iter_ls = 0
#     in_nbhd = false
#
#     # while iter_ls < length(beta_schedule)
#     while beta > 0
#         in_nbhd = false
#         iter_ls += 1
#
#         # beta = beta_schedule[iter_ls]
#         beta *= beta_decrease
#         if beta < T(0.01)
#             beta = zero(T) # pure centering
#         end
#         betam1 = 1 - beta
#
#         # TODO shouldn't need to reduce corr on cent?
#         tau_ls = tau + betam1 * (tau_cent + betam1 * tau_centcorr) + beta * (tau_pred + beta * tau_predcorr)
#         kap_ls = kap + betam1 * (kap_cent + betam1 * kap_centcorr) + beta * (kap_pred + beta * kap_predcorr)
#         taukap_ls = tau_ls * kap_ls
#         (tau_ls < eps(T) || kap_ls < eps(T) || taukap_ls < eps(T)) && continue
#
#         # order the cones by how long it takes to check neighborhood condition and iterate in that order, to improve efficiency
#         sortperm!(cone_order, cone_times, initialized = true) # NOTE stochastic
#
#         @. z_ls = z + betam1 * (z_cent + betam1 * z_centcorr) + beta * (z_pred + beta * z_predcorr)
#         @. s_ls = s + betam1 * (s_cent + betam1 * s_centcorr) + beta * (s_pred + beta * s_predcorr)
#
#         for k in cone_order
#             sz_ks[k] = dot(primals_ls[k], duals_ls[k])
#         end
#         any(<(eps(T)), sz_ks) && continue
#
#         mu_ls = (sum(sz_ks) + taukap_ls) / nup1
#         (mu_ls < eps(T)) && continue
#
#         # TODO experiment with SY nbhd for tau-kappa
#         (abs(taukap_ls - mu_ls) > max_nbhd * mu_ls) && continue
#
#         min_nbhd_mu = min_nbhd * mu_ls
#         (taukap_ls < min_nbhd_mu) && continue
#         any(sz_ks[k] < min_nbhd_mu * Cones.get_nu(cones[k]) for k in cone_order) && continue
#
#         rtmu = sqrt(mu_ls)
#         irtmu = inv(rtmu)
#         in_nbhd = true
#         for k in cone_order
#             cone_k = cones[k]
#             time_k = time_ns()
#
#             Cones.load_point(cone_k, primals_ls[k], irtmu)
#             Cones.load_dual_point(cone_k, duals_ls[k])
#             Cones.reset_data(cone_k)
#
#             in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && Cones.in_neighborhood(cone_k, rtmu, max_nbhd))
#             # in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && (isinf(max_nbhd) || Cones.in_neighborhood(cone_k, rtmu, max_nbhd)))
#             # TODO is_dual_feas function should fall back to a nbhd-like check (for ray maybe) if not using nbhd check
#             # in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k))
#
#             cone_times[k] = time_ns() - time_k
#             if !in_nbhd_k
#                 in_nbhd = false
#                 break
#             end
#         end
#         in_nbhd && break
#         iszero(beta) && break
#     end
#     # @show iter_ls
#
#     # TODO if zero not feasible, do backwards line search
#     if !in_nbhd
#         @show beta
#
#         copyto!(stepper.dir, dir_cent)
#
#         alpha = find_max_alpha(stepper, solver, prev_alpha = one(T), min_alpha = T(1e-3))
#         if iszero(alpha)
#             @warn("numerical failure: could not step in centering direction; terminating")
#             solver.status = NumericalFailure
#             return false
#         end
#         stepper.prev_alpha = alpha
#
#         # step distance alpha in combined direction
#         @. point.x += alpha * stepper.dir.x
#         @. point.y += alpha * stepper.dir.y
#         @. point.z += alpha * stepper.dir.z
#         @. point.s += alpha * stepper.dir.s
#         solver.point.tau += alpha * stepper.dir[stepper.tau_row]
#         solver.point.kap += alpha * stepper.dir[stepper.kap_row]
#     else
#         stepper.prev_gamma = gamma = beta # TODO
#
#         # step to combined point
#         copyto!(point.z, z_ls)
#         copyto!(point.s, s_ls)
#         solver.point.tau = tau_ls
#         solver.point.kap = kap_ls
#
#         # TODO improve
#         betam1 = 1 - beta
#         @. stepper.dir = betam1 * (dir_cent + betam1 * dir_centcorr) # TODO shouldn't need to reduce corr on cent
#         # @. stepper.dir = betam1 * (dir_cent + dir_centcorr)
#         @. point.x += stepper.dir.x
#         @. point.y += stepper.dir.y
#         @. stepper.dir = beta * (dir_pred + beta * dir_predcorr)
#         @. point.x += stepper.dir.x
#         @. point.y += stepper.dir.y
#     end
#
#     calc_mu(solver)
#
#     return true
# end
