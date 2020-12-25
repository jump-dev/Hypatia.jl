#=
predict or center stepper
=#

mutable struct PredOrCentStepper{T <: Real} <: Stepper{T}
    use_correction::Bool
    prev_alpha::T
    cent_count::Int

    rhs::Point{T}
    dir::Point{T}
    res::Point{T}
    dir_nocorr::Point{T}
    dir_corr::Point{T}
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
    stepper.prev_alpha = one(T)
    stepper.cent_count = 0

    model = solver.model
    stepper.rhs = Point(model)
    stepper.dir = Point(model)
    stepper.res = Point(model)
    dim = length(stepper.rhs.vec)
    if stepper.use_correction
        stepper.dir_nocorr = Point(model)
        stepper.dir_corr = Point(model)
    end
    stepper.dir_temp = zeros(T, dim)

    stepper.line_searcher = LineSearcher{T}(model)

    return stepper
end


function step(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    model = solver.model
    cones = model.cones
    rhs = stepper.rhs
    dir = stepper.dir

    # update linear system solver factorization
    update_lhs(solver.system_solver, solver)

    # decide whether to predict or center
    is_pred = (stepper.cent_count > 3) || all(Cones.in_neighborhood.(model.cones, sqrt(solver.mu), T(0.05))) # TODO tune
    stepper.cent_count = (is_pred ? 0 : stepper.cent_count + 1)
    (rhs_fun_nocorr, rhs_fun_corr) = (is_pred ? (update_rhs_pred, update_rhs_predcorr) : (update_rhs_cent, update_rhs_centcorr))

    # get initial direction
    rhs_fun_nocorr(solver, rhs)
    # get_directions(stepper, solver, is_pred, iter_ref_steps = 3)
    get_directions(stepper, solver, false, iter_ref_steps = 3) # TODO no NT for taukap for now

    # if !is_pred
        # # setup adjusted RHS for nonlinear equalities
        # # TODO for now just primal bar cones and tau
        # # s_k RHS
        # rhs_s = [zeros(T, Cones.dimension(cone_k)) for cone_k in cones]
        # # kap RHS
        # rhs_kap = zero(T)
        #
        #
        orig_rhs = copy(rhs.vec)
        #
        # # update adjusted RHS
        # # s_k RHS
        # # 1/2 ( ( \mu H_k (s_k) - \hmu H_k (\hs_k) ) \delta_{s,k} - ( z_k + \hz_k + \mu g_k (s_k) + \hmu g_k (\hs_k) ) )
        # rtmu = sqrt(solver.mu)
        # for (k, cone_k) in enumerate(cones)
        #     # rhs_s[k] .= rhs.primal_views[k]
        #     rhs_s_k = rhs_s[k]
        #     tmp = similar(rhs_s_k) # TODO
        #     Cones.hess_prod!(tmp, dir.primal_views[k], cone_k)
        #     duals_k = point.dual_views[k]
        #     grad_k = Cones.grad(cone_k)
        #     @show rtmu
        #     @show grad_k
        #     @. rhs_s[k] += tmp - duals_k - rtmu * grad_k
        # end
        # # kap RHS
        # tau = point.tau[1]
        # rhs_kap += solver.mu * (dir.tau[1] / tau + 1) / tau - point.kap[1]

        # get initial direction alpha to primal feasibility
        cand = stepper.res
        (alpha_prfeas, mu_search) = find_max_alpha_feas(point, cand, dir, model)
        @show alpha_prfeas
        @assert alpha_prfeas > 0

        # # update adjusted RHS
        # # s_k RHS
        # # 1/2 ( ( \mu H_k (s_k) - \hmu H_k (\hs_k) ) \delta_{s,k} - ( z_k + \hz_k + \mu g_k (s_k) + \hmu g_k (\hs_k) ) )
        # rtmu = sqrt(mu_search)
        # for (k, cone_k) in enumerate(cones)
        #     rhs_s_k = rhs_s[k]
        #     # NOTE mu and oracles here should be from line search
        #     tmp = similar(rhs_s_k)
        #     Cones.hess_prod!(tmp, dir.primal_views[k], cone_k)
        #     duals_k = cand.dual_views[k]
        #     grad_k = Cones.grad(cone_k)
        #     @show rtmu
        #     @show grad_k
        #     @. rhs_s[k] -= tmp + duals_k + rtmu * grad_k
        # end
        # # kap RHS
        # tau = cand.tau[1]
        # rhs_kap += mu_search * (-dir.tau[1] / tau + 1) / tau - cand.kap[1]
        #
        # # finalize new RHS and direction
        # rhs_fun_nocorr(solver, rhs) # TODO
        # for k in eachindex(cones)
        #     rhs.s_views[k] .= rhs_s[k] / 2
        # end
        # rhs.kap[1] = rhs_kap / 2

        rtmu = sqrt(mu_search)
        for (k, cone_k) in enumerate(cones)
            # NOTE mu and oracles here should be from line search
            duals_k = cand.dual_views[k]
            grad_k = Cones.grad(cone_k)
            @. rhs.s_views[k] += -alpha_prfeas * (duals_k + rtmu * grad_k)
        end
        rhs.kap[1] += alpha_prfeas * (-cand.kap[1] + mu_search / cand.tau[1])



        # @show rhs.vec - orig_rhs

        # TODO has to be at current interior point, not along line search
        get_directions(stepper, solver, false)#, iter_ref_steps = 3) # TODO no NT for taukap for now
    # end






    # get alpha step length
    # TODO change prev_alpha?
    alpha = find_max_alpha(point, dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3), max_nbhd = T(0.9))
    # alpha = find_max_alpha(point, dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3), max_nbhd = T(0.99))
    stepper.prev_alpha = alpha
    if iszero(alpha)
        # TODO attempt recovery
        @warn("very small alpha; terminating")
        solver.status = NumericalFailure
        return false
    end

    # step
    @. point.vec += alpha * dir.vec
    calc_mu(solver)

    return true
end

function find_max_alpha_feas(
    point::Point{T},
    cand::Point{T},
    dir::Point{T},
    model::Models.Model{T};
    ) where {T <: Real}
    cones = model.cones
    skzk = zeros(T, length(cones))
    nup1 = model.nu + 1

    min_alpha = T(1e-3)
    alpha_reduce = T(0.95)

    alpha = one(T)
    if dir.tau[1] < zero(T)
        alpha = min(alpha, -point.tau[1] / dir.tau[1])
    end
    if dir.kap[1] < zero(T)
        alpha = min(alpha, -point.kap[1] / dir.kap[1])
    end
    alpha *= T(0.9999) / alpha_reduce

    mu_c = zero(T)
    # TODO for feas, as soon as cone is feas, don't test feas again, since line search is backwards
    while true
        if alpha < min_alpha
            # alpha is very small so finish
            alpha = zero(T)
            break
        end
        alpha *= alpha_reduce

        @. cand.vec = point.vec + alpha * dir.vec # TODO only tau, kap, s, z

        min(cand.tau[1], cand.kap[1]) < eps(T) && continue
        taukap_c = cand.tau[1] * cand.kap[1]
        (taukap_c < eps(T)) && continue
        for k in eachindex(cones)
            skzk[k] = dot(cand.primal_views[k], cand.dual_views[k])
        end
        any(<(eps(T)), skzk) && continue

        mu_c = (sum(skzk) + taukap_c) / nup1
        (mu_c < eps(T)) && continue

        rtmu = sqrt(mu_c)
        irtmu = inv(rtmu)
        all_feas = true
        for (k, cone_k) in enumerate(cones)
            Cones.load_point(cone_k, cand.primal_views[k], irtmu)
            Cones.reset_data(cone_k)
            if !Cones.is_feas(cone_k)
                all_feas = false
                break
            end
        end
        all_feas && break
    end

    return (alpha, mu_c)
end



# function step(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
#     point = solver.point
#     model = solver.model
#     rhs = stepper.rhs
#     dir = stepper.dir
#
#     # update linear system solver factorization
#     update_lhs(solver.system_solver, solver)
#
#     # decide whether to predict or center
#     is_pred = (stepper.cent_count > 3) || all(Cones.in_neighborhood.(model.cones, sqrt(solver.mu), T(0.05))) # TODO tune
#     stepper.cent_count = (is_pred ? 0 : stepper.cent_count + 1)
#     (rhs_fun_nocorr, rhs_fun_corr) = (is_pred ? (update_rhs_pred, update_rhs_predcorr) : (update_rhs_cent, update_rhs_centcorr))
#
#     # get direction
#     rhs_fun_nocorr(solver, rhs)
#     get_directions(stepper, solver, is_pred, iter_ref_steps = 3)
#     # is_corrected = false
#     # if stepper.use_correction
#     #     rhs_fun_corr(solver, rhs, dir)
#     #     if !iszero(rhs.vec)
#     #         dir_nocorr = stepper.dir_nocorr
#     #         dir_corr = stepper.dir_corr
#     #         copyto!(dir_nocorr, dir.vec)
#     #         get_directions(stepper, solver, is_pred, iter_ref_steps = 3)
#     #         copyto!(dir_corr, dir.vec)
#     #
#     #         # get nocorr alpha step length
#     #         copyto!(dir.vec, dir_nocorr)
#     #         # TODO try max_nbhd = Inf, but careful of cones with no dual feas check
#     #         # TODO change prev_alpha?
#     #         alpha_nocorr = find_max_alpha(point, dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3))
#     #         if iszero(alpha_nocorr)
#     #             # @warn("very small alpha for uncorrected direction ($alpha_nocorr)")
#     #             alpha_nocorr = T(0.5) # attempt recovery TODO tune
#     #         end
#     #
#     #         # get direction with correction
#     #         @. dir.vec = dir_nocorr + alpha_nocorr * dir_corr
#     #         is_corrected = true
#     #     end
#     # end
#
#     # get alpha step length
#     # TODO change prev_alpha?
#     alpha = find_max_alpha(point, dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-3), max_nbhd = T(0.99))
#     # if is_corrected && alpha < 0.99 * alpha_nocorr
#     #     @warn("alpha worse after correction ($alpha < $alpha_nocorr)")
#     #     # # TODO test whether this helps:
#     #     # # step alpha_nocorr in uncorrected direction
#     #     # alpha = alpha_nocorr
#     #     # copyto!(dir.vec, dir_nocorr)
#     # end
#     stepper.prev_alpha = alpha
#     # if !is_pred && alpha < 0.99
#     #     @warn("small alpha for centering step")
#     # end
#     if iszero(alpha)
#         # TODO attempt recovery
#         @warn("very small alpha; terminating")
#         solver.status = NumericalFailure
#         return false
#     end
#
#     # step
#     @. point.vec += alpha * dir.vec
#     calc_mu(solver)
#
#     return true
# end

# function step(stepper::PredOrCentStepper{T}, solver::Solver{T}) where {T <: Real}
#     point = solver.point
#     model = solver.model
#     rhs = stepper.rhs
#     dir = stepper.dir
#
#     # update linear system solver factorization
#     update_lhs(solver.system_solver, solver)
#
#     # decide whether to predict or center
#     is_pred = (stepper.cent_count > 3) || all(Cones.in_neighborhood.(model.cones, sqrt(solver.mu), T(0.05))) # TODO tune
#     stepper.cent_count = (is_pred ? 0 : stepper.cent_count + 1)
#     (rhs_fun_nocorr, rhs_fun_corr) = (is_pred ? (update_rhs_pred, update_rhs_predcorr) : (update_rhs_cent, update_rhs_centcorr))
#
#     # get direction
#     dir_nocorr = stepper.dir_nocorr
#     dir_corr = stepper.dir_corr
#     rhs_fun_nocorr(solver, rhs)
#     get_directions(stepper, solver, is_pred, iter_ref_steps = 3)
#     copyto!(dir_nocorr.vec, dir.vec)
#     rhs_fun_corr(solver, rhs, dir)
#     get_directions(stepper, solver, is_pred, iter_ref_steps = 3)
#     copyto!(dir_corr.vec, dir.vec)
#
#     # get alpha and step
#     cand = stepper.res # TODO rename?
#     (a_pr, a_du) = find_alpha_curve(point, cand, dir_nocorr, dir_corr, model)
#     # @show a_pr, a_du
#     if iszero(a_pr) || iszero(a_du)
#         # try not using correction
#         @warn("very small alpha")
#         copyto!(dir.vec, dir_nocorr.vec)
#         alpha = find_max_alpha(point, dir, stepper.line_searcher, model, prev_alpha = one(T), min_alpha = T(1e-5), max_nbhd = T(0.9999))
#         if iszero(alpha)
#             @warn("very small alpha again; terminating")
#             solver.status = NumericalFailure
#             return false
#         end
#         @. point.vec += alpha * dir.vec
#     else
#         # # TODO or just use mu and point that come out of find_alpha_curve
#         # @. point.vec += alpha * (dir_nocorr.vec + alpha * dir_corr.vec)
#         point.kap[1] += a_pr * (dir_nocorr.kap[1] + a_pr * dir_corr.kap[1])
#         @. point.s += a_pr * (dir_nocorr.s + a_pr * dir_corr.s)
#         @. point.x += a_pr * (dir_nocorr.x + a_pr * dir_corr.x)
#         point.tau[1] += a_du * (dir_nocorr.tau[1] + a_du * dir_corr.tau[1])
#         @. point.z += a_du * (dir_nocorr.z + a_du * dir_corr.z)
#         @. point.y += a_du * (dir_nocorr.y + a_du * dir_corr.y)
#     end
#     stepper.prev_alpha = NaN
#     calc_mu(solver)
#
#     return true
# end


# function find_alpha_curve(
#     point,
#     cand,
#     dir_nocorr,
#     dir_corr,
#     model::Models.Model{T},
#     ) where {T <: Real}
#     cones = model.cones
#
#     min_nbhd = T(0.01)
#     # max_nbhd = T(0.99)
#     max_nbhd = T(Inf)
#
#     skzk = zeros(T, length(cones))
#     nup1 = model.nu + 1
#
#     min_alpha = T(1e-3)
#     alpha_reduce = T(0.95)
#     # TODO use an alpha schedule like T[0.9999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0] or a log-scale
#
#     a_pr = T(0.9999) / alpha_reduce # for x, s, kap
#     a_du = T(0.9999) / alpha_reduce # for y, z, tau
#
#     # primal alpha feasibility
#     while true
#         if a_pr < min_alpha
#             a_pr = zero(T)
#             break
#         end
#         a_pr *= alpha_reduce
#
#         kap_c = cand.kap[1] = point.kap[1] + a_pr * (dir_nocorr.kap[1] + a_pr * dir_corr.kap[1])
#         (kap_c < eps(T)) && continue
#
#         @. cand.s = point.s + a_pr * (dir_nocorr.s + a_pr * dir_corr.s)
#         all_feas = true
#         for (k, cone_k) in enumerate(cones)
#             if Cones.use_dual_barrier(cone_k)
#                 Cones.load_dual_point(cone_k, cand.dual_views[k])
#                 Cones.reset_data(cone_k)
#                 is_feas_k = Cones.is_dual_feas(cone_k)
#             else
#                 Cones.load_point(cone_k, cand.primal_views[k])
#                 Cones.reset_data(cone_k)
#                 is_feas_k = Cones.is_feas(cone_k)
#             end
#             if !is_feas_k
#                 all_feas = false
#                 break
#             end
#         end
#         all_feas && break
#     end
#
#     # dual alpha feasibility
#     while true
#         if a_du < min_alpha
#             a_du = zero(T)
#             break
#         end
#         a_du *= alpha_reduce
#
#         tau_c = cand.tau[1] = point.tau[1] + a_du * (dir_nocorr.tau[1] + a_du * dir_corr.tau[1])
#         (tau_c < eps(T)) && continue
#
#         @. cand.z = point.z + a_du * (dir_nocorr.z + a_du * dir_corr.z)
#         all_feas = true
#         for (k, cone_k) in enumerate(cones)
#             if Cones.use_dual_barrier(cone_k)
#                 Cones.load_point(cone_k, cand.primal_views[k])
#                 Cones.reset_data(cone_k)
#                 is_feas_k = Cones.is_feas(cone_k)
#             else
#                 Cones.load_dual_point(cone_k, cand.dual_views[k])
#                 Cones.reset_data(cone_k)
#                 is_feas_k = Cones.is_dual_feas(cone_k)
#             end
#             if !is_feas_k
#                 all_feas = false
#                 break
#             end
#         end
#         all_feas && break
#     end
#
#     # println("feas: ", a_pr, ", ", a_du)
#
#     # both alpha nbhd
#     a_pr /= alpha_reduce
#     a_du /= alpha_reduce
#     while true
#         # TODO ?
#         a_pr *= alpha_reduce
#         if a_pr < min_alpha
#             a_pr = zero(T)
#         end
#         a_du *= alpha_reduce
#         if a_du < min_alpha
#             a_du = zero(T)
#         end
#         iszero(a_pr * a_du) && break
#
#         kap_c = cand.kap[1] = point.kap[1] + a_pr * (dir_nocorr.kap[1] + a_pr * dir_corr.kap[1])
#         (kap_c < eps(T)) && continue
#         tau_c = cand.tau[1] = point.tau[1] + a_du * (dir_nocorr.tau[1] + a_du * dir_corr.tau[1])
#         (tau_c < eps(T)) && continue
#         taukap_c = tau_c * kap_c
#         (taukap_c < eps(T)) && continue
#
#         @. cand.s = point.s + a_pr * (dir_nocorr.s + a_pr * dir_corr.s)
#         # @. cand.x = point.x + a_pr * (dir_nocorr.x + a_pr * dir_corr.x)
#         @. cand.z = point.z + a_du * (dir_nocorr.z + a_du * dir_corr.z)
#         # @. cand.y = point.y + a_du * (dir_nocorr.y + a_du * dir_corr.y)
#         for k in eachindex(cones)
#             skzk[k] = dot(cand.primal_views[k], cand.dual_views[k])
#         end
#         mu_c = (sum(skzk) + taukap_c) / nup1
#         (mu_c < eps(T)) && continue
#
#         isfinite(max_nbhd) && (abs(taukap_c - mu_c) > max_nbhd * mu_c) && continue
#         min_nbhd_mu = min_nbhd * mu_c
#         (taukap_c < min_nbhd_mu) && continue
#         any(skzk[k] < min_nbhd_mu * Cones.get_nu(cone_k) for (k, cone_k) in enumerate(cones)) && continue
#
#         rtmu = sqrt(mu_c)
#         irtmu = inv(rtmu)
#         in_nbhd = true
#         for (k, cone_k) in enumerate(cones)
#             Cones.load_point(cone_k, cand.primal_views[k], irtmu)
#             Cones.load_dual_point(cone_k, cand.dual_views[k])
#             Cones.reset_data(cone_k)
#             in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k))# && (isinf(max_nbhd) || Cones.in_neighborhood(cone_k, rtmu, max_nbhd)))
#     Cones.in_neighborhood(cone_k, rtmu, max_nbhd)
#
#             if !in_nbhd_k
#                 in_nbhd = false
#                 break
#             end
#         end
#         in_nbhd && break
#     end
#
#
#     # println("final: ", a_pr, ", ", a_du)
#
#     return (a_pr, a_du)
# end



# # TODO move
# function find_alpha_curve(
#     point,
#     cand,
#     dir_nocorr,
#     dir_corr,
#     model::Models.Model{T},
#     ) where {T <: Real}
#     cones = model.cones
#
#     min_nbhd = T(0.01)
#     max_nbhd = T(0.99)
#
#     skzk = zeros(T, length(cones))
#     nup1 = model.nu + 1
#
#     min_alpha = T(1e-3)
#     alpha_reduce = T(0.95)
#     # TODO use an alpha schedule like T[0.9999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0] or a log-scale
#
#     alpha = T(0.9999)
#     alpha /= alpha_reduce
#     while true
#         if alpha < min_alpha
#             # alpha is very small so finish
#             alpha = zero(T)
#             break
#         end
#         alpha *= alpha_reduce
#
#         @. cand.vec = point.vec + alpha * (dir_nocorr + alpha * dir_corr)
#
#         min(cand.tau[1], cand.kap[1]) < eps(T) && continue
#         taukap_c = cand.tau[1] * cand.kap[1]
#         (taukap_c < eps(T)) && continue
#         for k in eachindex(cones)
#             skzk[k] = dot(cand.primal_views[k], cand.dual_views[k])
#         end
#         any(<(eps(T)), skzk) && continue
#
#         mu_c = (sum(skzk) + taukap_c) / nup1
#         (mu_c < eps(T)) && continue
#
#         min_nbhd_mu = min_nbhd * mu_c
#         (taukap_c < min_nbhd_mu) && continue
#         any(skzk[k] < min_nbhd_mu * Cones.get_nu(cone_k) for (k, cone_k) in enumerate(cones)) && continue
#         isfinite(max_nbhd) && (abs(taukap_c - mu_c) > max_nbhd * mu_c) && continue
#
#         rtmu = sqrt(mu_c)
#         irtmu = inv(rtmu)
#         in_nbhd = true
#         for (k, cone_k) in enumerate(cones)
#             Cones.load_point(cone_k, cand.primal_views[k], irtmu)
#             Cones.load_dual_point(cone_k, cand.dual_views[k])
#             Cones.reset_data(cone_k)
#             in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && (isinf(max_nbhd) || Cones.in_neighborhood(cone_k, rtmu, max_nbhd)))
#             if !in_nbhd_k
#                 in_nbhd = false
#                 break
#             end
#         end
#         in_nbhd && break
#     end
#
#     return alpha
# end

expect_improvement(stepper::PredOrCentStepper) = iszero(stepper.cent_count)

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
        step = (iszero(stepper.cent_count) ? "pred" : "cent")
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
