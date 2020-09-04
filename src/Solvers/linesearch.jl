#=
line search for s,z
=#

# # backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
# function find_max_alpha(
#     stepper::Stepper{T},
#     solver::Solver{T};
#     prev_alpha::T,
#     min_alpha::T,
#     ) where {T <: Real}
#     cones = solver.model.cones
#     cone_times = stepper.cone_times
#     cone_order = stepper.cone_order
#     z = solver.point.z
#     s = solver.point.s
#     tau = solver.point.tau
#     kap = solver.point.kap
#     z_dir = stepper.dir.z
#     s_dir = stepper.dir.s
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
    stepper::Stepper{T},
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
    tau = solver.point.tau[1]
    kap = solver.point.kap[1]
    z_dir = stepper.dir.z
    s_dir = stepper.dir.s
    tau_dir = stepper.dir.tau[1]
    kap_dir = stepper.dir.kap[1]
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
