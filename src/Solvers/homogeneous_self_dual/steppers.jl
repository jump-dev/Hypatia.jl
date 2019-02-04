


function find_max_alpha_in_nbhd(z_dir::AbstractVector{Float64}, s_dir::AbstractVector{Float64}, tau_dir::Float64, kap_dir::Float64, nbhd::Float64, solver::HSDSolver)
    point = solver.point
    model = solver.model
    cones = model.cones

    alpha = 1.0 # TODO maybe start at previous alpha but increased slightly, or use affine_alpha
    if kap_dir < 0.0
        alpha = min(alpha, -solver.kap / kap_dir)
    end
    if tau_dir < 0.0
        alpha = min(alpha, -solver.tau / tau_dir)
    end
    # TODO what about mu? quadratic equation. need dot(ls_s, ls_z) + ls_tau * ls_kap > 0
    alpha *= 0.99

    # ONLY BACKTRACK, NO FORWARDTRACK

    ls_z = similar(point.z) # TODO prealloc
    ls_s = similar(point.s)
    primal_views = [view(Cones.use_dual(cones[k]) ? ls_z : ls_s, model.cone_idxs[k]) for k in eachindex(cones)]
    dual_views = [view(Cones.use_dual(cones[k]) ? ls_s : ls_z, model.cone_idxs[k]) for k in eachindex(cones)]

    # cones_outside_nbhd = trues(length(cones))
    # TODO sort cones so that check the ones that failed in-cone check last iteration first

    ls_tau = ls_kap = ls_tk = ls_mu = 0.0
    num_pred_iters = 0
    while num_pred_iters < 100
        num_pred_iters += 1

        @. ls_z = point.z + alpha * z_dir
        @. ls_s = point.s + alpha * s_dir
        ls_tau = solver.tau + alpha * tau_dir
        ls_kap = solver.kap + alpha * kap_dir
        ls_tk = ls_tau * ls_kap
        ls_mu = (dot(ls_s, ls_z) + ls_tk) / (1.0 + model.nu)

        # accept primal iterate if
        # - decreased alpha and it is the first inside the cone and beta-neighborhood or
        # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
        # if ls_mu > 0.0 && abs(ls_tk - ls_mu) / ls_mu < nbhd # condition for 1-dim nonneg cone for tau and kap
        #     in_nbhds = true
        #     for k in eachindex(cones)
        #         cone_k = cones[k]
        #         Cones.load_point(cone_k, primal_views[k])
        #         if !Cones.check_in_cone(cone_k) || calc_neighborhood(cone_k, dual_views[k], ls_mu) > nbhd
        #             in_nbhds = false
        #             break
        #         end
        #     end
        #     if in_nbhds
        #         break
        #     end
        # end
        if ls_mu > 0.0
            full_nbhd_sqr = abs2(ls_tk - ls_mu)
            in_nbhds = true
            for k in eachindex(cones)
                cone_k = cones[k]
                Cones.load_point(cone_k, primal_views[k])
                if !Cones.check_in_cone(cone_k)
                    in_nbhds = false
                    break
                end

                # TODO no allocs
                temp = dual_views[k] + ls_mu * Cones.grad(cone_k)
                # TODO use cholesky L
                # nbhd = sqrt(temp' * Cones.inv_hess(cone) * temp) / mu
                full_nbhd_sqr += temp' * Cones.inv_hess(cone_k) * temp

                if full_nbhd_sqr > abs2(ls_mu * nbhd)
                    in_nbhds = false
                    break
                end
            end
            if in_nbhds
                break
            end
        end

        if alpha < 1e-5
            # alpha is very small
            return 0.0
        end

        # iterate is outside the neighborhood: decrease alpha
        alpha *= 0.8 # TODO option for parameter
    end

    return alpha
end



# function calc_neighborhood(cone::Cones.Cone, duals::AbstractVector{Float64}, mu::Float64)
#     # TODO no allocs
#     temp = duals + mu * Cones.grad(cone)
#     # TODO use cholesky L
#     # nbhd = sqrt(temp' * Cones.inv_hess(cone) * temp) / mu
#     nbhd = temp' * Cones.inv_hess(cone) * temp
#     # @show nbhd
#     return nbhd
# end
