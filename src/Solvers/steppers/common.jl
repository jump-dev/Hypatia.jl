#=
interior point stepping routines for algorithms based on homogeneous self dual embedding
=#

# update the RHS for prediction direction
function update_rhs_pred(
    solver::Solver{T},
    rhs::Point{T},
    ) where {T <: Real}
    rhs.x .= solver.x_residual
    rhs.y .= solver.y_residual
    rhs.z .= solver.z_residual
    rhs.tau[1] = solver.point.kap[1] + solver.primal_obj_t - solver.dual_obj_t

    for (s_k, d_k) in zip(rhs.s_views, solver.point.dual_views)
        @. s_k = -d_k
    end

    rhs.kap[1] = -solver.point.kap[1]

    return rhs
end

# update the prediction RHS with a correction
function update_rhs_predcorr(
    solver::Solver{T},
    rhs::Point{T},
    dir::Point{T};
    add::Bool = true,
    ) where {T <: Real}
    if !add
        rhs.vec .= 0
    end

    irtrtmu = inv(sqrt(sqrt(solver.mu))) # TODO or mu^-0.25
    for (k, cone_k) in enumerate(solver.model.cones)
        Cones.use_correction(cone_k) || continue

        # TODO avoid allocs
        prim_dir_k = dir.primal_views[k]
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
            @. rhs.s_views[k] += H_prim_dir_k + corr_k
        # else
        #     println("skip pred-corr: $corr_viol")
        end
    end

    # TODO NT way:
    rhs.kap[1] += dir.tau[1] * dir.kap[1] / solver.point.tau[1]
    # TODO SY way:
    # tau_dir_tau = dir.tau / solver.point.tau
    # rhs[end] += tau_dir_tau * solver.mu / solver.point.tau * (1 + tau_dir_tau)

    return rhs
end

# update the RHS for centering direction
function update_rhs_cent(
    solver::Solver{T},
    rhs::Point{T},
    ) where {T <: Real}
    rhs.x .= 0
    rhs.y .= 0
    rhs.z .= 0
    rhs.tau[1] = 0

    rtmu = sqrt(solver.mu)
    for (k, cone_k) in enumerate(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. rhs.s_views[k] = -duals_k - rtmu * grad_k
    end

    # kap
    rhs.kap[1] = -solver.point.kap[1] + solver.mu / solver.point.tau[1]

    return rhs
end

# update the centering RHS with a correction
function update_rhs_centcorr(
    solver::Solver{T},
    rhs::Point{T},
    dir::Point{T};
    add::Bool = true,
    ) where {T <: Real}
    if !add
        rhs.vec .= 0
    end

    irtrtmu = inv(sqrt(sqrt(solver.mu)))
    for (k, cone_k) in enumerate(solver.model.cones)
        Cones.use_correction(cone_k) || continue

        # TODO avoid allocs
        prim_dir_k = dir.primal_views[k]
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
            rhs.s_views[k] .+= corr_k
        # else
        #     println("skip cent-corr: $corr_viol")
        end
    end

    # TODO NT way:
    # rhs.kap = dir.tau * dir.kap / solver.point.tau
    # TODO SY way:
    tau_dir_tau = dir.tau[1] / solver.point.tau[1]
    rhs.kap[1] = tau_dir_tau * solver.mu / solver.point.tau[1] * tau_dir_tau

    return rhs
end
