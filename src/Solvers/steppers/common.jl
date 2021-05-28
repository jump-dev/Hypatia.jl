#=
interior point stepping routines for algorithms based on the
homogeneous self dual embedding
=#

# update the RHS for prediction direction
function update_rhs_pred(
    solver::Solver{T},
    rhs::Point{T},
    ) where {T <: Real}
    rhs.x .= solver.x_residual
    rhs.y .= solver.y_residual
    rhs.z .= solver.z_residual
    rhs.tau[] = solver.tau_residual

    for (s_k, d_k) in zip(rhs.s_views, solver.point.dual_views)
        @. s_k = -d_k
    end

    rhs.kap[] = -solver.point.kap[]

    return rhs
end

# update the prediction RHS with adjustment
function update_rhs_predadj(
    solver::Solver{T},
    rhs::Point{T},
    dir::Point{T},
    ) where {T <: Real}
    rhs.vec .= 0

    rteps = sqrt(eps(T))
    irtrtmu = inv(sqrt(sqrt(solver.mu)))
    for (k, cone_k) in enumerate(solver.model.cones)
        Cones.use_dder3(cone_k) || continue
        H_prim_dir_k = cone_k.vec1
        prim_k_scal = cone_k.vec2
        prim_dir_k = dir.primal_views[k]

        @. prim_k_scal = irtrtmu * prim_dir_k
        Cones.hess_prod_slow!(H_prim_dir_k, prim_dir_k, cone_k)
        dder3_k = Cones.dder3(cone_k, prim_k_scal)

        # only use third order deriv if it nearly satisfies an identity
        dot1 = dot(dder3_k, cone_k.point)
        dot2 = irtrtmu * dot(prim_k_scal, H_prim_dir_k)
        dder3_viol = abs(dot1 - dot2) / (rteps + abs(dot2))
        if dder3_viol < T(1e-4) # TODO tune
            @. rhs.s_views[k] = H_prim_dir_k + dder3_k
        end
    end

    taubar = solver.point.tau[]
    tau_dir_tau = dir.tau[] / taubar
    rhs.kap[] = tau_dir_tau * solver.mu / taubar * (1 + tau_dir_tau)

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
    rhs.tau[] = 0

    rtmu = sqrt(solver.mu)
    for (k, cone_k) in enumerate(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. rhs.s_views[k] = -duals_k - rtmu * grad_k
    end

    # kap
    rhs.kap[] = -solver.point.kap[] + solver.mu / solver.point.tau[]

    return rhs
end

# update the centering RHS with adjustment
function update_rhs_centadj(
    solver::Solver{T},
    rhs::Point{T},
    dir::Point{T},
    ) where {T <: Real}
    rhs.vec .= 0

    rteps = sqrt(eps(T))
    irtrtmu = inv(sqrt(sqrt(solver.mu)))
    for (k, cone_k) in enumerate(solver.model.cones)
        Cones.use_dder3(cone_k) || continue
        H_prim_dir_k_scal = cone_k.vec1
        prim_k_scal = cone_k.vec2
        prim_dir_k = dir.primal_views[k]

        @. prim_k_scal = irtrtmu * prim_dir_k
        Cones.hess_prod_slow!(H_prim_dir_k_scal, prim_k_scal, cone_k)
        dder3_k = Cones.dder3(cone_k, prim_k_scal)

        # only use third order deriv if it nearly satisfies an identity
        dot1 = dot(dder3_k, cone_k.point)
        dot2 = dot(prim_k_scal, H_prim_dir_k_scal)
        dder3_viol = abs(dot1 - dot2) / (rteps + abs(dot2))
        if dder3_viol < T(1e-4) # TODO tune
            rhs.s_views[k] .= dder3_k
        end
    end

    taubar = solver.point.tau[]
    tau_dir_tau = dir.tau[] / taubar
    rhs.kap[] = tau_dir_tau * solver.mu / taubar * tau_dir_tau

    return rhs
end

include("predorcent.jl")
include("combined.jl")
