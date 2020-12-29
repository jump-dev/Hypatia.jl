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
    rhs.tau[] = solver.point.kap[] + solver.primal_obj_t - solver.dual_obj_t

    for (s_k, d_k) in zip(rhs.s_views, solver.point.dual_views)
        @. s_k = -d_k
    end

    rhs.kap[] = -solver.point.kap[]

    return rhs
end

# update the prediction RHS with a correction
function update_rhs_predcorr(
    solver::Solver{T},
    rhs::Point{T},
    dir::Point{T},
    ) where {T <: Real}
    rhs.vec .= 0

    rteps = sqrt(eps(T))
    irtrtmu = inv(sqrt(sqrt(solver.mu))) # TODO or mu^-0.25
    for (k, cone_k) in enumerate(solver.model.cones)
        Cones.use_correction(cone_k) || continue
        H_prim_dir_k = cone_k.vec1
        prim_k_scal = cone_k.vec2
        prim_dir_k = dir.primal_views[k]

        @. prim_k_scal = irtrtmu * prim_dir_k
        Cones.hess_prod!(H_prim_dir_k, prim_dir_k, cone_k)
        corr_k = Cones.correction(cone_k, prim_k_scal)

        # only use correction if it nearly satisfies an identity
        dot1 = dot(corr_k, cone_k.point)
        dot2 = irtrtmu * dot(prim_k_scal, H_prim_dir_k)
        corr_viol = abs(dot1 - dot2) / (rteps + abs(dot2))
        if corr_viol < T(1e-4) # TODO tune
            @. rhs.s_views[k] = H_prim_dir_k + corr_k
        # else
        #     @warn("pred corr viol: $corr_viol")
        end
    end

    # TODO NT way:
    rhs.kap[] = dir.tau[] * dir.kap[] / solver.point.tau[]
    # TODO SY way:
    # tau_dir_tau = dir.tau / solver.point.tau
    # rhs[end] = tau_dir_tau * solver.mu / solver.point.tau * (1 + tau_dir_tau)

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

# update the centering RHS with a correction
function update_rhs_centcorr(
    solver::Solver{T},
    rhs::Point{T},
    dir::Point{T},
    ) where {T <: Real}
    rhs.vec .= 0

    rteps = sqrt(eps(T))
    irtrtmu = inv(sqrt(sqrt(solver.mu)))
    for (k, cone_k) in enumerate(solver.model.cones)
        Cones.use_correction(cone_k) || continue
        H_prim_dir_k_scal = cone_k.vec1
        prim_k_scal = cone_k.vec2
        prim_dir_k = dir.primal_views[k]

        @. prim_k_scal = irtrtmu * prim_dir_k
        Cones.hess_prod!(H_prim_dir_k_scal, prim_k_scal, cone_k)
        corr_k = Cones.correction(cone_k, prim_k_scal)

        # only use correction if it nearly satisfies an identity
        dot1 = dot(corr_k, cone_k.point)
        dot2 = dot(prim_k_scal, H_prim_dir_k_scal)
        corr_viol = abs(dot1 - dot2) / (rteps + abs(dot2))
        if corr_viol < T(1e-4) # TODO tune
            rhs.s_views[k] .= corr_k
        # else
        #     @warn("cent corr viol: $corr_viol")
        end
    end

    # TODO NT way:
    # rhs.kap = dir.tau * dir.kap / solver.point.tau
    # TODO SY way:
    tau_dir_tau = dir.tau[] / solver.point.tau[]
    rhs.kap[] = tau_dir_tau * solver.mu / solver.point.tau[] * tau_dir_tau

    return rhs
end

function print_header(stepper::Stepper, solver::Solver)
    @printf("\n%5s %12s %12s %9s %9s ", "iter", "p_obj", "d_obj", "abs_gap", "x_feas")
    if !iszero(solver.model.p)
        @printf("%9s ", "y_feas")
    end
    @printf("%9s %9s %9s %9s ", "z_feas", "tau", "kap", "mu")
    print_header_more(stepper, solver)
    println()
    return
end
print_header_more(stepper::Stepper, solver::Solver) = nothing

function print_iteration(stepper::Stepper, solver::Solver)
    @printf("%5d %12.4e %12.4e %9.2e %9.2e ",
        solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap, solver.x_feas
        )
    if !iszero(solver.model.p)
        @printf("%9.2e ", solver.y_feas)
    end
    @printf("%9.2e %9.2e %9.2e %9.2e ",
        solver.z_feas, solver.point.tau[], solver.point.kap[], solver.mu
        )
    print_iteration_more(stepper, solver)
    println()
    return
end
print_iteration_more(stepper::Stepper, solver::Solver) = nothing

include("predorcent.jl")
include("combined.jl")
