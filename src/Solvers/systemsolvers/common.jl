#=
helpers for linear system solving
=#

# calculate direction given rhs, and apply iterative refinement
function get_directions(
    stepper::Stepper{T},
    solver::Solver{T},
    use_nt::Bool;
    iter_ref_steps::Int = 0,
    ) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    dir_temp = stepper.dir_temp
    res = stepper.res
    system_solver = solver.system_solver
    timer = solver.timer

    tau_scal = (use_nt ? solver.point.kap[1] : solver.mu / solver.point.tau[1]) / solver.point.tau[1]

    solve_system(system_solver, solver, dir, rhs, tau_scal)

    # use iterative refinement
    copyto!(dir_temp, dir.vec)
    apply_lhs(stepper, solver, tau_scal) # modifies res
    res.vec .-= rhs.vec
    norm_inf = norm(res.vec, Inf)
    norm_2 = norm(res.vec, 2)
    # @show res

    for i in 1:iter_ref_steps
        # @show norm_inf
        if norm_inf < 100 * eps(T) # TODO change tolerance dynamically
            break
        end
        solve_system(system_solver, solver, dir, res, tau_scal)
        axpby!(true, dir_temp, -1, dir.vec)
        res = apply_lhs(stepper, solver, tau_scal) # modifies res
        res.vec .-= rhs.vec
        # @show res

        norm_inf_new = norm(res.vec, Inf)
        norm_2_new = norm(res.vec, 2)
        if norm_inf_new > norm_inf || norm_2_new > norm_2
            # residual has not improved
            copyto!(dir.vec, dir_temp)
            break
        end

        # residual has improved, so use the iterative refinement
        # TODO only print if using debug mode
        # solver.verbose && @printf("iter ref round %d norms: inf %9.2e to %9.2e, two %9.2e to %9.2e\n", i, norm_inf, norm_inf_new, norm_2, norm_2_new)
        copyto!(dir_temp, dir.vec)
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
function apply_lhs(
    stepper::Stepper{T},
    solver::Solver{T},
    tau_scal::T,
    ) where {T <: Real}
    model = solver.model
    dir = stepper.dir
    res = stepper.res
    tau_dir = dir.tau[1]
    kap_dir = dir.kap[1]

    # A'*y + G'*z + c*tau
    copyto!(res.x, model.c)
    mul!(res.x, model.G', dir.z, true, tau_dir)
    # -G*x + h*tau - s
    @. res.z = model.h * tau_dir - dir.s
    mul!(res.z, model.G, dir.x, -1, true)
    # -c'*x - b'*y - h'*z - kap
    res.tau[1] = -dot(model.c, dir.x) - dot(model.h, dir.z) - kap_dir
    # if p = 0, ignore A, b, y
    if !iszero(model.p)
        # A'*y + G'*z + c*tau
        mul!(res.x, model.A', dir.y, true, true)
        # -A*x + b*tau
        copyto!(res.y, model.b)
        mul!(res.y, model.A, dir.x, -1, tau_dir)
        # -c'*x - b'*y - h'*z - kap
        res.tau[1] -= dot(model.b, dir.y)
    end

    # s
    for (k, cone_k) in enumerate(model.cones)
        # (du bar) mu*H_k*z_k + s_k
        # (pr bar) z_k + mu*H_k*s_k
        s_res_k = res.s_views[k]
        Cones.hess_prod!(s_res_k, dir.primal_views[k], cone_k)
        @. s_res_k += dir.dual_views[k]
    end

    res.kap[1] = tau_scal * tau_dir + kap_dir

    return stepper.res
end
