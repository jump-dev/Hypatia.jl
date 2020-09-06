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

include("naive.jl")
include("naiveelim.jl")
include("symindef.jl")
include("qrchol.jl")

function solve_inner_system(
    system_solver::Union{NaiveElimSparseSystemSolver, SymIndefSparseSystemSolver},
    sol::Vector,
    rhs::Vector,
    )
    inv_prod(system_solver.fact_cache, sol, system_solver.lhs_sub, rhs)
    return sol
end

function solve_inner_system(
    system_solver::Union{NaiveElimDenseSystemSolver, SymIndefDenseSystemSolver},
    sol::Vector,
    rhs::Vector,
    )
    copyto!(sol, rhs)
    inv_prod(system_solver.fact_cache, sol)
    return sol
end

# reduce to 4x4 subsystem
function solve_system(
    system_solver::Union{NaiveElimSystemSolver{T}, SymIndefSystemSolver{T}, QRCholSystemSolver{T}},
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    tau_scal::T,
    ) where {T <: Real}
    model = solver.model

    solve_subsystem4(system_solver, solver, sol, rhs, tau_scal)
    tau = sol.tau[1]

    # lift to get s and kap
    # s = -G*x + h*tau - zrhs
    @. sol.s = model.h * tau - rhs.z
    mul!(sol.s, model.G, sol.x, -1, true)

    # kap = -kapbar/taubar*tau + kaprhs
    sol.kap[1] = -tau_scal * tau + rhs.kap[1]

    return sol
end

# reduce to 3x3 subsystem
function solve_subsystem4(
    system_solver::Union{SymIndefSystemSolver{T}, QRCholSystemSolver{T}},
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    tau_scal::T,
    ) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    rhs_sub = system_solver.rhs_sub
    sol_sub = system_solver.sol_sub
    dim3 = length(rhs_sub)
    x_rows = 1:n
    y_rows = n .+ (1:p)
    z_rows = (n + p) .+ (1:q)

    @. @views rhs_sub[x_rows] = rhs.x
    @. @views rhs_sub[y_rows] = -rhs.y

    setup_rhs3(system_solver, model, rhs, sol, rhs_sub)

    solve_subsystem3(system_solver, solver, sol_sub, rhs_sub)

    # TODO maybe use higher precision here
    const_sol = system_solver.const_sol

    # lift to get tau
    @views tau_num = rhs.tau[1] + rhs.kap[1] + dot(model.c, sol_sub[x_rows]) + dot(model.b, sol_sub[y_rows]) + dot(model.h, sol_sub[z_rows])
    @views tau_denom = tau_scal - dot(model.c, const_sol[x_rows]) - dot(model.b, const_sol[y_rows]) - dot(model.h, const_sol[z_rows])
    sol_tau = tau_num / tau_denom

    @. sol.vec[1:dim3] = sol_sub + sol_tau * const_sol
    sol.tau[1] = sol_tau

    return sol
end
