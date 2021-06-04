#=
linear system solver subroutines

6x6 nonsymmetric system in (x, y, z, tau, s, kap):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x + h*tau - s = zrhs
-c'*x - b'*y - h'*z - kap = taurhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
mu/(taubar^2)*tau + kap = kaprhs
=#

# calculate direction given rhs, and apply iterative refinement
function get_directions(
    stepper::Stepper{T},
    solver::Solver{T},
    min_impr_tol::T = T(0.5) # improvement tolerance for iterative refinement
    ) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    dir_temp = stepper.dir_temp
    res = stepper.temp
    syssolver = solver.syssolver
    res_norm_cutoff = solver.res_norm_cutoff
    max_ref_steps = solver.max_ref_steps

    solve_system(syssolver, solver, dir, rhs)

    iszero(max_ref_steps) && return dir

    # compute residual norm
    copyto!(dir_temp, dir.vec)
    apply_lhs(stepper, solver) # modifies res
    res.vec .-= rhs.vec
    res_norm = norm(res.vec, Inf)

    # use iterative refinement if residual norm exceeds cutoff
    if res_norm > res_norm_cutoff
        is_prev_slow = false
        prev_res_norm = res_norm

        for i in 1:max_ref_steps
            # compute refined direction
            solve_system(syssolver, solver, dir, res)
            axpby!(true, dir_temp, -1, dir.vec)

            # compute residual
            apply_lhs(stepper, solver) # modifies res
            res.vec .-= rhs.vec

            res_norm_new = norm(res.vec, Inf)
            if res_norm_new >= res_norm
                # residual has not improved
                copyto!(dir.vec, dir_temp)
                break
            end

            # residual has improved
            copyto!(dir_temp, dir.vec)
            res_norm = res_norm_new

            (res_norm < res_norm_cutoff) && break
            is_curr_slow = (res_norm > min_impr_tol * prev_res_norm)
            is_prev_slow && is_curr_slow && break # slow progress

            prev_res_norm = res_norm
            is_prev_slow = is_curr_slow
        end
    end

    @assert !isnan(res_norm) # TODO error instead
    solver.worst_dir_res = max(solver.worst_dir_res, res_norm)

    return dir
end

# calculate residual on 6x6 linear system
function apply_lhs(
    stepper::Stepper{T},
    solver::Solver{T},
    ) where {T <: Real}
    model = solver.model
    dir = stepper.dir
    res = stepper.temp
    tau_dir = dir.tau[]
    kap_dir = dir.kap[]

    # A'*y + G'*z + c*tau
    copyto!(res.x, model.c)
    mul!(res.x, model.G', dir.z, true, tau_dir)
    # -G*x + h*tau - s
    @. res.z = model.h * tau_dir - dir.s
    mul!(res.z, model.G, dir.x, -1, true)
    # -c'*x - b'*y - h'*z - kap
    res.tau[] = -dot(model.c, dir.x) - dot(model.h, dir.z) - kap_dir
    # if p = 0, ignore A, b, y
    if !iszero(model.p)
        # A'*y + G'*z + c*tau
        mul!(res.x, model.A', dir.y, true, true)
        # -A*x + b*tau
        copyto!(res.y, model.b)
        mul!(res.y, model.A, dir.x, -1, tau_dir)
        # -c'*x - b'*y - h'*z - kap
        res.tau[] -= dot(model.b, dir.y)
    end

    # s
    for (k, cone_k) in enumerate(model.cones)
        # (du bar) mu*H_k*z_k + s_k
        # (pr bar) z_k + mu*H_k*s_k
        s_res_k = res.s_views[k]
        Cones.hess_prod_slow!(s_res_k, dir.primal_views[k], cone_k)
        @. s_res_k += dir.dual_views[k]
    end

    tau = solver.point.tau[]
    res.kap[] = solver.mu / tau * tau_dir / tau + kap_dir

    return res
end

include("naive.jl")
include("naiveelim.jl")
include("symindef.jl")
include("qrchol.jl")

# reduce to 4x4 subsystem
function solve_system(
    syssolver::Union{NaiveElimSystemSolver{T}, SymIndefSystemSolver{T},
        QRCholSystemSolver{T}},
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    ) where {T <: Real}
    model = solver.model

    solve_subsystem4(syssolver, solver, sol, rhs)
    tau = sol.tau[]

    # lift to get s and kap
    # s = -G*x + h*tau - zrhs
    @. sol.s = model.h * tau - rhs.z
    mul!(sol.s, model.G, sol.x, -1, true)

    # kap = -kapbar/taubar*tau + kaprhs
    taubar = solver.point.tau[]
    sol.kap[] = -solver.mu / taubar / taubar * tau + rhs.kap[]

    return sol
end

# reduce to 3x3 subsystem
function solve_subsystem4(
    syssolver::Union{SymIndefSystemSolver{T}, QRCholSystemSolver{T}},
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    ) where {T <: Real}
    model = solver.model
    rhs_sub = syssolver.rhs_sub
    sol_sub = syssolver.sol_sub

    @. rhs_sub.x = rhs.x
    @. rhs_sub.y = -rhs.y
    setup_rhs3(syssolver, model, rhs, sol, rhs_sub)

    solve_subsystem3(syssolver, solver, sol_sub, rhs_sub)

    # lift to get tau
    sol_const = syssolver.sol_const
    tau_num = rhs.tau[] + rhs.kap[] + dot_obj(model, sol_sub)
    taubar = solver.point.tau[]
    tau_denom = solver.mu / taubar / taubar - dot_obj(model, sol_const)
    sol_tau = tau_num / tau_denom

    dim3 = length(sol_sub.vec)
    @. sol.vec[1:dim3] = sol_sub.vec + sol_tau * sol_const.vec
    sol.tau[] = sol_tau

    return sol
end

function setup_point_sub(
    syssolver::Union{QRCholSystemSolver{T}, SymIndefSystemSolver{T}},
    model::Models.Model{T},
    ) where {T <: Real}
    (n, p, q) = (model.n, model.p, model.q)
    dim_sub = n + p + q
    z_start = model.n + model.p

    sol_sub = syssolver.sol_sub = Point{T}()
    rhs_sub = syssolver.rhs_sub = Point{T}()
    rhs_const = syssolver.rhs_const = Point{T}()
    sol_const = syssolver.sol_const = Point{T}()
    for point_sub in (sol_sub, rhs_sub, rhs_const, sol_const)
        point_sub.vec = zeros(T, dim_sub)
        @views point_sub.x = point_sub.vec[1:n]
        @views point_sub.y = point_sub.vec[n .+ (1:p)]
        @views point_sub.z = point_sub.vec[n + p .+ (1:q)]
        point_sub.z_views = [view(point_sub.z, idxs) for idxs in model.cone_idxs]
    end
    @. rhs_const.x = -model.c
    @. rhs_const.y = model.b
    @. rhs_const.z = model.h

    return
end

dot_obj(model::Models.Model, point::Point) = dot(model.c, point.x) +
    dot(model.b, point.y) + dot(model.h, point.z)
