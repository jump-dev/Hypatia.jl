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
    use_nt::Bool,
    ) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    dir_temp = stepper.dir_temp
    res = stepper.temp
    system_solver = solver.system_solver
    res_norm_cutoff = solver.res_norm_cutoff
    max_ref_steps = solver.max_ref_steps

    tau = solver.point.tau[]
    tau_scal = (use_nt ? solver.point.kap[] : solver.mu / tau) / tau

    solve_system(system_solver, solver, dir, rhs, tau_scal)

    iszero(max_ref_steps) && return dir

    # compute residual norm
    copyto!(dir_temp, dir.vec)
    apply_lhs(stepper, solver, tau_scal) # modifies res
    res.vec .-= rhs.vec
    res_norm = norm(res.vec, Inf)



# TODO if prev and curr IR steps both didn't improve res by more than a threshold, then stop doing IR
# and maybe allow 5 steps max again

    # use iterative refinement if residual norm exceeds cutoff
    if res_norm <= res_norm_cutoff
        max_ref_steps = 0
    end
    for i in 1:max_ref_steps
        (res_norm < res_norm_cutoff) && break

        # compute refined direction
        solve_system(system_solver, solver, dir, res, tau_scal)
        axpby!(true, dir_temp, -1, dir.vec)

        # compute residual
        apply_lhs(stepper, solver, tau_scal) # modifies res
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
    end

    @assert !isnan(res_norm) # TODO error instead
    solver.worst_dir_res = max(solver.worst_dir_res, res_norm)

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
        Cones.hess_prod!(s_res_k, dir.primal_views[k], cone_k)
        @. s_res_k += dir.dual_views[k]
    end

    res.kap[] = tau_scal * tau_dir + kap_dir

    return res
end

include("naive.jl")
include("naiveelim.jl")
include("symindef.jl")
include("qrchol.jl")

function solve_inner_system(
    system_solver::Union{NaiveElimSparseSystemSolver, SymIndefSparseSystemSolver},
    sol::Point,
    rhs::Point,
    )
    inv_prod(system_solver.fact_cache, sol.vec, system_solver.lhs_sub, rhs.vec)
    return sol
end

function solve_inner_system(
    system_solver::Union{NaiveElimDenseSystemSolver, SymIndefDenseSystemSolver},
    sol::Point,
    rhs::Point,
    )
    copyto!(sol.vec, rhs.vec)
    inv_prod(system_solver.fact_cache, sol.vec)
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
    tau = sol.tau[]

    # lift to get s and kap
    # s = -G*x + h*tau - zrhs
    @. sol.s = model.h * tau - rhs.z
    mul!(sol.s, model.G, sol.x, -1, true)

    # kap = -kapbar/taubar*tau + kaprhs
    sol.kap[] = -tau_scal * tau + rhs.kap[]

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
    rhs_sub = system_solver.rhs_sub
    sol_sub = system_solver.sol_sub

    @. rhs_sub.x = rhs.x
    @. rhs_sub.y = -rhs.y
    setup_rhs3(system_solver, model, rhs, sol, rhs_sub)

    solve_subsystem3(system_solver, solver, sol_sub, rhs_sub)

    # lift to get tau
    sol_const = system_solver.sol_const
    tau_num = rhs.tau[] + rhs.kap[] + dot_obj(model, sol_sub)
    tau_denom = tau_scal - dot_obj(model, sol_const)
    sol_tau = tau_num / tau_denom

    dim3 = length(sol_sub.vec)
    @. @views sol.vec[1:dim3] = sol_sub.vec + sol_tau * sol_const.vec
    sol.tau[] = sol_tau

    return sol
end

function setup_point_sub(system_solver::Union{QRCholSystemSolver{T}, SymIndefSystemSolver{T}}, model::Models.Model{T}) where {T <: Real}
    (n, p, q) = (model.n, model.p, model.q)
    dim_sub = n + p + q
    z_start = model.n + model.p

    sol_sub = system_solver.sol_sub = Point{T}()
    rhs_sub = system_solver.rhs_sub = Point{T}()
    rhs_const = system_solver.rhs_const = Point{T}()
    sol_const = system_solver.sol_const = Point{T}()
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

    return nothing
end

dot_obj(model::Models.Model, point::Point) = dot(model.c, point.x) + dot(model.b, point.y) + dot(model.h, point.z)
