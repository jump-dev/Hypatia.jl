#=
Copyright 2018, Chris Coey and contributors

naive+elimination linear system solver
solves linear system in naive.jl via the following procedure

eliminate s
-G*x + h*tau - s = zrhs
so if using primal barrier
z_k + mu*H_k*s_k = srhs_k --> s_k = (mu*H_k)\(srhs_k - z_k)
-->
-G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
-->
-mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
or if using dual barrier
mu*H_k*z_k + s_k = srhs_k --> s_k = srhs_k - mu*H_k*z_k
-->
-G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k

eliminate kap
-c'x - b'y - h'z - kap = taurhs
so
mu/(taubar^2)*tau + kap = kaprhs --> kap = kaprhs - mu/(taubar^2)*tau
-->
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs

4x4 nonsymmetric system in (x, y, z, tau):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
(pr bar) -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
(du bar) -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs

TODO add iterative method
=#

mutable struct NaiveElimSystemSolver{T <: Real} <: SystemSolver{T}
    solver::Solver{T}
    use_sparse::Bool

    tau_row::Int

    rhs::Matrix{T}
    rhs_x1
    rhs_x2
    rhs_y1
    rhs_y2
    rhs_z1
    rhs_z2
    rhs_z1_k
    rhs_z2_k

    sol::Matrix{T}
    sol_x1
    sol_x2
    sol_y1
    sol_y2
    sol_z1
    sol_z2
    sol_s1
    sol_s2

    lhs_copy
    lhs

    fact_cache

    function NaiveElimSystemSolver{T}(; use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_sparse = use_sparse
        return system_solver
    end
end

# create the system_solver cache
function load(system_solver::NaiveElimSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.solver = solver
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    dim = n + p + 2q + 2

    rhs = zeros(T, dim, 2)
    sol = zeros(T, dim, 2)
    system_solver.rhs = rhs
    system_solver.sol = sol
    rows = 1:n
    system_solver.rhs_x1 = view(rhs, rows, 1)
    system_solver.rhs_x2 = view(rhs, rows, 2)
    system_solver.sol_x1 = view(sol, rows, 1)
    system_solver.sol_x2 = view(sol, rows, 2)
    rows = (n + 1):(n + p)
    system_solver.rhs_y1 = view(rhs, rows, 1)
    system_solver.rhs_y2 = view(rhs, rows, 2)
    system_solver.sol_y1 = view(sol, rows, 1)
    system_solver.sol_y2 = view(sol, rows, 2)
    z_start = n + p
    rows = (z_start + 1):(z_start + q)
    system_solver.rhs_z1 = view(rhs, rows, 1)
    system_solver.rhs_z2 = view(rhs, rows, 2)
    system_solver.rhs_z1_k = [view(rhs, z_start .+ idxs_k, 1) for idxs_k in cone_idxs]
    system_solver.rhs_z2_k = [view(rhs, z_start .+ idxs_k, 2) for idxs_k in cone_idxs]
    system_solver.sol_z1 = view(sol, rows, 1)
    system_solver.sol_z2 = view(sol, rows, 2)
    tau_row = n + p + q + 1
    system_solver.tau_row = tau_row
    rows = tau_row .+ (1:q)
    system_solver.sol_s1 = view(sol, rows, 1)
    system_solver.sol_s2 = view(sol, rows, 2)

    if system_solver.use_sparse
        system_solver.lhs = T[
            spzeros(T,n,n)  model.A'        model.G'              model.c;
            -model.A        spzeros(T,p,p)  spzeros(T,p,q)        model.b;
            -model.G        spzeros(T,q,p)  sparse(one(T)*I,q,q)  model.h;
            -model.c'       -model.b'       -model.h'             one(T);
            ]
        dropzeros!(system_solver.lhs)
        @assert issparse(system_solver.lhs)
    else
        system_solver.lhs_copy = T[
            zeros(T,n,n)  model.A'      model.G'              model.c;
            -model.A      zeros(T,p,p)  zeros(T,p,q)          model.b;
            -model.G      zeros(T,q,p)  Matrix(one(T)*I,q,q)  model.h;
            -model.c'     -model.b'     -model.h'             one(T);
            ]
        system_solver.lhs = similar(system_solver.lhs_copy)
        # system_solver.fact_cache = HypLUSolveCache(system_solver.sol, system_solver.lhs, rhs)
    end

    return system_solver
end

# update the system solver cache to prepare for solve
function update(system_solver::NaiveElimSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    tau_row = system_solver.tau_row

    # update rhs and lhs matrices
    system_solver.rhs_x1 .= solver.x_residual
    system_solver.rhs_x2 .= zero(T)
    system_solver.rhs_y1 .= solver.y_residual
    system_solver.rhs_y2 .= zero(T)

    system_solver.rhs[tau_row, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    system_solver.rhs[tau_row, 2] = zero(T)
    system_solver.rhs[tau_row, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    system_solver.rhs[tau_row, 2] = zero(T)

    kap_rhs1 = -solver.kap
    kap_rhs2 = -solver.kap + solver.mu / solver.tau
    rhs[end, 1] = tau_rhs1 + kap_rhs1
    rhs[end, 2] = kap_rhs2

    copyto!(lhs, system_solver.lhs_copy)
    lhs[end, end] = solver.mu / solver.tau / solver.tau

    sqrtmu = sqrt(solver.mu)
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        rows_k = (model.n + model.p) .+ idxs_k
        z1_kk = system_solver.rhs_z1_k[k]
        z2_kk = system_solver.rhs_z2_k[k]

        # TODO optimize by preallocing the views
        if Cones.use_dual(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            lhs[rows_k, rows_k] .= Cones.hess(cone_k)
            @views copyto!(z1_kk, solver.z_residual[idxs_k])
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(lhs[rows_k, 1:model.n], model.G[idxs_k, :], cone_k)
            @. lhs[rows_k, 1:model.n] *= -1
            @views Cones.hess_prod!(lhs[rows_k, end], model.h[idxs_k], cone_k)
            @views Cones.hess_prod!(z1_kk, solver.z_residual[idxs_k], cone_k)
        end

        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. z1_kk -= duals_k
        @. z2_kk = -duals_k - grad_k * sqrtmu
    end

    # factorize LHS
    system_solver.fact_cache = lu!(lhs)

    return system_solver
end

# solve without outer iterative refinement
function solve(system_solver::NaiveElimSystemSolver{T}, sol_curr, rhs_curr) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    lhs = system_solver.lhs

    dim4 = size(lhs, 1)
    sol4 = view(sol_curr, 1:dim4, :)
    rhs4 = view(rhs_curr, 1:dim4, :)
    if system_solver.use_sparse
        sol4 .= system_solver.fact_cache \ rhs4
    else
        # if !hyp_lu_solve!(system_solver.fact_cache, sol4, lhs, rhs4)
        #     @warn("numerical failure: could not fix linear solve failure (mu is $(solver.mu))")
        # end
        ldiv!(sol4, system_solver.fact_cache, rhs4)
    end

    # lift to get s and kap
    tau = sol4[end:end, :]

    # s = -G*x + h*tau - zrhs
    s = @view sol_curr[(dim4 + 1):(end - 1), :]
    mul!(s, model.h, tau)
    x = view(sol_curr, 1:model.n, :)
    mul!(s, model.G, x, -one(T), true)
    z_start = model.n + model.p
    @. @views s -= rhs_curr[(z_start + 1):(z_start + model.q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol_curr[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs_curr[end:end, :]

    return sol_curr
end

# # return directions
# # TODO make this function the same for all system solvers, move to solver.jl
# function get_combined_directions(system_solver::NaiveElimSystemSolver{T}) where {T <: Real}
#     solver = system_solver.solver
#     model = solver.model
#     lhs = system_solver.lhs
#     rhs = system_solver.rhs
#     sol = system_solver.sol
#     x1 = system_solver.x1
#     x2 = system_solver.x2
#     s1 = system_solver.s1
#     s2 = system_solver.s2
#
#     update(system_solver)
#
#     # solve system
#     if system_solver.use_sparse
#         rhs .= lu(lhs) \ rhs
#     else
#         if !hyp_lu_solve!(system_solver.solvecache, system_solver.solvesol, lhs, rhs)
#             @warn("numerical failure: could not fix linear solve failure (mu is $(solver.mu))")
#         end
#         copyto!(rhs, system_solver.solvesol)
#     end
#
#     return (x1, x2, system_solver.y1, system_solver.y2, system_solver.z1, system_solver.z2, tau1, tau2, s1, s2, kap1, kap2)
# end
