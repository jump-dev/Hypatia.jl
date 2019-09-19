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
    use_iterative::Bool
    use_sparse::Bool

    tau_row::Int

    rhs::Matrix{T}
    rhs_x1
    rhs_x2
    rhs_y1
    rhs_y2
    rhs_z1
    rhs_z2
    rhs_s1
    rhs_s2
    rhs_s1_k
    rhs_s2_k

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

    function NaiveElimSystemSolver{T}(; use_iterative::Bool = false, use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
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
    rows = (n + p + 1):(n + p + q)
    system_solver.rhs_z1 = view(rhs, rows, 1)
    system_solver.rhs_z2 = view(rhs, rows, 2)
    system_solver.sol_z1 = view(sol, rows, 1)
    system_solver.sol_z2 = view(sol, rows, 2)
    tau_row = n + p + q + 1
    system_solver.tau_row = tau_row
    rows = tau_row .+ (1:q)
    system_solver.rhs_s1 = view(rhs, rows, 1)
    system_solver.rhs_s2 = view(rhs, rows, 2)
    system_solver.rhs_s1_k = [view(rhs, tau_row .+ idxs_k, 1) for idxs_k in cone_idxs]
    system_solver.rhs_s2_k = [view(rhs, tau_row .+ idxs_k, 2) for idxs_k in cone_idxs]
    system_solver.sol_s1 = view(sol, rows, 1)
    system_solver.sol_s2 = view(sol, rows, 2)

    if system_solver.use_iterative
        system_solver.lhs = setup_block_lhs(system_solver)
    else
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
    end

    return system_solver
end

# for iterative methods, build block matrix for efficient multiplication
function setup_block_lhs(system_solver::NaiveElimSystemSolver{T}) where {T <: Real}
    error("not implemented")
end

# update the LHS factorization to prepare for solve
function update_fact(system_solver::NaiveElimSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    lhs = system_solver.lhs

    if !system_solver.use_sparse
        copyto!(lhs, system_solver.lhs_copy)
    end
    lhs[end, end] = solver.mu / solver.tau / solver.tau
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        if Cones.use_dual(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            lhs[z_rows_k, z_rows_k] .= Cones.hess(cone_k)
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(lhs[z_rows_k, 1:n], model.G[idxs_k, :], cone_k)
            @. lhs[z_rows_k, 1:n] *= -1
            @views Cones.hess_prod!(lhs[z_rows_k, end], model.h[idxs_k], cone_k)
        end
    end

    # factorize LHS
    system_solver.fact_cache = lu!(lhs)

    return system_solver
end

# solve system without outer iterative refinement
function solve_system(system_solver::NaiveElimSystemSolver{T}, sol_curr, rhs_curr) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = system_solver.tau_row

    # TODO in-place
    sol4 = view(sol_curr, 1:tau_row, :)
    # rhs4 = view(rhs_curr, 1:tau_row, :)
    rhs4 = rhs_curr[1:tau_row, :]

    for (k, cone_k) in enumerate(model.cones)
        z_rows_k = (n + p) .+ model.cone_idxs[k]
        s_rows_k = (q + 1) .+ z_rows_k
        if Cones.use_dual(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            @. @views rhs4[z_rows_k, :] += rhs_curr[s_rows_k, :]
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(rhs4[z_rows_k, :], rhs_curr[z_rows_k, :], cone_k)
            @. @views rhs4[z_rows_k, :] += rhs_curr[s_rows_k, :]
        end
    end
    # -c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs
    @. @views rhs4[end, :] += rhs_curr[end, :]

    if system_solver.use_iterative
        error("not implemented")
        # for j in 1:size(rhs_curr, 2)
        #     rhs_j = view(rhs4, :, j)
        #     sol_j = view(sol4, :, j)
        #     IterativeSolvers.gmres!(sol_j, system_solver.lhs, rhs_j, restart = tau_row)
        # end
    else
        if system_solver.use_sparse
            sol4 .= system_solver.fact_cache \ rhs4
        else
            # if !hyp_lu_solve!(system_solver.fact_cache, sol4, lhs, rhs4)
            #     @warn("numerical failure: could not fix linear solve failure (mu is $(solver.mu))")
            # end
            ldiv!(sol4, system_solver.fact_cache, rhs4)
        end
    end

    # lift to get s and kap
    tau = sol4[end:end, :]

    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol_curr[(tau_row + 1):(end - 1), :]
    mul!(s, model.h, tau)
    x = @view sol_curr[1:n, :]
    mul!(s, model.G, x, -one(T), true)
    @. @views s -= rhs_curr[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol_curr[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs_curr[end:end, :]

    return sol_curr
end
