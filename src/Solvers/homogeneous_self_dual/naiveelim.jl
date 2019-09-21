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
=#

abstract type NaiveElimSystemSolver{T <: Real} <: SystemSolver{T} end

function solve_system(system_solver::NaiveElimSystemSolver{T}, solver::Solver{T}, sol, rhs) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = system_solver.tau_row

    # TODO in-place
    sol4 = view(sol, 1:tau_row, :)
    rhs4 = rhs[1:tau_row, :]

    for (k, cone_k) in enumerate(model.cones)
        z_rows_k = (n + p) .+ model.cone_idxs[k]
        s_rows_k = (q + 1) .+ z_rows_k
        if Cones.use_dual(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            @. @views rhs4[z_rows_k, :] += rhs[s_rows_k, :]
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(rhs4[z_rows_k, :], rhs[z_rows_k, :], cone_k)
            @. @views rhs4[z_rows_k, :] += rhs[s_rows_k, :]
        end
    end
    # -c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs
    @. @views rhs4[end, :] += rhs[end, :]

    # TODO use dispatch here
    @assert system_solver isa NaiveElimDenseSystemSolver{T}
    ldiv!(sol4, system_solver.fact_cache, rhs4)

    # lift to get s and kap
    tau = sol4[end:end, :]

    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(tau_row + 1):(end - 1), :]
    mul!(s, model.h, tau)
    x = @view sol[1:n, :]
    mul!(s, model.G, x, -one(T), true)
    @. @views s -= rhs[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs[end:end, :]

    return sol
end

#=
direct dense
=#

mutable struct NaiveElimDenseSystemSolver{T <: Real} <: NaiveElimSystemSolver{T}
    tau_row
    lhs
    lhs_copy
    fact_cache
    NaiveElimDenseSystemSolver{T}() where {T <: Real} = new{T}()
end

function load(system_solver::NaiveElimDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    system_solver.tau_row = n + p + q + 1
    system_solver.lhs_copy = T[
        zeros(T,n,n)  model.A'      model.G'              model.c;
        -model.A      zeros(T,p,p)  zeros(T,p,q)          model.b;
        -model.G      zeros(T,q,p)  Matrix(one(T)*I,q,q)  model.h;
        -model.c'     -model.b'     -model.h'             one(T);
        ]
    system_solver.lhs = similar(system_solver.lhs_copy)
    # system_solver.fact_cache = HypLUSolveCache(system_solver.sol, system_solver.lhs, rhs)
    return system_solver
end

function update_fact(system_solver::NaiveElimDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    lhs = system_solver.lhs
    copyto!(lhs, system_solver.lhs_copy)
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
    system_solver.fact_cache = lu!(system_solver.lhs) # TODO use wrapped lapack function
    return system_solver
end
