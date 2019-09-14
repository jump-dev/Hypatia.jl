#=
Copyright 2018, Chris Coey and contributors

symmetric-indefinite linear system solver
solves linear system in naive.jl by first eliminating s and kap via the method in naiveelim.jl and then eliminating tau via a procedure similar to that described by S7.4 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

3x3 nonsymmetric system in (x, y, z):
A'*y + G'*z = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) mu*H_k*G_k*x - z_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]

multiply pr bar constraint by (mu*H_k)^-1 to get 3x3 symmetric indefinite system
A'*y + G'*z = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]

or to avoid inverse hessian products, let for pr bar w_k = (mu*H_k)\z_k (later recover z_k = mu*H_k*w_k) to get 3x3 symmetric indefinite system
A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]

TODO
- add iterative method
- improve numerics of method with use_hess_inv = false
=#

mutable struct SymIndefSystemSolver{T <: Real} <: SystemSolver{T}
    use_sparse::Bool
    use_hess_inv::Bool

    solver::Solver{T}

    x1
    x2
    x3
    y1
    y2
    y3
    z1
    z2
    z3
    z1_k
    z2_k
    z3_k
    z_k
    zcopy_k
    s2::Vector{T}
    s3::Vector{T}

    lhs_copy
    lhs
    rhs::Matrix{T}

    solvesol
    solvecache

    function SymIndefSystemSolver{T}(; use_sparse::Bool = false, use_hess_inv::Bool = true) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_sparse = use_sparse
        system_solver.use_hess_inv = use_hess_inv
        return system_solver
    end
end

function load(system_solver::SymIndefSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.solver = solver

    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    rhs = zeros(T, n + p + q, 3)
    system_solver.rhs = rhs
    rows = 1:n
    system_solver.x1 = view(rhs, rows, 1)
    system_solver.x2 = view(rhs, rows, 2)
    system_solver.x3 = view(rhs, rows, 3)
    rows = (n + 1):(n + p)
    system_solver.y1 = view(rhs, rows, 1)
    system_solver.y2 = view(rhs, rows, 2)
    system_solver.y3 = view(rhs, rows, 3)
    rows = (n + p + 1):(n + p + q)
    system_solver.z1 = view(rhs, rows, 1)
    system_solver.z2 = view(rhs, rows, 2)
    system_solver.z3 = view(rhs, rows, 3)
    z_start = n + p
    system_solver.z1_k = [view(rhs, z_start .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
    system_solver.z2_k = [view(rhs, z_start .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]
    system_solver.z3_k = [view(rhs, z_start .+ model.cone_idxs[k], 3) for k in eachindex(model.cones)]
    if !system_solver.use_hess_inv
        system_solver.z_k = [Cones.use_dual(model.cones[k]) ? nothing : view(rhs, z_start .+ model.cone_idxs[k], :) for k in eachindex(model.cones)]
        system_solver.zcopy_k = [Cones.use_dual(model.cones[k]) ? nothing : zeros(T, length(system_solver.z1_k[k]), 3) for k in eachindex(model.cones)]
    end
    system_solver.s2 = similar(rhs, q)
    system_solver.s3 = similar(rhs, q)

    # symmetric, lower triangle filled only
    if system_solver.use_sparse
        system_solver.lhs_copy = T[
            spzeros(T,n,n)  spzeros(T,n,p)  spzeros(T,n,q);
            model.A         spzeros(T,p,p)  spzeros(T,p,q);
            model.G         spzeros(T,q,p)  sparse(-one(T)*I,q,q);
            ]
        dropzeros!(system_solver.lhs_copy)
        @assert issparse(system_solver.lhs_copy)
    else
        system_solver.lhs_copy = T[
            zeros(T,n,n)  zeros(T,n,p)  zeros(T,n,q);
            model.A       zeros(T,p,p)  zeros(T,p,q);
            model.G       zeros(T,q,p)  Matrix(-one(T)*I,q,q);
            ]
    end

    system_solver.lhs = similar(system_solver.lhs_copy)

    if !system_solver.use_sparse
        system_solver.solvesol = Matrix{T}(undef, size(system_solver.lhs, 1), 3)
        system_solver.solvecache = HypBKSolveCache('L', system_solver.solvesol, system_solver.lhs, system_solver.rhs)
    end

    return system_solver
end

function get_combined_directions(system_solver::SymIndefSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    cones = model.cones
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    x1 = system_solver.x1
    x2 = system_solver.x2
    x3 = system_solver.x3
    y1 = system_solver.y1
    y2 = system_solver.y2
    y3 = system_solver.y3
    z1 = system_solver.z1
    z2 = system_solver.z2
    z3 = system_solver.z3
    z1_k = system_solver.z1_k
    z2_k = system_solver.z2_k
    z3_k = system_solver.z3_k
    s2 = system_solver.s2
    s3 = system_solver.s3

    sqrtmu = sqrt(solver.mu)

    # update rhs and lhs matrices
    @. x1 = -model.c
    x2 .= solver.x_residual
    x3 .= zero(T)
    y1 .= model.b
    @. y2 = -solver.y_residual
    y3 .= zero(T)
    @. z2 .= -solver.z_residual
    z3 .= zero(T)

    copyto!(lhs, system_solver.lhs_copy)
    for k in eachindex(cones)
        cone_k = cones[k]
        idxs_k = model.cone_idxs[k]
        rows = (model.n + model.p) .+ idxs_k
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            H = Cones.hess(cone_k)
            @. lhs[rows, rows] = -H
            @. z2_k[k] += duals_k
            @. z3_k[k] = duals_k + grad_k * sqrtmu
            @views copyto!(z1_k[k], model.h[idxs_k])
        else
            if system_solver.use_hess_inv
                # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
                Hinv = Cones.inv_hess(cone_k)
                @. lhs[rows, rows] = -Hinv
                Cones.inv_hess_prod!(z1_k[k], duals_k, cone_k)
                @. z2_k[k] += z1_k[k]
                @. z1_k[k] = duals_k + grad_k * sqrtmu
                Cones.inv_hess_prod!(z3_k[k], z1_k[k], cone_k)
                @views copyto!(z1_k[k], model.h[idxs_k])
            else
                # A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
                # mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
                H = Cones.hess(cone_k)
                @. lhs[rows, rows] = -H
                @views Cones.hess_prod!(lhs[rows, 1:n], model.G[idxs_k, :], cone_k)
                @views Cones.hess_prod!(z1_k[k], model.h[idxs_k], cone_k)
                Cones.hess_prod!(z3_k[k], z2_k[k], cone_k)
                @. z2_k[k] = duals_k + z3_k[k]
                @. z3_k[k] = duals_k + grad_k * sqrtmu
            end
        end
    end

    # solve system
    if system_solver.use_sparse
        lhs_symm = Symmetric(lhs, :L)
        F = ldlt(lhs_symm, check = false)
        if !issuccess(F)
            F = ldlt(lhs_symm, shift = eps(T), check = true)
        end
        rhs .= F \ rhs
    else
        if !hyp_bk_solve!(system_solver.solvecache, system_solver.solvesol, lhs, rhs)
            @warn("numerical failure: could not fix linear solve failure (mu is $(solver.mu))")
        end
        copyto!(rhs, system_solver.solvesol)
    end

    if !system_solver.use_hess_inv
        for k in eachindex(cones)
            cone_k = cones[k]
            if !Cones.use_dual(cone_k)
                # recover z_k = mu*H_k*w_k
                zk = system_solver.z_k[k]
                zcopyk = system_solver.zcopy_k[k]
                Cones.hess_prod!(zcopyk, zk, cone_k)
                copyto!(zk, zcopyk)
            end
        end
    end

    return lift_twice(solver, x1, y1, z1, x2, y2, z2, s2, x3, y3, z3, s3)
end

# lift to get tau, s, and kap
function lift_twice(solver::Solver{T}, x1, y1, z1, x2, y2, z2, s2, x3, y3, z3, s3) where {T <: Real}
    model = solver.model
    tau_denom = solver.mu / solver.tau / solver.tau - dot(model.c, x1) - dot(model.b, y1) - dot(model.h, z1)

    (tau_pred, kap_pred) = lift_once(solver, tau_denom, x1, y1, z1, x2, y2, z2, s2, solver.kap + solver.primal_obj_t - solver.dual_obj_t, -solver.kap)
    @. s2 -= solver.z_residual

    (tau_corr, kap_corr) = lift_once(solver, tau_denom, x1, y1, z1, x3, y3, z3, s3, zero(T), -solver.kap + solver.mu / solver.tau)

    return (x2, x3, y2, y3, z2, z3, tau_pred, tau_corr, s2, s3, kap_pred, kap_corr)
end

function lift_once(solver::Solver{T}, tau_denom, x1, y1, z1, x, y, z, s, tau_rhs, kap_rhs) where {T <: Real}
    model = solver.model
    tau_sol = (tau_rhs + kap_rhs + dot(model.c, x) + dot(model.b, y) + dot(model.h, z)) / tau_denom

    @. x += tau_sol * x1
    @. y += tau_sol * y1
    @. z += tau_sol * z1

    copyto!(s, model.h)
    mul!(s, model.G, x, -one(T), tau_sol)

    kap_sol = -dot(model.c, x) - dot(model.b, y) - dot(model.h, z) - tau_rhs

    return (tau_sol, kap_sol)
end
