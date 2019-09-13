#=
Copyright 2018, Chris Coey and contributors

symmetric-indefinite linear system solver
solves linear system in naive.jl via the following procedure

A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
(pr bar) -G_k*x + (mu*H_k)^-1*z_k + h_k*tau = zrhs_k + (mu*H_k)^-1*srhs_k
(du bar) -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs

optionally avoid use of inverse Hessians, by replacing the (pr bar) constraints with
(pr bar) -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
i.e. premultiplying by mu*H_k

eliminate tau via a procedure similar to that described by S7.4 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
and symmetrize the LHS matrix by multiplying some equations by -1

TODO
- add iterative method
- improve numerics of method with use_hess_inv = false (possibly a bug)
=#

mutable struct SymIndefSystemSolver{T <: Real} <: SystemSolver{T}
    use_sparse::Bool
    use_hess_inv::Bool

    solver::Solver{T}

    lhs_copy
    lhs
    rhs::Matrix{T}

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
    s1::Vector{T}
    s2::Vector{T}
    s3::Vector{T}
    s1_k
    s2_k
    s3_k

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
    npq = n + p + q

    # x y z
    # symmetric, lower triangle filled only
    if system_solver.use_sparse
        system_solver.lhs_copy = T[
            spzeros(T,n,n)  spzeros(T,n,p)  spzeros(T,n,q);
            model.A         spzeros(T,p,p)  spzeros(T,p,q);
            model.G         spzeros(T,q,p)  sparse(-one(T)*I,q,q);
        ]
        @assert issparse(system_solver.lhs_copy)
    else
        system_solver.lhs_copy = T[
            zeros(T,n,n)  zeros(T,n,p)  zeros(T,n,q);
            model.A       zeros(T,p,p)  zeros(T,p,q);
            model.G       zeros(T,q,p)  Matrix(-one(T)*I,q,q);
        ]
    end

    system_solver.lhs = similar(system_solver.lhs_copy)

    rhs = zeros(T, npq, 3)
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
    system_solver.s1 = similar(rhs, q)
    system_solver.s2 = similar(rhs, q)
    system_solver.s3 = similar(rhs, q)

    if !system_solver.use_sparse
        system_solver.solvesol = Matrix{T}(undef, size(system_solver.lhs, 1), 3)
        system_solver.solvecache = HypBKSolveCache('L', system_solver.solvesol, system_solver.lhs, system_solver.rhs)
    end

    return system_solver
end

function get_combined_directions(system_solver::SymIndefSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    use_hess_inv = system_solver.use_hess_inv
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    mu = solver.mu
    tau = solver.tau
    kap = solver.kap

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
    s1 = system_solver.s1
    s2 = system_solver.s2
    s3 = system_solver.s3

    @. x1 = -model.c
    x2 .= solver.x_residual
    x3 .= zero(T)
    y1 .= model.b
    @. y2 = -solver.y_residual
    y3 .= zero(T)
    @. z2 .= -solver.z_residual
    z3 .= zero(T)

    # update lhs matrix
    sqrtmu = sqrt(mu)
    copyto!(lhs, system_solver.lhs_copy)
    for k in eachindex(cones)
        cone_k = cones[k]
        idxs = model.cone_idxs[k]
        rows = (n + p) .+ idxs
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = zrhs_k + srhs_k
            H = Cones.hess(cone_k)
            @. lhs[rows, rows] = -H
            @. z2_k[k] += duals_k
            @. z3_k[k] = duals_k + grad_k * sqrtmu
            @views copyto!(z1_k[k], model.h[idxs])
        else
            if use_hess_inv
                # G_k*x - (mu*H_k)^-1*z_k = zrhs_k + (mu*H_k)^-1*srhs_k
                Hinv = Cones.inv_hess(cone_k)
                @. lhs[rows, rows] = -Hinv
                Cones.inv_hess_prod!(z1_k[k], duals_k, cone_k)
                @. z2_k[k] += z1_k[k]
                @. z1_k[k] = duals_k + grad_k * sqrtmu
                Cones.inv_hess_prod!(z3_k[k], z1_k[k], cone_k)
                @views copyto!(z1_k[k], model.h[idxs])
            else
                # mu*H_k*G_k*x - z_k = mu*H_k*zrhs_k + srhs_k
                H = Cones.hess(cone_k)
                @. lhs[rows, rows] = -H
                @views Cones.hess_prod!(lhs[rows, 1:n], model.G[idxs, :], cone_k)
                @views Cones.hess_prod!(z1_k[k], model.h[idxs], cone_k)
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
            @warn("numerical failure: could not fix linear solve failure (mu is $mu)")
        end
        copyto!(rhs, system_solver.solvesol)
    end

    if !use_hess_inv
        for k in eachindex(cones)
            cone_k = cones[k]
            if !Cones.use_dual(cone_k)
                # z_k is premultiplied by (mu * H_k)^-1
                zk = system_solver.z_k[k]
                zcopyk = system_solver.zcopy_k[k]
                Cones.hess_prod!(zcopyk, zk, cone_k)
                copyto!(zk, zcopyk)
            end
        end
    end

    # lift to HSDE space
    tau_denom = mu / tau / tau - dot(model.c, x1) - dot(model.b, y1) - dot(model.h, z1)

    function lift!(x, y, z, s, tau_rhs, kap_rhs)
        tau_sol = (tau_rhs + kap_rhs + dot(model.c, x) + dot(model.b, y) + dot(model.h, z)) / tau_denom
        @. x += tau_sol * x1
        @. y += tau_sol * y1
        @. z += tau_sol * z1
        copyto!(s, model.h)
        mul!(s, model.G, x, -one(T), tau_sol)
        kap_sol = -dot(model.c, x) - dot(model.b, y) - dot(model.h, z) - tau_rhs
        return (tau_sol, kap_sol)
    end

    (tau_pred, kap_pred) = lift!(x2, y2, z2, s2, kap + solver.primal_obj_t - solver.dual_obj_t, -kap)
    @. s2 -= solver.z_residual
    (tau_corr, kap_corr) = lift!(x3, y3, z3, s3, zero(T), -kap + mu / tau)

    return (x2, x3, y2, y3, z2, z3, s2, s3, tau_pred, tau_corr, kap_pred, kap_corr)
end
