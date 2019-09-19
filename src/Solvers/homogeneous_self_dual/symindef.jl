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
- improve numerics of method with use_inv_hess = false
=#

mutable struct SymIndefSystemSolver{T <: Real} <: SystemSolver{T}
    solver::Solver{T}
    use_iterative::Bool
    use_sparse::Bool
    use_inv_hess::Bool

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

    function SymIndefSystemSolver{T}(; use_iterative::Bool = false, use_sparse::Bool = false, use_inv_hess::Bool = true) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
        system_solver.use_sparse = use_sparse
        system_solver.use_inv_hess = use_inv_hess
        return system_solver
    end
end

function load(system_solver::SymIndefSystemSolver{T}, solver::Solver{T}) where {T <: Real}
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

    # symmetric, lower triangle filled only
    if system_solver.use_iterative
        system_solver.lhs = setup_block_lhs(system_solver)
    else
        if system_solver.use_sparse
            system_solver.lhs = T[
                spzeros(T,n,n)  spzeros(T,n,p)  spzeros(T,n,q);
                model.A         spzeros(T,p,p)  spzeros(T,p,q);
                model.G         spzeros(T,q,p)  sparse(-one(T)*I,q,q);
                ]
            dropzeros!(system_solver.lhs)
            @assert issparse(system_solver.lhs)
        else
            system_solver.lhs_copy = T[
                zeros(T,n,n)  zeros(T,n,p)  zeros(T,n,q);
                model.A       zeros(T,p,p)  zeros(T,p,q);
                model.G       zeros(T,q,p)  Matrix(-one(T)*I,q,q);
                ]
            system_solver.lhs = similar(system_solver.lhs_copy)
            # system_solver.fact_cache = HypLUSolveCache(system_solver.sol, system_solver.lhs, rhs)
        end
    end

    return system_solver
end

# for iterative methods, build block matrix for efficient multiplication
function setup_block_lhs(system_solver::SymIndefSystemSolver{T}) where {T <: Real}
    error("not implemented")
end

# update the LHS factorization to prepare for solve
function update_fact(system_solver::SymIndefSystemSolver{T}) where {T <: Real}
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
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            H = Cones.hess(cone_k)
            @. lhs[z_rows_k, z_rows_k] = -H
        elseif system_solver.use_inv_hess
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Hinv = Cones.inv_hess(cone_k)
            @. lhs[z_rows_k, z_rows_k] = -Hinv
        else
            # A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
            # mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
            H = Cones.hess(cone_k)
            @. lhs[z_rows_k, z_rows_k] = -H
            @views Cones.hess_prod!(lhs[z_rows_k, 1:n], model.G[idxs_k, :], cone_k)
        end
    end

    # factorize symmetric indefinite LHS
    lhs_symm = Symmetric(lhs, :L)
    if system_solver.use_sparse
        system_solver.fact_cache = ldlt(lhs_symm, check = false)
        if !issuccess(F) # TODO maybe just use shift above and remove this
            system_solver.fact_cache = ldlt(lhs_symm, shift = eps(T))
        end
    else
        system_solver.fact_cache = (T == BigFloat ? lu!(lhs_symm) : bunchkaufman!(lhs_symm))
    end

    return system_solver
end

# solve system without outer iterative refinement
function solve_system(system_solver::SymIndefSystemSolver{T}, sol_curr, rhs_curr) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = system_solver.tau_row

    # TODO in-place
    dim3 = tau_row - 1
    sol3 = zeros(T, dim3, 3)
    rhs3 = zeros(T, dim3, 3)

    @. @views rhs3[1:n, 1:2] = rhs_curr[1:n, :]
    @. @views rhs3[n .+ (1:p), 1:2] = -rhs_curr[n .+ (1:p), :]
    @. rhs3[1:n, 3] = -model.c
    @. rhs3[n .+ (1:p), 3] = model.b

    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        s_rows_k = tau_row .+ idxs_k
        zk12 = @view rhs_curr[z_rows_k, :]
        sk12 = @view rhs_curr[s_rows_k, :]
        hk = @view model.h[idxs_k]
        zk12_new = @view rhs3[z_rows_k, 1:2]
        zk3_new = @view rhs3[z_rows_k, 3]

        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            @. zk12_new = -zk12 - sk12
            @. zk3_new = hk
        elseif system_solver.use_inv_hess
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Cones.inv_hess_prod!(zk12_new, sk12, cone_k)
            @. zk12_new *= -1
            @. zk12_new -= zk12
            @. zk3_new = hk
        else
            # A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
            # mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
            Cones.hess_prod!(zk12_new, zk12, cone_k)
            @. zk12_new *= -1
            @. zk12_new -= sk12
            Cones.hess_prod!(zk3_new, hk, cone_k)
        end
    end

    if system_solver.use_iterative
        error("not implemented")
        # for j in 1:size(rhs3, 2)
        #     rhs_j = view(rhs3, :, j)
        #     sol_j = view(sol3, :, j)
        #     IterativeSolvers.minres!(sol_j, system_solver.lhs, rhs_j, restart = tau_row)
        # end
    else
        if system_solver.use_sparse
            sol3 .= system_solver.fact_cache \ rhs3
        else
            # if !hyp_bk_solve!(system_solver.fact_cache, sol3, lhs, rhs3)
            #     @warn("numerical failure: could not fix linear solve failure (mu is $(solver.mu))")
            # end
            ldiv!(sol3, system_solver.fact_cache, rhs3)
        end
    end

    if !system_solver.use_inv_hess
        for (k, cone_k) in enumerate(model.cones)
            if !Cones.use_dual(cone_k)
                # recover z_k = mu*H_k*w_k
                z_rows_k = (n + p) .+ model.cone_idxs[k]
                z_copy_k = sol3[z_rows_k, :] # TODO do in-place
                @views Cones.hess_prod!(sol3[z_rows_k, :], z_copy_k, cone_k)
            end
        end
    end

    # lift to get tau
    x3 = @view sol3[1:n, 3]
    y3 = @view sol3[n .+ (1:p), 3]
    z3 = @view sol3[(n + p) .+ (1:q), 3]
    x = @view sol3[1:n, 1:2]
    y = @view sol3[n .+ (1:p), 1:2]
    z = @view sol3[(n + p) .+ (1:q), 1:2]

    # TODO maybe use higher precision here
    tau_denom = solver.mu / solver.tau / solver.tau - dot(model.c, x3) - dot(model.b, y3) - dot(model.h, z3)
    tau = @view sol_curr[tau_row:tau_row, :]
    @. @views tau = rhs_curr[tau_row:tau_row, :] + rhs_curr[end:end, :]
    tau .+= model.c' * x + model.b' * y + model.h' * z # TODO in place
    @. tau /= tau_denom

    @. x += tau * x3
    @. y += tau * y3
    @. z += tau * z3

    @views sol_curr[1:dim3, :] = sol3[:, 1:2]

    # lift to get s and kap
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
