#=
Copyright 2018, Chris Coey and contributors

QR+Cholesky linear system solver
requires precomputed QR factorization of A'
solves linear system in naive.jl by first eliminating s, kap, and tau via the method in the symindef solver, then reducing the 3x3 symmetric indefinite system to a series of low-dimensional operations via a procedure similar to that described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf (the dominating subroutine is a positive definite linear solve with RHS of dimension n-p x 3)
=#

mutable struct QRCholSystemSolver{T <: Real} <: SystemSolver{T}
    use_iterative::Bool
    use_sparse::Bool

    lhs
    fact_cache

    GQ1
    GQ2
    QpbxGHbz
    Q1pbxGHbz
    Q2div
    GQ1x
    HGQ1x
    HGQ2
    Gx
    HGx
    HGQ1x_k
    GQ1x_k
    HGQ2_k
    GQ2_k
    HGx_k
    Gx_k

    function QRCholSystemSolver{T}(; use_iterative::Bool = false, use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
        system_solver.use_sparse = use_sparse
        return system_solver
    end
end

function load(system_solver::QRCholSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cone_idxs = model.cone_idxs

    if system_solver.use_iterative
        error("not implemented")
    else
        if !isa(model.G, Matrix{T}) && isa(solver.Ap_Q, SuiteSparse.SPQR.QRSparseQ)
            # TODO very inefficient method used for sparse G * QRSparseQ : see https://github.com/JuliaLang/julia/issues/31124#issuecomment-501540818
            # TODO remove workaround and warning
            @warn("in QRChol, converting G to dense before multiplying by sparse Householder Q due to very inefficient dispatch")
            GQ = Matrix(model.G) * solver.Ap_Q
        else
            GQ = model.G * solver.Ap_Q
        end
        system_solver.GQ1 = GQ[:, 1:p]
        system_solver.GQ2 = GQ[:, (p + 1):end]

        nmp = n - p
        if system_solver.use_sparse
            if !issparse(GQ)
                error("to use sparse factorization for direction finding, cannot use dense A or G matrices (GQ is of type $(typeof(GQ)))")
            end
            system_solver.HGQ2 = spzeros(T, q, nmp)
            system_solver.lhs = spzeros(T, nmp, nmp)
        else
            system_solver.HGQ2 = Matrix{T}(undef, q, nmp)
            system_solver.lhs = Matrix{T}(undef, nmp, nmp)
        end
    end

    system_solver.QpbxGHbz = Matrix{T}(undef, n, 3)
    system_solver.Q1pbxGHbz = view(system_solver.QpbxGHbz, 1:p, :)
    system_solver.Q2div = view(system_solver.QpbxGHbz, (p + 1):n, :)
    system_solver.GQ1x = Matrix{T}(undef, q, 3)
    system_solver.HGQ1x = similar(system_solver.GQ1x)
    system_solver.Gx = similar(system_solver.GQ1x)
    system_solver.HGx = similar(system_solver.Gx)
    system_solver.HGQ1x_k = [view(system_solver.HGQ1x, idxs, :) for idxs in cone_idxs]
    system_solver.GQ1x_k = [view(system_solver.GQ1x, idxs, :) for idxs in cone_idxs]
    system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in cone_idxs]
    system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in cone_idxs]
    system_solver.HGx_k = [view(system_solver.HGx, idxs, :) for idxs in cone_idxs]
    system_solver.Gx_k = [view(system_solver.Gx, idxs, :) for idxs in cone_idxs]

    return system_solver
end

# update the LHS factorization to prepare for solve
function update_fact(system_solver::QRCholSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.use_iterative && return system_solver

    isempty(system_solver.Q2div) && return system_solver

    block_hessian_product(solver.model.cones, system_solver.HGQ2_k, system_solver.GQ2_k)
    mul!(system_solver.lhs, system_solver.GQ2', system_solver.HGQ2)

    lhs_psd = Symmetric(system_solver.lhs, :U)
    if system_solver.use_sparse
        system_solver.fact_cache = ldlt(lhs_psd, shift = sqrt(eps(T)))
    else
        set_min_diag!(system_solver.lhs, sqrt(eps(T)))
        system_solver.fact_cache = (T == BigFloat ? cholesky!(lhs_psd) : bunchkaufman!(lhs_psd))
    end

    return system_solver
end

# solve system without outer iterative refinement
function solve_system(system_solver::QRCholSystemSolver{T}, solver::Solver{T}, sol, rhs) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = n + p + q + 1

    # TODO in-place
    x = zeros(T, n, 3)
    @. @views x[:, 1:2] = rhs[1:n, :]
    @. @views x[:, 3] = -model.c
    x_sub1 = @view x[1:p, :]
    x_sub2 = @view x[(p + 1):end, :]

    y = zeros(T, p, 3)
    @. @views y[:, 1:2] = -rhs[n .+ (1:p), :]
    @. @views y[:, 3] = model.b

    z = zeros(T, q, 3)

    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        s_rows_k = tau_row .+ idxs_k
        zk12 = @view rhs[z_rows_k, :]
        sk12 = @view rhs[s_rows_k, :]
        hk = @view model.h[idxs_k]
        zk12_new = @view z[idxs_k, 1:2]
        zk3_new = @view z[idxs_k, 3]

        if Cones.use_dual(cone_k)
            zk12_temp = -zk12 - sk12 # TODO in place
            Cones.inv_hess_prod!(zk12_new, zk12_temp, cone_k)
            Cones.inv_hess_prod!(zk3_new, hk, cone_k)
        else
            Cones.hess_prod!(zk12_new, zk12, cone_k)
            @. zk12_new *= -1
            @. zk12_new -= sk12
            Cones.hess_prod!(zk3_new, hk, cone_k)
        end
    end

    ldiv!(solver.Ap_R', y)

    copyto!(system_solver.QpbxGHbz, x) # TODO can be avoided
    mul!(system_solver.QpbxGHbz, model.G', z, true, true)
    lmul!(solver.Ap_Q', system_solver.QpbxGHbz)

    copyto!(x_sub1, y)

    if !isempty(system_solver.Q2div)
        mul!(system_solver.GQ1x, system_solver.GQ1, y)
        block_hessian_product(model.cones, system_solver.HGQ1x_k, system_solver.GQ1x_k)
        mul!(system_solver.Q2div, system_solver.GQ2', system_solver.HGQ1x, -1, true)

        if system_solver.use_iterative
            error("not implemented")
            # for j in 1:size(Q2div, 2)
            #     rhs_j = view(Q2div, :, j)
            #     sol_j = view(x_sub2, :, j)
            #     IterativeSolvers.minres!(sol_j, system_solver.lhs, rhs_j, restart = size(Q2div, 1))
            # end
        else
            if system_solver.use_sparse
                x_sub2 .= system_solver.fact_cache \ system_solver.Q2div
            else
                # if !hyp_bk_solve!(system_solver.fact_cache, x_sub2, lhs, Q2div)
                #     @warn("numerical failure: could not fix linear solve failure (mu is $(solver.mu))")
                # end
                ldiv!(x_sub2, system_solver.fact_cache, system_solver.Q2div)
            end
        end
    end

    lmul!(solver.Ap_Q, x)

    mul!(system_solver.Gx, model.G, x)
    block_hessian_product(model.cones, system_solver.HGx_k, system_solver.Gx_k)

    @. z = system_solver.HGx - z

    if !isempty(y)
        copyto!(y, system_solver.Q1pbxGHbz)
        mul!(y, system_solver.GQ1', system_solver.HGx, -1, true)
        ldiv!(solver.Ap_R, y)
    end

    x3 = @view x[:, 3]
    y3 = @view y[:, 3]
    z3 = @view z[:, 3]
    x12 = @view x[:, 1:2]
    y12 = @view y[:, 1:2]
    z12 = @view z[:, 1:2]

    # lift to get tau
    # TODO maybe use higher precision here
    tau_denom = solver.mu / solver.tau / solver.tau - dot(model.c, x3) - dot(model.b, y3) - dot(model.h, z3)
    tau = @view sol[tau_row:tau_row, :]
    @. @views tau = rhs[tau_row:tau_row, :] + rhs[end:end, :]
    tau .+= model.c' * x12 + model.b' * y12 + model.h' * z12 # TODO in place
    @. tau /= tau_denom

    @. x12 += tau * x3
    @. y12 += tau * y3
    @. z12 += tau * z3

    @views sol[1:n, :] = x12
    @views sol[n .+ (1:p), :] = y12
    @views sol[(n + p) .+ (1:q), :] = z12

    # lift to get s and kap
    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(tau_row + 1):(end - 1), :]
    mul!(s, model.h, tau)
    mul!(s, model.G, sol[1:n, :], -one(T), true)
    @. @views s -= rhs[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs[end:end, :]

    return sol
end

function block_hessian_product(cones, prod_k, arr_k)
    for (k, cone_k) in enumerate(cones)
        if Cones.use_dual(cone_k)
            Cones.inv_hess_prod!(prod_k[k], arr_k[k], cone_k)
        else
            Cones.hess_prod!(prod_k[k], arr_k[k], cone_k)
        end
    end
end
