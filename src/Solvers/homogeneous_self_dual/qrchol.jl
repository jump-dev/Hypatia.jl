#=
Copyright 2018, Chris Coey and contributors

QR+Cholesky linear system solver
requires precomputed QR factorization of A'
solves linear system in naive.jl by first eliminating s, kap, and tau via the method in the symindef solver, then reducing the 3x3 symmetric indefinite system to a series of low-dimensional operations via a procedure similar to that described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf (the dominating subroutine is a positive definite linear solve with RHS of dimension n-p x 3)
=#

mutable struct QRCholSystemSolver{T <: Real} <: SystemSolver{T}
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

    function QRCholSystemSolver{T}(; use_iterative::Bool = false, use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
        system_solver.use_sparse = use_sparse
        return system_solver
    end
end

function load(system_solver::QRCholSystemSolver{T}, solver::Solver{T}) where {T <: Real}
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
    if system_solver.use_sparse
        if !issparse(GQ)
            error("to use sparse factorization for direction finding, cannot use dense A or G matrices (GQ is of type $(typeof(GQ)))")
        end
        system_solver.lhs = spzeros(T, nmp, nmp)
        system_solver.HGQ2 = spzeros(T, q, nmp)
    else
        system_solver.lhs = Matrix{T}(undef, nmp, nmp)
        system_solver.HGQ2 = Matrix{T}(undef, q, nmp)
    end
    system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in cone_idxs]
    system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in cone_idxs]

    # system_solver.QpbxGHbz = Matrix{T}(undef, n, 3)
    # system_solver.Q1pbxGHbz = view(system_solver.QpbxGHbz, 1:p, :)
    # system_solver.Q2div = view(system_solver.QpbxGHbz, (p + 1):n, :)
    # system_solver.Rpib = solver.Ap_R' \ model.b
    # system_solver.GQ1x = zeros(T, q, 3)
    # @views mul!(system_solver.GQ1x[:, 1], system_solver.GQ1, system_solver.Rpib)
    # system_solver.HGQ1x = similar(system_solver.GQ1x)
    # system_solver.Gxi = similar(system_solver.GQ1x)
    # system_solver.HGxi = similar(system_solver.Gxi)
    #
    # system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in cone_idxs]
    # system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in cone_idxs]
    # system_solver.Gxi_k = [view(system_solver.Gxi, idxs, :) for idxs in cone_idxs]
    # system_solver.HGxi_k = [view(system_solver.HGxi, idxs, :) for idxs in cone_idxs]
    #
    # system_solver.xi11 = view(system_solver.xi, 1:p, 1)
    # system_solver.xi12 = view(system_solver.xi, 1:p, 2)
    # system_solver.xi13 = view(system_solver.xi, 1:p, 3)
    # system_solver.QpbxGHbz1 = view(system_solver.QpbxGHbz, :, 1)
    # system_solver.QpbxGHbz2 = view(system_solver.QpbxGHbz, :, 2)
    # system_solver.QpbxGHbz3 = view(system_solver.QpbxGHbz, :, 3)
    # system_solver.GQ1x2 = view(system_solver.GQ1x, :, 2)
    # system_solver.GQ1x12_k = [view(system_solver.GQ1x, idxs, 1:2) for idxs in cone_idxs]
    # system_solver.HGQ1x12_k = [view(system_solver.HGQ1x, idxs, 1:2) for idxs in cone_idxs]
    # system_solver.HGQ1x12 = view(system_solver.HGQ1x, :, 1:2)
    # system_solver.Q2div12 = view(system_solver.Q2div, :, 1:2)
    #
    # if !system_solver.use_sparse
    #     system_solver.solvesol = Matrix{T}(undef, nmp, 3)
    #     system_solver.solvecache = HypBKSolveCache('U', system_solver.Q2GHGQ2)
    # end

    return system_solver
end

# for iterative methods, build block matrix for efficient multiplication
function setup_block_lhs(system_solver::QRCholSystemSolver{T}) where {T <: Real}
    error("not implemented")
end

# update the LHS factorization to prepare for solve
function update_fact(system_solver::QRCholSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    lhs = system_solver.lhs

    if !system_solver.use_sparse
        copyto!(lhs, system_solver.lhs_copy)
    end

    block_hessian_product(cones, system_solver.HGQ2_k, system_solver.GQ2_k)
    mul!(lhs, system_solver.GQ2', system_solver.HGQ2)

    # factorize positive definite LHS
    lhs_psd = Symmetric(lhs, :U)
    if system_solver.use_sparse
        system_solver.fact_cache = cholesky(lhs_psd, check = false) # TODO or ldlt
        if !issuccess(F) # TODO maybe just use shift above and remove this
            system_solver.fact_cache = ldlt(lhs_psd, shift = eps(T))
        end
    else
        set_min_diag!(lhs_psd, sqrt(eps(T)))
        system_solver.fact_cache = (T == BigFloat ? cholesky!(lhs_psd) : bunchkaufman!(lhs_psd))
    end

    return system_solver
end

# solve system without outer iterative refinement
function solve_system(system_solver::QRCholSystemSolver{T}, sol_curr, rhs_curr) where {T <: Real}
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

function block_hessian_product(cones, prod_k, arr_k)
    for (k, cone_k) in enumerate(cones)
        if Cones.use_dual(cone_k)
            Cones.inv_hess_prod!(prod_k[k], arr_k[k], cone_k)
        else
            Cones.hess_prod!(prod_k[k], arr_k[k], cone_k)
        end
    end
end


#
# mutable struct QRCholSystemSolver{T <: Real} <: SystemSolver{T}
#     use_sparse::Bool
#
#     solver::Solver{T}
#
#     xi::Matrix{T}
#     yi::Matrix{T}
#     zi::Matrix{T}
#
#     x1
#     y1
#     z1
#     xi2
#     x2
#     y2
#     z2
#     x3
#     y3
#     z3
#     z_k
#     z2_k
#     z3_k
#     z1_temp
#     z2_temp
#     z3_temp
#     z_temp_k
#     z2_temp_k
#     z3_temp_k
#
#     GQ1
#     GQ2
#     QpbxGHbz
#     Q1pbxGHbz
#     Q2div
#     Rpib
#     GQ1x
#     HGQ1x
#     HGQ2
#     Q2GHGQ2
#     Gxi
#     HGxi
#
#     GQ2_k
#     HGQ2_k
#     Gxi_k
#     HGxi_k
#
#     xi11
#     xi12
#     xi13
#     QpbxGHbz1
#     QpbxGHbz2
#     QpbxGHbz3
#     GQ1x2
#     GQ1x12_k
#     HGQ1x12_k
#     HGQ1x12
#     Q2div12
#
#     solvesol
#     solvecache
#
#     function QRCholSystemSolver{T}(; use_sparse::Bool = false) where {T <: Real}
#         system_solver = new{T}()
#         system_solver.use_sparse = use_sparse
#         return system_solver
#     end
# end
#
# function load(system_solver::QRCholSystemSolver{T}, solver::Solver{T}) where {T <: Real}
#     system_solver.solver = solver
#
#     model = solver.model
#     (n, p, q) = (model.n, model.p, model.q)
#     nmp = n - p
#     cones = model.cones
#     cone_idxs = model.cone_idxs
#
#     xi = Matrix{T}(undef, n, 3)
#     yi = Matrix{T}(undef, p, 3)
#     zi = Matrix{T}(undef, q, 3)
#     system_solver.xi = xi
#     system_solver.yi = yi
#     system_solver.zi = zi
#
#     system_solver.xi2 = view(xi, (p + 1):n, :)
#     system_solver.x1 = view(xi, :, 1)
#     system_solver.y1 = view(yi, :, 1)
#     system_solver.z1 = view(zi, :, 1)
#     system_solver.x2 = view(xi, :, 2)
#     system_solver.y2 = view(yi, :, 2)
#     system_solver.z2 = view(zi, :, 2)
#     system_solver.x3 = view(xi, :, 3)
#     system_solver.y3 = view(yi, :, 3)
#     system_solver.z3 = view(zi, :, 3)
#     system_solver.z_k = [Cones.use_dual(cone_k) ? view(zi, idxs, :) : view(zi, idxs, 1:2) for (cone_k, idxs) in zip(cones, cone_idxs)]
#     system_solver.z2_k = [view(zi, idxs, 2) for idxs in cone_idxs]
#     system_solver.z3_k = [view(zi, idxs, 3) for idxs in cone_idxs]
#     zi_temp = similar(zi)
#     system_solver.z1_temp = view(zi_temp, :, 1)
#     system_solver.z2_temp = view(zi_temp, :, 2)
#     system_solver.z3_temp = view(zi_temp, :, 3)
#     system_solver.z_temp_k = [Cones.use_dual(cone_k) ? view(zi_temp, idxs, :) : view(zi_temp, idxs, 1:2) for (cone_k, idxs) in zip(cones, cone_idxs)]
#     system_solver.z2_temp_k = [view(zi_temp, idxs, 2) for idxs in cone_idxs]
#     system_solver.z3_temp_k = [view(zi_temp, idxs, 3) for idxs in cone_idxs]
#
#
#     if !isa(model.G, Matrix{T}) && isa(solver.Ap_Q, SuiteSparse.SPQR.QRSparseQ)
#         # TODO very inefficient method used for sparse G * QRSparseQ : see https://github.com/JuliaLang/julia/issues/31124#issuecomment-501540818
#         # TODO remove workaround and warning
#         @warn("in QRChol, converting G to dense before multiplying by sparse Householder Q due to very inefficient dispatch")
#         GQ = Matrix(model.G) * solver.Ap_Q
#     else
#         GQ = model.G * solver.Ap_Q
#     end
#     system_solver.GQ1 = GQ[:, 1:p]
#     system_solver.GQ2 = GQ[:, (p + 1):end]
#     if system_solver.use_sparse
#         if !issparse(GQ)
#             error("to use sparse factorization for direction finding, cannot use dense A or G matrices (GQ is of type $(typeof(GQ)))")
#         end
#         system_solver.HGQ2 = spzeros(T, q, nmp)
#         system_solver.Q2GHGQ2 = spzeros(T, nmp, nmp)
#     else
#         system_solver.HGQ2 = Matrix{T}(undef, q, nmp)
#         system_solver.Q2GHGQ2 = Matrix{T}(undef, nmp, nmp)
#     end
#     system_solver.QpbxGHbz = Matrix{T}(undef, n, 3)
#     system_solver.Q1pbxGHbz = view(system_solver.QpbxGHbz, 1:p, :)
#     system_solver.Q2div = view(system_solver.QpbxGHbz, (p + 1):n, :)
#     system_solver.Rpib = solver.Ap_R' \ model.b
#     system_solver.GQ1x = zeros(T, q, 3)
#     @views mul!(system_solver.GQ1x[:, 1], system_solver.GQ1, system_solver.Rpib)
#     system_solver.HGQ1x = similar(system_solver.GQ1x)
#     system_solver.Gxi = similar(system_solver.GQ1x)
#     system_solver.HGxi = similar(system_solver.Gxi)
#
#     system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in cone_idxs]
#     system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in cone_idxs]
#     system_solver.Gxi_k = [view(system_solver.Gxi, idxs, :) for idxs in cone_idxs]
#     system_solver.HGxi_k = [view(system_solver.HGxi, idxs, :) for idxs in cone_idxs]
#
#     system_solver.xi11 = view(system_solver.xi, 1:p, 1)
#     system_solver.xi12 = view(system_solver.xi, 1:p, 2)
#     system_solver.xi13 = view(system_solver.xi, 1:p, 3)
#     system_solver.QpbxGHbz1 = view(system_solver.QpbxGHbz, :, 1)
#     system_solver.QpbxGHbz2 = view(system_solver.QpbxGHbz, :, 2)
#     system_solver.QpbxGHbz3 = view(system_solver.QpbxGHbz, :, 3)
#     system_solver.GQ1x2 = view(system_solver.GQ1x, :, 2)
#     system_solver.GQ1x12_k = [view(system_solver.GQ1x, idxs, 1:2) for idxs in cone_idxs]
#     system_solver.HGQ1x12_k = [view(system_solver.HGQ1x, idxs, 1:2) for idxs in cone_idxs]
#     system_solver.HGQ1x12 = view(system_solver.HGQ1x, :, 1:2)
#     system_solver.Q2div12 = view(system_solver.Q2div, :, 1:2)
#
#     if !system_solver.use_sparse
#         system_solver.solvesol = Matrix{T}(undef, nmp, 3)
#         system_solver.solvecache = HypBKSolveCache('U', system_solver.Q2GHGQ2)
#     end
#
#     return system_solver
# end
#
# function get_combined_directions(system_solver::QRCholSystemSolver{T}) where {T <: Real}
#     solver = system_solver.solver
#     model = solver.model
#     cones = model.cones
#     xi = system_solver.xi
#     yi = system_solver.yi
#     zi = system_solver.zi
#     xi2 = system_solver.xi2
#     x1 = system_solver.x1
#     y1 = system_solver.y1
#     z1 = system_solver.z1
#     x2 = system_solver.x2
#     y2 = system_solver.y2
#     z2 = system_solver.z2
#     x3 = system_solver.x3
#     y3 = system_solver.y3
#     z3 = system_solver.z3
#     z_k = system_solver.z_k
#     z2_k = system_solver.z2_k
#     z3_k = system_solver.z3_k
#     z1_temp = system_solver.z1_temp
#     z2_temp = system_solver.z2_temp
#     z3_temp = system_solver.z3_temp
#     z_temp_k = system_solver.z_temp_k
#     z2_temp_k = system_solver.z2_temp_k
#     z3_temp_k = system_solver.z3_temp_k
#     GQ1 = system_solver.GQ1
#     GQ2 = system_solver.GQ2
#     QpbxGHbz = system_solver.QpbxGHbz
#     Q1pbxGHbz = system_solver.Q1pbxGHbz
#     Q2div = system_solver.Q2div
#     Rpib = system_solver.Rpib
#     GQ1x = system_solver.GQ1x
#     HGQ1x = system_solver.HGQ1x
#     HGQ2 = system_solver.HGQ2
#     Q2GHGQ2 = system_solver.Q2GHGQ2
#     Gxi = system_solver.Gxi
#     HGxi = system_solver.HGxi
#     GQ2_k = system_solver.GQ2_k
#     HGQ2_k = system_solver.HGQ2_k
#     Gxi_k = system_solver.Gxi_k
#     HGxi_k = system_solver.HGxi_k
#     xi11 = system_solver.xi11
#     xi12 = system_solver.xi12
#     xi13 = system_solver.xi13
#     QpbxGHbz1 = system_solver.QpbxGHbz1
#     QpbxGHbz2 = system_solver.QpbxGHbz2
#     QpbxGHbz3 = system_solver.QpbxGHbz3
#     GQ1x2 = system_solver.GQ1x2
#     GQ1x12_k = system_solver.GQ1x12_k
#     HGQ1x12_k = system_solver.HGQ1x12_k
#     HGQ1x12 = system_solver.HGQ1x12
#     Q2div12 = system_solver.Q2div12
#
#     sqrtmu = sqrt(solver.mu)
#
#     # solve 3x3 system
#     @. z1_temp = model.h
#     @. z2_temp = -solver.z_residual
#     for (k, cone_k) in enumerate(cones)
#         duals_k = solver.point.dual_views[k]
#         grad_k = Cones.grad(cone_k)
#         if Cones.use_dual(cone_k)
#             @. z2_temp_k[k] += duals_k
#             @. z3_temp_k[k] = duals_k + grad_k * sqrtmu
#             Cones.inv_hess_prod!(z_k[k], z_temp_k[k], cone_k)
#         else
#             Cones.hess_prod!(z_k[k], z_temp_k[k], cone_k)
#             @. z2_k[k] += duals_k
#             @. z3_k[k] = duals_k + grad_k * sqrtmu
#         end
#     end
#
#     @. xi11 = Rpib
#     @. xi12 = -solver.y_residual
#     ldiv!(solver.Ap_R', xi12)
#     @. xi13 = zero(T)
#
#     @. QpbxGHbz1 = -model.c
#     @. QpbxGHbz2 = solver.x_residual
#     @. QpbxGHbz3 = zero(T)
#     mul!(QpbxGHbz, model.G', zi, true, true)
#     lmul!(solver.Ap_Q', QpbxGHbz)
#
#     if !isempty(Q2div)
#         mul!(GQ1x2, GQ1, xi12)
#         block_hessian_product(cones, HGQ1x12_k, GQ1x12_k)
#         mul!(Q2div12, GQ2', HGQ1x12, -one(T), true)
#
#         block_hessian_product(cones, HGQ2_k, GQ2_k)
#         mul!(Q2GHGQ2, GQ2', HGQ2)
#
#         if system_solver.use_sparse
#             F = ldlt(Symmetric(Q2GHGQ2), check = false)
#             if !issuccess(F)
#                 @warn("sparse linear system matrix factorization failed")
#                 mul!(Q2GHGQ2, GQ2', HGQ2)
#                 F = ldlt(Symmetric(Q2GHGQ2), shift = cbrt(eps(T)), check = false)
#                 if !issuccess(F)
#                     @warn("numerical failure: could not fix failure of positive definiteness (mu is $(solver.mu)")
#                 end
#             end
#             xi2 .= F \ Q2div # TODO eliminate allocs (see https://github.com/JuliaLang/julia/issues/30084)
#         else
#             if !hyp_bk_solve!(system_solver.solvecache, system_solver.solvesol, Q2GHGQ2, Q2div)
#                 @warn("dense linear system matrix factorization failed")
#                 mul!(Q2GHGQ2, GQ2', HGQ2) # TODO is this needed? not if bk doesn't destroy
#                 set_min_diag!(Q2GHGQ2, cbrt(eps(T)))
#                 if !hyp_bk_solve!(system_solver.solvecache, system_solver.solvesol, Q2GHGQ2, Q2div)
#                     @warn("numerical failure: could not fix failure of positive definiteness (mu is $(solver.mu))")
#                 end
#             end
#             copyto!(xi2, system_solver.solvesol)
#         end
#     end
#
#     lmul!(solver.Ap_Q, xi) # x finished
#
#     mul!(Gxi, model.G, xi)
#     block_hessian_product(cones, HGxi_k, Gxi_k)
#
#     axpby!(true, HGxi, -one(T), zi) # z finished
#
#     if !isempty(yi)
#         mul!(yi, GQ1', HGxi)
#         axpby!(true, Q1pbxGHbz, -one(T), yi)
#         ldiv!(solver.Ap_R, yi) # y finished
#     end
#
#     return lift_twice(solver, x1, y1, z1, x2, y2, z2, z2_temp, x3, y3, z3, z3_temp)
# end
