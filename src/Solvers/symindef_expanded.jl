#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

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
=#


mutable struct SymIndefSparseExpandedSystemSolver{T <: Real} <: SymIndefSystemSolver{T}
    use_inv_hess::Bool
    tau_row::Int
    lhs3::SparseMatrixCSC # TODO type will depend on Int type
    rhs3::Matrix{T}
    rhs3_expanded::Matrix{T}
    sol3::Matrix{T}
    sol3_expanded::Matrix{T}
    hess_idxs::Vector
    fact_cache::SparseSymCache{T}
    function SymIndefSparseExpandedSystemSolver{Float64}(;
        use_inv_hess::Bool = true,
        fact_cache::SparseSymCache{Float64} = SparseSymCache{Float64}(),
        )
        system_solver = new{Float64}()
        if !use_inv_hess
            @warn("SymIndefSparseExpandedSystemSolver is not implemented with `use_inv_hess` set to `false`, using `true` instead.")
        end
        system_solver.use_inv_hess = true
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function solve_system(system_solver::SymIndefSparseExpandedSystemSolver{T}, solver::Solver{T}, sol::Matrix{T}, rhs::Matrix{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = system_solver.tau_row

    sol3 = system_solver.sol3
    sol3_expanded = system_solver.sol3_expanded
    rhs3_expanded = system_solver.rhs3_expanded

    # @. @views rhs3[1:n, 1:2] = rhs[1:n, :]
    # @. @views rhs3[n .+ (1:p), 1:2] = -rhs[n .+ (1:p), :]
    # @. rhs3[1:n, 3] = -model.c
    # @. rhs3[n .+ (1:p), 3] = model.b

    rhs3_expanded .= 0
    @. @views rhs3_expanded[1:n, 1:2] = rhs[1:n, :]
    @. @views rhs3_expanded[n .+ (1:p), 1:2] = -rhs[n .+ (1:p), :]
    @. rhs3_expanded[1:n, 3] = -model.c
    @. rhs3_expanded[n .+ (1:p), 3] = model.b
    @show "earlier try", rhs3_expanded

    ns = 0
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k] .+ ns
        z_rows_k = (n + p) .+ idxs_k
        s_rows_k = tau_row .+ idxs_k
        zk12 = @view rhs[z_rows_k, :]
        sk12 = @view rhs[s_rows_k, :]
        hk = @view model.h[idxs_k]
        zk12_new = @view rhs3_expanded[z_rows_k, 1:2]
        zk3_new = @view rhs3_expanded[z_rows_k, 3]

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
        if isa(cone_k, Cones.EpiNormEucl)
            ns += 1
        end
    end

    # @show "try", rhs3_expanded
    @timeit solver.timer "solve_system" solve_subsystem(system_solver, sol3_expanded, rhs3_expanded)
    # @show "try", sol3_expanded
    copyto!(sol3[1:(n + p), :], sol3_expanded[1:(n + p), :])
    idx = n + p + 1
    ns = 0
    for (k, cone_k) in enumerate(model.cones)
        dim = Cones.dimension(cone_k)
        copyto!(sol3[idx:(idx + dim - 1)], sol3_expanded[(idx + ns):(idx + ns + dim - 1)])
        idx += dim
        if isa(cone_k, Cones.EpiNormEucl)
            ns += 1
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

    x3 = @view sol3[1:n, 3]
    y3 = @view sol3[n .+ (1:p), 3]
    z3 = @view sol3[(n + p) .+ (1:q), 3]
    x12 = @view sol3[1:n, 1:2]
    y12 = @view sol3[n .+ (1:p), 1:2]
    z12 = @view sol3[(n + p) .+ (1:q), 1:2]

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

    @views sol[1:(tau_row - 1), :] = sol3[:, 1:2]

    # lift to get s and kap
    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(tau_row + 1):(end - 1), :]
    mul!(s, model.h, tau)
    mul!(s, model.G, sol[1:n, :], -one(T), true)
    @. @views s -= rhs[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs[end:end, :]
    # @show "try", sol

    return sol
end

function load(system_solver::SymIndefSparseExpandedSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    system_solver.tau_row = n + p + q + 1
    cones = model.cones
    cone_idxs = model.cone_idxs

    system_solver.sol3 = zeros(n + p + q, 3)
    system_solver.sol3_expanded = zeros(n + p + q, 3)
    system_solver.rhs3_expanded = similar(system_solver.sol3_expanded)

    ns = sum(isa(cone_k, Cones.EpiNormEucl) for cone_k in cones)

    # form sparse LHS without Hessians and inverse Hessians in z/z block
    pert = sqrt(eps(T))
    lhs3 = T[
        sparse(pert * I, n, n)    spzeros(T, n, p)         spzeros(T, n, q)           spzeros(T, n, ns);
        model.A                   sparse(pert * I, p, p)   spzeros(T, p, q)           spzeros(T, p, ns);
        model.G                   spzeros(T, q, p)         sparse(-one(T) * I, q, q)  spzeros(T, q, ns);
        spzeros(T, ns, n)         spzeros(T, ns, p)        spzeros(T, ns, q)          sparse(-one(T) * I, ns, ns)
        ]
    idx = n + p + 1
    for (k, cone_k) in enumerate(solver.model.cones)
        dim = Cones.dimension(cone_k)
        if isa(cone_k, Cones.EpiNormEucl)
            # there are already ones on the diagonal
            lhs3[(idx + dim), idx:(idx + dim - 1)] .= cone_k.point
            idx += (dim + 1)
        else
            lhs3[idx:(idx + dim - 1), idx:(idx + dim - 1)] .= ones(T, dim, dim)
            idx += dim
        end
    end
    @assert issparse(lhs3)
    dropzeros!(lhs3)
    (Is, Js, Vs) = findnz(lhs3)
    #
    # # add I, J, V for Hessians and inverse Hessians
    # hess_nz_total = 0
    # if !isempty(cones)
    #     for cone_k in cones
    #         hess_nz_total += (Cones.use_dual(cone_k) ? Cones.hess_nz_count_symindef(cone_k, true) : Cones.inv_hess_nz_count_symindef(cone_k, true))
    #     end
    # end
    # H_Is = Vector{Int}(undef, hess_nz_total)
    # H_Js = Vector{Int}(undef, hess_nz_total)
    # offset = 1
    # y_start = n + p - 1
    # ns = 0
    # for (k, cone_k) in enumerate(cones)
    #     z_start_k = y_start + first(cone_idxs_k)
    #     if isa(cone_k, Cones.EpiNormEucl)
    #         ns += 1
    #
    #         dim = Cones.dimension(cone_k)
    #         for j in 1:Cones.dimension(cone_k)
    #
    #             H_Is[offset:(offset + dim - 1)] = z_start_k:(z_start_k + Cones.dimension(cone_k))
    #             H_Js[offset:(offset + dim - 1)] =
    #             offset += len_kj
    #     else
    #         cone_idxs_k = cone_idxs[k] .+ ns
    #         for j in 1:Cones.dimension(cone_k)
    #             nz_rows_kj = z_start_k .+ (Cones.use_dual(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j, true) : Cones.inv_hess_nz_idxs_col(cone_k, j, true))
    #             len_kj = length(nz_rows_kj)
    #             IJV_idxs = offset:(offset + len_kj - 1)
    #             offset += len_kj
    #             @. H_Is[IJV_idxs] = nz_rows_kj
    #             @. H_Js[IJV_idxs] = z_start_k + j
    #         end
    #     end
    # end
    # @assert offset == hess_nz_total + 1
    # append!(Is, H_Is)
    # append!(Js, H_Js)
    # append!(Vs, ones(T, hess_nz_total))
    #
    # pert = T(system_solver.fact_cache.diag_pert) # TODO not happy with where this is stored
    # append!(Is, 1:(n + p))
    # append!(Js, 1:(n + p))
    # append!(Vs, fill(pert, n))
    # append!(Vs, fill(-pert, p))

    dim = size(lhs3, 1)
    # integer type supported by the sparse system solver library to be used
    Ti = int_type(system_solver.fact_cache)
    # prefer conversions of integer types to happen here than inside external wrappers
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    lhs3 = system_solver.lhs3 = sparse(Is, Js, Vs, dim, dim)

    # cache indices of nonzeros of Hessians and inverse Hessians in sparse LHS nonzeros vector
    # system_solver.hess_idxs = [Vector{Union{UnitRange, Vector{Int}}}(undef, Cones.dimension(cone_k)) for cone_k in cones]
    # for (k, cone_k) in enumerate(cones)
    #     cone_idxs_k = cone_idxs[k]
    #     z_start_k = y_start + first(cone_idxs_k)
    #     for j in 1:Cones.dimension(cone_k)
    #         col = z_start_k + j
    #         # get nonzero rows in the current column of the LHS
    #         col_idx_start = lhs3.colptr[col]
    #         nz_rows = lhs3.rowval[col_idx_start:(lhs3.colptr[col + 1] - 1)]
    #         # get nonzero rows in column j of the Hessian or inverse Hessian
    #         nz_hess_indices = (Cones.use_dual(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j, true) : Cones.inv_hess_nz_idxs_col(cone_k, j, true))
    #         # get index corresponding to first nonzero Hessian element of the current column of the LHS
    #         first_H = findfirst(isequal(z_start_k + first(nz_hess_indices)), nz_rows)
    #         # indices of nonzero values for cone k column j
    #         system_solver.hess_idxs[k][j] = (col_idx_start + first_H - 2) .+ (1:length(nz_hess_indices))
    #     end
    # end

    return system_solver
end

function update_fact(system_solver::SymIndefSparseExpandedSystemSolver, solver::Solver)
    # for (k, cone_k) in enumerate(solver.model.cones)
    #     if isa(cone_k, Cones.EpiNormEucl)
    #
    #     else
    #         H = (Cones.use_dual(cone_k) ? Cones.hess(cone_k) : Cones.inv_hess(cone_k))
    #         for j in 1:Cones.dimension(cone_k)
    #             nz_rows = (Cones.use_dual(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j, true) : Cones.inv_hess_nz_idxs_col(cone_k, j, true))
    #             @. @views system_solver.lhs3.nzval[system_solver.hess_idxs[k][j]] = -H[nz_rows, j]
    #         end
    #     end
    # end

    idx = solver.model.n + solver.model.p + 1
    for (k, cone_k) in enumerate(solver.model.cones)
        dim = Cones.dimension(cone_k)
        if isa(cone_k, Cones.EpiNormEucl)
            Cones.update_feas(cone_k)
            system_solver.lhs3[idx, idx] = -cone.dist
            for j in (idx + 1):(idx + dim - 1)
                system_solver.lhs3[j, j] = cone_k.dist
            end
            system_solver.lhs3[(idx + dim), (idx + dim)] = -1
            system_solver.lhs3[(idx + dim), idx:(idx + dim - 1)] .= cone_k.point
            idx += (dim + 1)
        else
            H = (Cones.use_dual(cone_k) ? Cones.hess(cone_k) : Cones.inv_hess(cone_k))
            system_solver.lhs3[idx:(idx + dim - 1), idx:(idx + dim - 1)] .= -H
            idx += dim
        end
    end


    @timeit solver.timer "update_fact" update_fact(system_solver.fact_cache, system_solver.lhs3)

    # @show "try" Matrix(system_solver.lhs3)


    return system_solver
end

function solve_subsystem(system_solver::SymIndefSparseExpandedSystemSolver, sol3::Matrix, rhs3::Matrix)
    solve_system(system_solver.fact_cache, sol3, system_solver.lhs3, rhs3)
    return sol3
end
