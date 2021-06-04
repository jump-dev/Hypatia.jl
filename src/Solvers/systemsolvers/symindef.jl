#=
symmetric-indefinite linear system solver
solves linear system in naive.jl by first eliminating s and kap via the method
in naiveelim.jl and then eliminating tau via a procedure similar to that
described by S7.4 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

3x3 nonsymmetric system in (x, y, z):
A'*y + G'*z = [xrhs, -c]
-A*x = [yrhs, -b]
(pr bar) -mu*H_k*G_k*x + z_k = [mu*H_k*zrhs_k + srhs_k, -mu*H_k*h_k]
(du bar) -G_k*x + mu*H_k*z_k = [zrhs_k + srhs_k, -h_k]

multiply pr bar constraint by (mu*H_k)^-1 to get 3x3 symmetric indefinite system
A'*y + G'*z = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]

TODO not implemented: to avoid inverse hessian products, let for pr bar
w_k = (mu*H_k)\z_k (later recover z_k = mu*H_k*w_k)
to get 3x3 symmetric indefinite system
A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
=#

abstract type SymIndefSystemSolver{T <: Real} <: SystemSolver{T} end

function setup_rhs3(
    ::SymIndefSystemSolver{T},
    model::Models.Model{T},
    rhs::Point{T},
    sol::Point{T},
    rhs_sub::Point{T},
    ) where {T <: Real}
    @inbounds for (k, cone_k) in enumerate(model.cones)
        rhs_z_k = rhs.z_views[k]
        rhs_s_k = rhs.s_views[k]
        rhs_sub_z_k = rhs_sub.z_views[k]
        if Cones.use_dual_barrier(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            @. rhs_sub_z_k = -rhs_z_k - rhs_s_k
        else
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Cones.inv_hess_prod!(rhs_sub_z_k, rhs_s_k, cone_k)
            axpby!(-1, rhs_z_k, -1, rhs_sub_z_k)
        end
    end
    return
end

#=
direct sparse
=#

mutable struct SymIndefSparseSystemSolver{T <: Real} <: SymIndefSystemSolver{T}
    lhs_sub::SparseMatrixCSC{T}
    fact_cache::SparseSymCache{T}
    hess_idxs::Vector
    rhs_sub::Point{T}
    sol_sub::Point{T}
    sol_const::Point{T}
    rhs_const::Point{T}
    function SymIndefSparseSystemSolver{T}(;
        fact_cache::SparseSymCache{T} = SparseSymCache{T}(),
        ) where {T <: Real}
        syssolver = new{T}()
        syssolver.fact_cache = fact_cache
        return syssolver
    end
end

function load(
    syssolver::SymIndefSparseSystemSolver{T},
    solver::Solver{T},
    ) where {T <: Real}
    syssolver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs

    # form lower triangle of symmetric sparse LHS
    # without Hessians and inverse Hessians in z/z block
    # TODO check for inefficiency, may need to implement more manually
    spz(a, b) = spzeros(T, a, b)
    lhs_sub = hvcat((1, 2, 2),
        spz(n, n + p + q),
        model.A, spz(p, p + q),
        model.G, spz(q, p + q))
    @assert lhs_sub isa SparseMatrixCSC{T}
    dropzeros!(lhs_sub)
    (Is, Js, Vs) = findnz(lhs_sub)

    # add I, J, V for Hessians and inverse Hessians
    if isempty(cones)
        hess_nz_total = 0
    else
        hess_nz_total = sum(Cones.use_dual_barrier(cone_k) ?
            Cones.hess_nz_count_tril(cone_k) :
            Cones.inv_hess_nz_count_tril(cone_k) for cone_k in cones)
    end
    H_Is = Vector{Int}(undef, hess_nz_total)
    H_Js = Vector{Int}(undef, hess_nz_total)
    offset = 1
    y_start = n + p - 1
    for (cone_k, idxs_k) in zip(cones, cone_idxs)
        z_start_k = y_start + first(idxs_k)
        for j in 1:Cones.dimension(cone_k)
            nz_rows_kj = z_start_k .+ (Cones.use_dual_barrier(cone_k) ?
                Cones.hess_nz_idxs_col_tril(cone_k, j) :
                Cones.inv_hess_nz_idxs_col_tril(cone_k, j))
            len_kj = length(nz_rows_kj)
            IJV_idxs = offset:(offset + len_kj - 1)
            offset += len_kj
            @. H_Is[IJV_idxs] = nz_rows_kj
            @. H_Js[IJV_idxs] = z_start_k + j
        end
    end
    append!(Is, H_Is)
    append!(Js, H_Js)
    append!(Vs, ones(T, hess_nz_total))

    diag_pert = T(diag_min(syssolver.fact_cache))
    append!(Is, 1:(n + p))
    append!(Js, 1:(n + p))
    append!(Vs, fill(diag_pert, n))
    append!(Vs, fill(-diag_pert, p))

    # integer type supported by the sparse system solver library to be used
    Ti = int_type(syssolver.fact_cache)
    # convert integer types here rather than inside external wrappers
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    dim = size(lhs_sub, 1)
    lhs_sub = syssolver.lhs_sub = sparse(Is, Js, Vs, dim, dim)

    # cache indices of nonzeros of Hessians in sparse LHS nonzeros vector
    syssolver.hess_idxs = [Vector{Union{UnitRange, Vector{Int}}}(
        undef, Cones.dimension(cone_k)) for cone_k in cones]
    for (k, cone_k) in enumerate(cones)
        cone_idxs_k = cone_idxs[k]
        z_start_k = y_start + first(cone_idxs_k)
        for j in 1:Cones.dimension(cone_k)
            col = z_start_k + j
            # get nonzero rows in the current column of the LHS
            col_idx_start = lhs_sub.colptr[col]
            nz_rows = lhs_sub.rowval[col_idx_start:(lhs_sub.colptr[col + 1] - 1)]
            # get nonzero rows in column j of the Hessian or inverse Hessian
            nz_hess_indices = (Cones.use_dual_barrier(cone_k) ?
                Cones.hess_nz_idxs_col_tril(cone_k, j) :
                Cones.inv_hess_nz_idxs_col_tril(cone_k, j))
            # get index corresponding to first nonzero element of the current col
            first_H = findfirst(isequal(z_start_k + first(nz_hess_indices)),
                nz_rows)
            # indices of nonzero values for cone k column j
            syssolver.hess_idxs[k][j] = (col_idx_start + first_H - 2) .+
                (1:length(nz_hess_indices))
        end
    end

    setup_point_sub(syssolver, model)

    return syssolver
end

function update_lhs(syssolver::SymIndefSparseSystemSolver, solver::Solver)
    for (k, cone_k) in enumerate(solver.model.cones)
        H_k = (Cones.use_dual_barrier(cone_k) ? Cones.hess(cone_k) :
            Cones.inv_hess(cone_k))
        for j in 1:Cones.dimension(cone_k)
            nz_rows = (Cones.use_dual_barrier(cone_k) ?
                Cones.hess_nz_idxs_col_tril(cone_k, j) :
                Cones.inv_hess_nz_idxs_col_tril(cone_k, j))
            @. @views syssolver.lhs_sub.nzval[syssolver.hess_idxs[k][j]] =
                -H_k[nz_rows, j]
        end
    end

    solver.time_upfact += @elapsed update_fact(syssolver.fact_cache,
        syssolver.lhs_sub)
    solve_subsystem3(syssolver, solver, syssolver.sol_const, syssolver.rhs_const)

    return syssolver
end

function solve_subsystem3(
    syssolver::SymIndefSparseSystemSolver,
    ::Solver,
    sol::Point,
    rhs::Point,
    )
    inv_prod(syssolver.fact_cache, sol.vec, syssolver.lhs_sub, rhs.vec)
    return sol
end

#=
direct dense
=#

mutable struct SymIndefDenseSystemSolver{T <: Real} <: SymIndefSystemSolver{T}
    lhs_sub::Symmetric{T, Matrix{T}}
    lhs_sub_fact::Symmetric{T, Matrix{T}}
    fact::Factorization{T}
    rhs_sub::Point{T}
    sol_sub::Point{T}
    sol_const::Point{T}
    rhs_const::Point{T}
    function SymIndefDenseSystemSolver{T}(;
        ) where {T <: Real}
        syssolver = new{T}()
        return syssolver
    end
end

function load(
    syssolver::SymIndefDenseSystemSolver{T},
    solver::Solver{T},
    ) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    npq = n + p + q

    # fill symmetric lower triangle
    lhs_sub = zeros(T, npq, npq)
    @views copyto!(lhs_sub[n .+ (1:p), 1:n], model.A)
    @views copyto!(lhs_sub[n + p .+ (1:q), 1:n], model.G)
    @assert lhs_sub isa Matrix{T}
    syssolver.lhs_sub = Symmetric(lhs_sub, :L)
    syssolver.lhs_sub_fact = zero(syssolver.lhs_sub)

    setup_point_sub(syssolver, model)

    return syssolver
end

function update_lhs(syssolver::SymIndefDenseSystemSolver, solver::Solver)
    model = solver.model
    z_start = model.n + model.p
    lhs_sub = syssolver.lhs_sub.data

    for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
        z_rows_k = z_start .+ idxs_k
        H_k = (Cones.use_dual_barrier(cone_k) ? Cones.hess :
            Cones.inv_hess)(cone_k)
        @. lhs_sub[z_rows_k, z_rows_k] = -H_k
    end

    solver.time_upfact += @elapsed syssolver.fact =
        symm_fact_copy!(syssolver.lhs_sub_fact, syssolver.lhs_sub)

    if !issuccess(syssolver.fact)
        println("symmetric linear system factorization failed")
    end

    solve_subsystem3(syssolver, solver, syssolver.sol_const, syssolver.rhs_const)

    return syssolver
end

function solve_subsystem3(
    syssolver::SymIndefDenseSystemSolver,
    ::Solver,
    sol::Point,
    rhs::Point,
    )
    ldiv!(sol.vec, syssolver.fact, rhs.vec)
    return sol
end

#=
indirect (using LinearMaps and IterativeSolvers)
TODO
- precondition
- optimize operations
- tune tolerances etc
- try to make initial point in sol a good guess (currently zeros)
=#

mutable struct SymIndefIndirectSystemSolver{T <: Real} <: SymIndefSystemSolver{T}
    lhs::LinearMaps.LinearMap{T}
    rhs_sub::Point{T}
    sol_sub::Point{T}
    sol_const::Point{T}
    rhs_const::Point{T}
    SymIndefIndirectSystemSolver{T}() where {T <: Real} = new{T}()
end

function load(
    syssolver::SymIndefIndirectSystemSolver{T},
    solver::Solver{T},
    ) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    x_idxs = 1:n
    y_idxs = n .+ (1:p)
    z_start = n + p
    z_idxs = z_start .+ (1:q)

    function symindef_mul(b::AbstractVector, a::AbstractVector)
        # x part
        @views mul!(b[x_idxs], model.A', a[y_idxs])
        @views mul!(b[x_idxs], model.G', a[z_idxs], true, true)
        # y part
        @views mul!(b[y_idxs], model.A, a[x_idxs])
        # z part
        for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
            z_rows_k = z_start .+ idxs_k
            prod_fun = (Cones.use_dual_barrier(cone_k) ? Cones.hess_prod! :
                Cones.inv_hess_prod!)
            @views prod_fun(b[z_rows_k], a[z_rows_k], cone_k)
        end
        @views mul!(b[z_idxs], model.G, a[x_idxs], true, -one(T))
        return b
    end

    syssolver.lhs = LinearMaps.LinearMap{T}(symindef_mul, z_start + q,
        ismutating = true, issymmetric = true, isposdef = false)

    setup_point_sub(syssolver, model)

    return syssolver
end

function update_lhs(syssolver::SymIndefIndirectSystemSolver, solver::Solver)
    solve_subsystem3(syssolver, solver, syssolver.sol_const, syssolver.rhs_const)
    return syssolver
end

function solve_subsystem3(
    syssolver::SymIndefIndirectSystemSolver,
    ::Solver,
    sol::Point,
    rhs::Point,
    )
    sol.vec .= 0 # initially_zero = true
    # TODO tune options, initial guess?
    IterativeSolvers.minres!(sol.vec, syssolver.lhs, rhs.vec,
        initially_zero = true) #, maxiter = 2 * size(sol.vec, 1))
    return sol
end
