#=
symmetric-indefinite linear system solver
solves linear system in naive.jl by first eliminating s and kap via the method in naiveelim.jl and then eliminating tau via a procedure similar to that described by S7.4 of
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

TODO not implemented: to avoid inverse hessian products, let for pr bar w_k = (mu*H_k)\z_k (later recover z_k = mu*H_k*w_k) to get 3x3 symmetric indefinite system
A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
=#

abstract type SymIndefSystemSolver{T <: Real} <: SystemSolver{T} end

function setup_rhs3(
    ::SymIndefSystemSolver,
    model,
    rhs,
    sol,
    rhs_sub,
    )
    @inbounds for k in eachindex(model.cones)
        cone_k = model.cones[k]
        rhs_z_k = rhs.z_views[k]
        rhs_s_k = rhs.s_views[k]
        z_rows_k = (model.n + model.p) .+ model.cone_idxs[k]
        @views z3_k = rhs_sub[z_rows_k]
        if Cones.use_dual_barrier(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            @. z3_k = -rhs_z_k - rhs_s_k
        else
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Cones.inv_hess_prod!(z3_k, rhs_s_k, cone_k)
            axpby!(-1, rhs_z_k, -1, z3_k)
        end
    end
    return nothing
end

#=
direct sparse
=#

mutable struct SymIndefSparseSystemSolver{T <: Real} <: SymIndefSystemSolver{T}
    lhs_sub::SparseMatrixCSC # TODO type will depend on Int type
    rhs_sub::Vector{T}
    sol_sub::Vector{T}
    hess_idxs::Vector
    fact_cache::SparseSymCache{T}
    const_sol::Vector{T}
    const_rhs::Vector{T}
    function SymIndefSparseSystemSolver{T}(;
        fact_cache::SparseSymCache{T} = SparseSymCache{T}(),
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::SymIndefSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    dim = n + p + q

    system_solver.sol_sub = zeros(dim)
    system_solver.rhs_sub = similar(system_solver.sol_sub)

    # form sparse LHS without Hessians and inverse Hessians in z/z block
    lhs_sub = T[
        spzeros(T, n, n)  spzeros(T, n, p)  spzeros(T, n, q);
        model.A           spzeros(T, p, p)  spzeros(T, p, q);
        model.G           spzeros(T, q, p)  sparse(-one(T) * I, q, q);
        ]
    @assert issparse(lhs_sub)
    dropzeros!(lhs_sub)
    (Is, Js, Vs) = findnz(lhs_sub)

    # add I, J, V for Hessians and inverse Hessians
    if isempty(cones)
        hess_nz_total = 0
    else
        hess_nz_total = sum(Cones.use_dual_barrier(cone_k) ? Cones.hess_nz_count_tril(cone_k) : Cones.inv_hess_nz_count_tril(cone_k) for cone_k in cones)
    end
    H_Is = Vector{Int}(undef, hess_nz_total)
    H_Js = Vector{Int}(undef, hess_nz_total)
    offset = 1
    y_start = n + p - 1
    for (cone_k, idxs_k) in zip(cones, cone_idxs)
        z_start_k = y_start + first(idxs_k)
        for j in 1:Cones.dimension(cone_k)
            nz_rows_kj = z_start_k .+ (Cones.use_dual_barrier(cone_k) ? Cones.hess_nz_idxs_col_tril(cone_k, j) : Cones.inv_hess_nz_idxs_col_tril(cone_k, j))
            len_kj = length(nz_rows_kj)
            IJV_idxs = offset:(offset + len_kj - 1)
            offset += len_kj
            @. H_Is[IJV_idxs] = nz_rows_kj
            @. H_Js[IJV_idxs] = z_start_k + j
        end
    end
    @assert offset == hess_nz_total + 1
    append!(Is, H_Is)
    append!(Js, H_Js)
    append!(Vs, ones(T, hess_nz_total))

    pert = T(system_solver.fact_cache.diag_pert) # TODO change where this is stored
    append!(Is, 1:(n + p))
    append!(Js, 1:(n + p))
    append!(Vs, fill(pert, n))
    append!(Vs, fill(-pert, p))

    # integer type supported by the sparse system solver library to be used
    Ti = int_type(system_solver.fact_cache)
    # prefer conversions of integer types to happen here than inside external wrappers
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    lhs_sub = system_solver.lhs_sub = sparse(Is, Js, Vs, dim, dim)

    # cache indices of nonzeros of Hessians and inverse Hessians in sparse LHS nonzeros vector
    system_solver.hess_idxs = [Vector{Union{UnitRange, Vector{Int}}}(undef, Cones.dimension(cone_k)) for cone_k in cones]
    for (k, cone_k) in enumerate(cones)
        cone_idxs_k = cone_idxs[k]
        z_start_k = y_start + first(cone_idxs_k)
        for j in 1:Cones.dimension(cone_k)
            col = z_start_k + j
            # get nonzero rows in the current column of the LHS
            col_idx_start = lhs_sub.colptr[col]
            nz_rows = lhs_sub.rowval[col_idx_start:(lhs_sub.colptr[col + 1] - 1)]
            # get nonzero rows in column j of the Hessian or inverse Hessian
            nz_hess_indices = (Cones.use_dual_barrier(cone_k) ? Cones.hess_nz_idxs_col_tril(cone_k, j) : Cones.inv_hess_nz_idxs_col_tril(cone_k, j))
            # get index corresponding to first nonzero Hessian element of the current column of the LHS
            first_H = findfirst(isequal(z_start_k + first(nz_hess_indices)), nz_rows)
            # indices of nonzero values for cone k column j
            system_solver.hess_idxs[k][j] = (col_idx_start + first_H - 2) .+ (1:length(nz_hess_indices))
        end
    end

    system_solver.const_rhs = vcat(-model.c, model.b, model.h)
    system_solver.const_sol = similar(system_solver.const_rhs)

    return system_solver
end

function update_lhs(system_solver::SymIndefSparseSystemSolver, solver::Solver)
    for (k, cone_k) in enumerate(solver.model.cones)
        H_k = (Cones.use_dual_barrier(cone_k) ? Cones.hess(cone_k) : Cones.inv_hess(cone_k))
        for j in 1:Cones.dimension(cone_k)
            nz_rows = (Cones.use_dual_barrier(cone_k) ? Cones.hess_nz_idxs_col_tril(cone_k, j) : Cones.inv_hess_nz_idxs_col_tril(cone_k, j))
            @. @views system_solver.lhs_sub.nzval[system_solver.hess_idxs[k][j]] = -H_k[nz_rows, j]
        end
    end

    @timeit solver.timer "update_fact" update_fact(system_solver.fact_cache, system_solver.lhs_sub)
    @timeit solver.timer "solve_subsystem3" solve_subsystem3(system_solver, solver, system_solver.const_sol, system_solver.const_rhs)

    return system_solver
end

function solve_subsystem3(
    system_solver::SymIndefSparseSystemSolver,
    ::Solver,
    sol_sub::Vector,
    rhs_sub::Vector,
    )
    inv_prod(system_solver.fact_cache, sol_sub, system_solver.lhs_sub, rhs_sub)
    return sol_sub
end

#=
direct dense
=#

mutable struct SymIndefDenseSystemSolver{T <: Real} <: SymIndefSystemSolver{T}
    lhs_sub::Symmetric{T, Matrix{T}}
    rhs_sub::Vector{T}
    sol_sub::Vector{T}
    fact_cache::DenseSymCache{T}
    const_sol::Vector{T}
    const_rhs::Vector{T}
    function SymIndefDenseSystemSolver{T}(;
        fact_cache::DenseSymCache{T} = DenseSymCache{T}(),
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::SymIndefDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    system_solver.sol_sub = zeros(T, n + p + q)
    system_solver.rhs_sub = similar(system_solver.sol_sub)

    # fill symmetric lower triangle
    system_solver.lhs_sub = Symmetric(T[
        zeros(T, n, n)  zeros(T, n, p)  zeros(T, n, q);
        model.A         zeros(T, p, p)  zeros(T, p, q);
        model.G         zeros(T, q, p)  Matrix(-one(T) * I, q, q);
        ], :L)

    load_matrix(system_solver.fact_cache, system_solver.lhs_sub)

    system_solver.const_rhs = vcat(-model.c, model.b, model.h)
    system_solver.const_sol = similar(system_solver.const_rhs)

    return system_solver
end

function update_lhs(system_solver::SymIndefDenseSystemSolver, solver::Solver)
    model = solver.model
    (n, p) = (model.n, model.p)
    lhs_sub = system_solver.lhs_sub.data

    for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
        z_rows_k = (n + p) .+ idxs_k
        if Cones.use_dual_barrier(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            H_k = Cones.hess(cone_k)
            @. lhs_sub[z_rows_k, z_rows_k] = -H_k
        else
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Hi_k = Cones.inv_hess(cone_k)
            @. lhs_sub[z_rows_k, z_rows_k] = -Hi_k
        end
    end

    @timeit solver.timer "update_fact" update_fact(system_solver.fact_cache, system_solver.lhs_sub)
    @timeit solver.timer "solve_subsystem3" solve_subsystem3(system_solver, solver, system_solver.const_sol, system_solver.const_rhs)

    return system_solver
end

function solve_subsystem3(
    system_solver::SymIndefDenseSystemSolver,
    ::Solver,
    sol_sub::Vector,
    rhs_sub::Vector,
    )
    copyto!(sol_sub, rhs_sub)
    inv_prod(system_solver.fact_cache, sol_sub)
    return sol_sub
end
