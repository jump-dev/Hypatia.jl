#=
Copyright 2018, Chris Coey and contributors

naive linear system solver

6x6 nonsymmetric system in (x, y, z, tau, s, kap):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x + h*tau - s = zrhs
-c'*x - b'*y - h'*z - kap = taurhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
mu/(taubar^2)*tau + kap = kaprhs

TODO not sure lhs_copy is needed
=#

import Pardiso

struct DefaultSparseSolver end
SparseSystemSolver = Union{DefaultSparseSolver, Pardiso.PardisoSolver}

function analyze_sparse_system(ps::Pardiso.PardisoSolver, lhs::SparseMatrixCSC, rhs::Matrix)
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    Pardiso.set_iparm!(ps, 12, 1)
    # Pardiso.set_iparm!(ps, 6, 1)
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
    Pardiso.pardiso(ps, lhs, rhs)
    return
end
analyze_sparse_system(::DefaultSparseSolver, ::SparseMatrixCSC, ::Matrix) = nothing

function solve_sparse_system(ps::Pardiso.PardisoSolver, lhs::SparseMatrixCSC, rhs::Matrix)
    # if Pardiso.get_phase(ps) == Pardiso.ANALYSIS_NUM_FACT_SOLVE_REFINE
        analyze_sparse_system(ps, lhs, rhs)
    # end
    sol = copy(rhs)
    Pardiso.set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
    Pardiso.pardiso(ps, sol, lhs, rhs)
    @show norm(rhs - lhs * sol)
    rhs .= sol
    return sol
end

solve_sparse_system(ps::DefaultSparseSolver, lhs::SparseMatrixCSC, rhs::Matrix) = rhs .= lu(lhs) \ rhs

function free_sparse_solver_memory(ps::Pardiso.PardisoSolver)
    Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(ps, sol, lhs, rhs)
    return
end
free_sparse_solver_memory(::DefaultSparseSolver) = nothing
free_sparse_solver_memory(s::SystemSolver) = free_sparse_solver_memory(s.sparse_solver)

mutable struct NaiveSparseSystemSolver{T <: Real} <: SystemSolver{T}
    use_iterative::Bool
    use_sparse::Bool

    solver::Solver{T}

    x1
    x2
    y1
    y2
    z1
    z2
    tau_row::Int # remove
    tau_idx::Int
    s1
    s2
    s1_k
    s2_k

    lhs_copy
    lhs
    # for debugging
    lhs_actual_copy # remove
    lhs_actual # remove

    lhs_H_k # remove
    lhs_H_Vs
    Is::Vector{Int}
    Js::Vector{Int}
    Vs::Vector{T}
    sparse_solver::SparseSystemSolver

    rhs::Matrix{T}
    prevsol1::Vector{T}
    prevsol2::Vector{T}

    solvesol
    solvecache

    function NaiveSparseSystemSolver{T}(;
            use_iterative::Bool = false,
            use_sparse::Bool = false,
            sparse_solver = Pardiso.PardisoSolver(),
            ) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
        system_solver.use_sparse = use_sparse
        system_solver.sparse_solver = sparse_solver
        return system_solver
    end
end

function load(system_solver::NaiveSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.solver = solver

    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    dim = n + p + 2q + 2

    rhs = zeros(T, dim, 2)
    system_solver.rhs = rhs
    rows = 1:n
    system_solver.x1 = view(rhs, rows, 1)
    system_solver.x2 = view(rhs, rows, 2)
    rows = (n + 1):(n + p)
    system_solver.y1 = view(rhs, rows, 1)
    system_solver.y2 = view(rhs, rows, 2)
    rows = (n + p + 1):(n + p + q)
    system_solver.z1 = view(rhs, rows, 1)
    system_solver.z2 = view(rhs, rows, 2)
    tau_row = n + p + q + 1 # remove
    system_solver.tau_row = tau_row # remove
    rows = tau_row .+ (1:q)
    system_solver.s1 = view(rhs, rows, 1)
    system_solver.s2 = view(rhs, rows, 2)
    system_solver.s1_k = [view(rhs, tau_row .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
    system_solver.s2_k = [view(rhs, tau_row .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]

    # x y z kap s tau

    system_solver.lhs_actual_copy = T[
        spzeros(T,n,n)  model.A'        model.G'              model.c       spzeros(T,n,q)         spzeros(T,n);
        -model.A        spzeros(T,p,p)  spzeros(T,p,q)        model.b       spzeros(T,p,q)         spzeros(T,p);
        -model.G        spzeros(T,q,p)  spzeros(T,q,q)        model.h       sparse(-one(T)*I,q,q)  spzeros(T,q);
        -model.c'       -model.b'       -model.h'             zero(T)       spzeros(T,1,q)         -one(T);
        spzeros(T,q,n)  spzeros(T,q,p)  sparse(one(T)*I,q,q)  spzeros(T,q)  sparse(one(T)*I,q,q)   spzeros(T,q);
        spzeros(T,1,n)  spzeros(T,1,p)  spzeros(T,1,q)        one(T)        spzeros(T,1,q)         one(T);
        ]
    dropzeros!(system_solver.lhs_actual_copy)
    system_solver.lhs_actual = similar(system_solver.lhs_actual_copy)

    hess_nnzs = sum(Cones.dimension(cone_k) + Cones.dimension(cone_k) ^ 2 for cone_k in model.cones)

    total_nnz = 2 * (nnz(sparse(model.A)) + nnz(sparse(model.G)) + n + p + q + 1) + q + 1 + hess_nnzs
    Is = system_solver.Is = Vector{Int}(undef, total_nnz)
    Js = system_solver.Js = Vector{Int}(undef, total_nnz)
    Vs = system_solver.Vs = Vector{T}(undef, total_nnz)

    function add_I_J_V(k, start_row, start_col, vec::Vector{T})
        if !isempty(vec)
            for i in eachindex(vec)
                Is[k] = i + start_row
                Js[k] = start_col + 1
                Vs[k] = vec[i]
                k += 1
            end
        end
        return k
    end

    # function add_I_J_V(k, start_row, start_col, vec::Vector{T})
    #     n = length(vec)
    #     if !isempty(vec)
    #         Is[k:(k + n - 1)] .= (start_row + 1):(start_row + n)
    #         Js[k:(k + n - 1)] .= start_col + 1
    #         Vs[k:(k + n - 1)] .= vec
    #     end
    #     return k + n
    # end

    function add_I_J_V(k, start_row, start_col, vec::Adjoint{T, Array{T, 1}})
        if !isempty(vec)
            for j in eachindex(vec)
                Is[k] = start_row + 1
                Js[k] = j + start_col
                Vs[k] = vec[j]
                k += 1
            end
        end
        return k
    end

    function add_I_J_V(k, start_row, start_col, mat::SparseMatrixCSC{T, Int64})
        if !isempty(mat)
            for (i, j, v) in zip(findnz(mat)...)
                Is[k] = i + start_row
                Js[k] = j + start_col
                Vs[k] = v
                k += 1
            end
        end
        return k
    end

    rc1 = 0
    rc2 = n
    rc3 = n + p
    rc4 = n + p + q
    rc5 = n + p + q + 1
    rc6 = dim - 1
    offset = 1
    offset = add_I_J_V(offset, rc1, rc2, sparse(model.A'))
    offset = add_I_J_V(offset, rc1, rc3, sparse(model.G'))
    offset = add_I_J_V(offset, rc1, rc4, model.c)
    offset = add_I_J_V(offset, rc2, rc1, -sparse(model.A))
    offset = add_I_J_V(offset, rc2, rc4, model.b)
    offset = add_I_J_V(offset, rc3, rc1, -sparse(model.G))
    offset = add_I_J_V(offset, rc3, rc4, model.h)
    offset = add_I_J_V(offset, rc3, rc5, sparse(-one(T) * I, q, q))
    offset = add_I_J_V(offset, rc4, rc1, -model.c')
    offset = add_I_J_V(offset, rc4, rc2, -model.b')
    offset = add_I_J_V(offset, rc4, rc3, -model.h')
    offset = add_I_J_V(offset, rc4, rc6, -[one(T)])
    system_solver.tau_idx = offset
    offset = add_I_J_V(offset, rc6, rc4, [one(T)])
    offset = add_I_J_V(offset, rc6, rc6, [one(T)])

    # add I, J, V for Hessians and cache indices to modify later
    H_indices = [Vector{Int}(undef, Cones.dimension(cone_k)) for cone_k in model.cones]
    nz_rows_added = 0
    for (k, cone_k) in enumerate(model.cones)
        cone_dim = Cones.dimension(cone_k)
        rows = rc5 + nz_rows_added
        # indices of I, J, V affected
        H_indices[k] = offset:(offset + cone_dim ^ 2 - 1)
        if Cones.use_dual(cone_k)
            H_cols = rc3 + nz_rows_added
            id_cols = rows
        else
            id_cols = rc3 + nz_rows_added
            H_cols = rows
        end
        offset = add_I_J_V(offset, rows, H_cols, sparse(ones(T, cone_dim, cone_dim)))
        offset = add_I_J_V(offset, rows, id_cols, sparse(one(T) * I, cone_dim, cone_dim))
        nz_rows_added += cone_dim
    end

    system_solver.lhs_copy = sparse(Is, Js, Vs)
    system_solver.lhs = similar(system_solver.lhs_copy)
    system_solver.lhs_actual = similar(system_solver.lhs_actual_copy)

    function view_k(k::Int)
        rows = tau_row .+ model.cone_idxs[k]
        if Cones.use_dual(model.cones[k])
            cols = (n + p) .+ model.cone_idxs[k]
        else
            cols = rows
        end
        return view(system_solver.lhs_actual, rows, cols)
    end
    system_solver.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]

    view_k2(k::Int) = view(system_solver.Vs, H_indices[k])
    system_solver.lhs_H_Vs = [view_k2(k) for k in eachindex(model.cones)]

    return system_solver
end

function get_combined_directions(system_solver::NaiveSparseSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    cones = model.cones
    lhs = system_solver.lhs
    lhs_actual = system_solver.lhs_actual

    rhs = system_solver.rhs
    tau_row = system_solver.tau_row
    x1 = system_solver.x1
    x2 = system_solver.x2
    y1 = system_solver.y1
    y2 = system_solver.y2
    z1 = system_solver.z1
    z2 = system_solver.z2
    s1 = system_solver.s1
    s2 = system_solver.s2
    s1_k = system_solver.s1_k
    s2_k = system_solver.s2_k

    sqrtmu = sqrt(solver.mu)
    mtt = solver.mu / solver.tau / solver.tau

    # update rhs matrix
    x1 .= solver.x_residual
    x2 .= zero(T)
    y1 .= solver.y_residual
    y2 .= zero(T)
    z1 .= solver.z_residual
    z2 .= zero(T)
    rhs[tau_row, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    rhs[tau_row, 2] = zero(T)
    for k in eachindex(cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cones[k])
        @. s1_k[k] = -duals_k
        @. s2_k[k] = -duals_k - grad_k * sqrtmu
    end
    rhs[end, 1] = -solver.kap
    rhs[end, 2] = -solver.kap + solver.mu / solver.tau

    # solve system
    # update lhs matrix
    copyto!(lhs_actual, system_solver.lhs_actual_copy)
    lhs_actual[end, tau_row] = mtt
    for k in eachindex(cones)
        copyto!(system_solver.lhs_H_k[k], Cones.hess(cones[k]))
    end

    for k in eachindex(cones)
        copyto!(system_solver.lhs_H_Vs[k], vec(Cones.hess(cones[k])))
    end
    system_solver.Vs[system_solver.tau_idx] = mtt
    lhs = sparse(system_solver.Is, system_solver.Js, system_solver.Vs)

    # ps = PardisoSolver()
    # rhs .= Pardiso.solve(ps, lhs, rhs)

    # rhs .= lu(lhs_actual) \ rhs
    sol = copy(rhs)
    solve_sparse_system(system_solver.sparse_solver, lhs, rhs)

    return (x1, x2, y1, y2, z1, z2, rhs[tau_row, 1], rhs[tau_row, 2], s1, s2, rhs[end, 1], rhs[end, 2])
end
