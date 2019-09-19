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

TODO for iterative method
- precondition
- optimize operations
- fix IterativeSolvers so that methods can take matrix RHS
=#

mutable struct NaiveSystemSolver{T <: Real} <: SystemSolver{T}
    use_iterative::Bool
    use_sparse::Bool

    tau_row
    lhs
    lhs_copy
    lhs_H_k
    fact_cache

    function NaiveSystemSolver{T}(; use_iterative::Bool = false, use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
        system_solver.use_sparse = use_sparse
        return system_solver
    end
end

# create the system_solver cache
function load(system_solver::NaiveSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    tau_row = n + p + q + 1
    system_solver.tau_row = tau_row
    dim = n + p + 2q + 2

    if system_solver.use_iterative
        # setup block LHS
        rc1 = 1:n
        rc2 = n .+ (1:p)
        rc3 = (n + p) .+ (1:q)
        rc4 = tau_row:tau_row
        rc5 = tau_row .+ (1:q)
        rc6 = dim:dim

        k_len = 2 * length(model.cones)
        cone_rows = Vector{UnitRange{Int}}(undef, k_len)
        cone_cols = Vector{UnitRange{Int}}(undef, k_len)
        cone_blocks = Vector{Any}(undef, k_len)
        for (k, cone_k) in enumerate(model.cones)
            idxs_k = model.cone_idxs[k]
            rows = tau_row .+ idxs_k
            k1 = 2k - 1
            k2 = 2k
            cone_rows[k1] = cone_rows[k2] = rows
            cone_cols[k1] = (n + p) .+ idxs_k
            cone_cols[k2] = rows
            if Cones.use_dual(cone_k)
                cone_blocks[k1] = cone_k
                cone_blocks[k2] = I
            else
                cone_blocks[k1] = I
                cone_blocks[k2] = cone_k
            end
        end

        system_solver.lhs = BlockMatrix{T}(dim, dim,
            [cone_blocks...,
                model.A', model.G', reshape(model.c, :, 1),
                -model.A, reshape(model.b, :, 1),
                -model.G, reshape(model.h, :, 1), -I,
                -model.c', -model.b', -model.h', -ones(T, 1, 1),
                solver, ones(T, 1, 1)],
            [cone_rows...,
                rc1, rc1, rc1,
                rc2, rc2,
                rc3, rc3, rc3,
                rc4, rc4, rc4, rc4,
                rc6, rc6],
            [cone_cols...,
                rc2, rc3, rc4,
                rc1, rc4,
                rc1, rc4, rc5,
                rc1, rc2, rc3, rc6,
                rc4, rc6],
            )
    else
        if system_solver.use_sparse
            system_solver.lhs = T[
                spzeros(T,n,n)  model.A'        model.G'              model.c       spzeros(T,n,q)         spzeros(T,n);
                -model.A        spzeros(T,p,p)  spzeros(T,p,q)        model.b       spzeros(T,p,q)         spzeros(T,p);
                -model.G        spzeros(T,q,p)  spzeros(T,q,q)        model.h       sparse(-one(T)*I,q,q)  spzeros(T,q);
                -model.c'       -model.b'       -model.h'             zero(T)       spzeros(T,1,q)         -one(T);
                spzeros(T,q,n)  spzeros(T,q,p)  sparse(one(T)*I,q,q)  spzeros(T,q)  sparse(one(T)*I,q,q)   spzeros(T,q);
                spzeros(T,1,n)  spzeros(T,1,p)  spzeros(T,1,q)        one(T)        spzeros(T,1,q)         one(T);
                ]
            dropzeros!(system_solver.lhs)
            @assert issparse(system_solver.lhs)
        else
            system_solver.lhs_copy = T[
                zeros(T,n,n)  model.A'      model.G'              model.c     zeros(T,n,q)           zeros(T,n);
                -model.A      zeros(T,p,p)  zeros(T,p,q)          model.b     zeros(T,p,q)           zeros(T,p);
                -model.G      zeros(T,q,p)  zeros(T,q,q)          model.h     Matrix(-one(T)*I,q,q)  zeros(T,q);
                -model.c'     -model.b'     -model.h'             zero(T)     zeros(T,1,q)           -one(T);
                zeros(T,q,n)  zeros(T,q,p)  Matrix(one(T)*I,q,q)  zeros(T,q)  Matrix(one(T)*I,q,q)   zeros(T,q);
                zeros(T,1,n)  zeros(T,1,p)  zeros(T,1,q)          one(T)      zeros(T,1,q)           one(T);
                ]
            system_solver.lhs = similar(system_solver.lhs_copy)
            # system_solver.fact_cache = HypLUSolveCache(system_solver.sol, system_solver.lhs, rhs)
        end

        function view_H_k(cone_k, idxs_k)
            rows = tau_row .+ idxs_k
            cols = Cones.use_dual(cone_k) ? (n + p) .+ idxs_k : rows
            return view(system_solver.lhs, rows, cols)
        end
        system_solver.lhs_H_k = [view_H_k(cone_k, idxs_k) for (cone_k, idxs_k) in zip(cones, cone_idxs)]
    end

    return system_solver
end

# update the LHS factorization to prepare for solve
function update_fact(system_solver::NaiveSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.use_iterative && return system_solver

    if !system_solver.use_sparse
        copyto!(system_solver.lhs, system_solver.lhs_copy)
    end

    system_solver.lhs[end, system_solver.tau_row] = solver.mu / solver.tau / solver.tau
    for (k, cone_k) in enumerate(solver.model.cones)
        copyto!(system_solver.lhs_H_k[k], Cones.hess(cone_k))
    end

    if system_solver.use_sparse
        system_solver.fact_cache = lu(system_solver.lhs)
    else
        system_solver.fact_cache = lu!(system_solver.lhs) # TODO use wrapped lapack function
    end
    
    return system_solver
end

# solve system without outer iterative refinement
function solve_system(system_solver::NaiveSystemSolver{T}, solver::Solver{T}, sol, rhs) where {T <: Real}
    if system_solver.use_iterative
        # TODO tune number of restarts and tolerances etc, ensure initial point in sol helps
        for j in 1:size(rhs, 2)
            rhs_j = view(rhs, :, j)
            sol_j = view(sol, :, j)
            IterativeSolvers.gmres!(sol_j, system_solver.lhs, rhs_j, restart = size(rhs, 1))
        end
    else
        if system_solver.use_sparse
            sol .= system_solver.fact_cache \ rhs
        else
            # if !hyp_lu_solve!(system_solver.fact_cache, sol, lhs, rhs)
            #     @warn("numerical failure: could not fix linear solve failure")
            # end
            ldiv!(sol, system_solver.fact_cache, rhs) # TODO use wrapped lapack function
        end
    end

    return sol
end
