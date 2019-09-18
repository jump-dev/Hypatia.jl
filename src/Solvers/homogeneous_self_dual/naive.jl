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
- try linear maps:
    # lhs_lin_map(arr) = @views vcat(apply_lhs(solver, arr[1:n], arr[(n + 1):(n + p)], arr[(n + p + 1):(n + p + q)], arr[tau_row], arr[tau_row .+ (1:q)], arr[end]))
    # system_solver.lhs_map = LinearMaps.FunctionMap{T}(lhs_lin_map)

TODO remove the lhs_copy? only need if factorization overwrites lhs
=#

mutable struct NaiveSystemSolver{T <: Real} <: SystemSolver{T}
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
    lhs_H_k

    dense_cache

    function NaiveSystemSolver{T}(; use_iterative::Bool = false, use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
        system_solver.use_sparse = use_sparse
        return system_solver
    end
end

# create the system_solver cache
function load(system_solver::NaiveSystemSolver{T}, solver::Solver{T}) where {T <: Real}
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
    system_solver.sol_s1 = view(sol, rows, 1)
    system_solver.sol_s2 = view(sol, rows, 2)
    system_solver.rhs_s1_k = [view(rhs, tau_row .+ idxs_k, 1) for idxs_k in cone_idxs]
    system_solver.rhs_s2_k = [view(rhs, tau_row .+ idxs_k, 2) for idxs_k in cone_idxs]

    if system_solver.use_iterative
        system_solver.lhs = setup_block_lhs(solver)
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
        end

        function view_H_k(cone_k, idxs_k)
            rows = tau_row .+ idxs_k
            cols = Cones.use_dual(cone_k) ? (n + p) .+ idxs_k : rows
            return view(system_solver.lhs, rows, cols)
        end
        system_solver.lhs_H_k = [view_H_k(cone_k, idxs_k) for (cone_k, idxs_k) in zip(cones, cone_idxs)]

        if !system_solver.use_sparse
            system_solver.dense_cache = HypLUSolveCache(system_solver.sol, system_solver.lhs, rhs)
        end
    end

    return system_solver
end

# update the system solver cache to prepare for solve
function update(system_solver::NaiveSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    tau_row = system_solver.tau_row

    # update rhs matrix
    system_solver.rhs_x1 .= solver.x_residual
    system_solver.rhs_x2 .= zero(T)
    system_solver.rhs_y1 .= solver.y_residual
    system_solver.rhs_y2 .= zero(T)
    system_solver.rhs_z1 .= solver.z_residual
    system_solver.rhs_z2 .= zero(T)
    rhs[tau_row, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    rhs[tau_row, 2] = zero(T)
    sqrtmu = sqrt(solver.mu)
    for (k, cone_k) in enumerate(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. system_solver.rhs_s1_k[k] = -duals_k
        @. system_solver.rhs_s2_k[k] = -duals_k - grad_k * sqrtmu
    end
    rhs[end, 1] = -solver.kap
    rhs[end, 2] = -solver.kap + solver.mu / solver.tau

    if !system_solver.use_iterative
        # update lhs matrix
        copyto!(lhs, system_solver.lhs_copy)
        lhs[end, tau_row] = solver.mu / solver.tau / solver.tau
        for (k, cone_k) in enumerate(solver.model.cones)
            copyto!(system_solver.lhs_H_k[k], Cones.hess(cone_k))
        end
    end

    return system_solver
end

# solve without outer iterative refinement
function solve(system_solver::NaiveSystemSolver{T}, sol_curr, rhs_curr) where {T <: Real}
    solver = system_solver.solver
    lhs = system_solver.lhs

    if system_solver.use_iterative
        rhs1 = view(rhs_curr, :, 1)
        rhs2 = view(rhs_curr, :, 2)
        sol1 = view(sol_curr, :, 1)
        sol2 = view(sol_curr, :, 2)
        IterativeSolvers.gmres!(sol1, lhs, rhs1, restart = size(lhs, 2))
        IterativeSolvers.gmres!(sol2, lhs, rhs2, restart = size(lhs, 2))
    else
        if system_solver.use_sparse
            sol_curr .= lu(lhs) \ rhs_curr
        else
            if !hyp_lu_solve!(system_solver.dense_cache, sol_curr, lhs, rhs_curr)
                @warn("numerical failure: could not fix linear solve failure (mu is $(solver.mu))")
            end
        end
    end

    return sol_curr
end

# return directions
# TODO make this function the same for all system solvers, move to solver.jl
function get_combined_directions(system_solver::NaiveSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    sol = system_solver.sol

    update(system_solver)

    refine = true # TODO handle
    if !refine
        solve(system_solver, sol, rhs) # NOTE dense solve destroys RHS
    else
        solve(system_solver, sol, copy(rhs)) # TODO remove need for copy?

        # test residual
        res = apply_lhs(solver, sol) - rhs
        sol_curr = zeros(T, size(res, 1), 2)
        res_sol = solve(system_solver, sol_curr, copy(res)) # TODO remove need for copy
        sol_new = sol - res_sol
        res_new = apply_lhs(solver, sol_new) - rhs

        norm_inf = norm(res, Inf)
        norm_inf_new = norm(res_new, Inf)
        norm_2 = norm(res, 2)
        norm_2_new = norm(res_new, 2)
        if norm_inf_new < norm_inf && norm_2_new < norm_2
            println("used iter ref")
            println(norm_inf, "\t", norm_2)
            println(norm_inf_new, "\t", norm_2_new)
            copyto!(sol, sol_new)
        end
    end

    return (system_solver.sol_x1, system_solver.sol_x2, system_solver.sol_y1, system_solver.sol_y2, system_solver.sol_z1, system_solver.sol_z2, sol[system_solver.tau_row, 1], sol[system_solver.tau_row, 2], system_solver.sol_s1, system_solver.sol_s2, sol[end, 1], sol[end, 2])
end

# TODO make efficient
function apply_lhs(solver, sol_in)
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = n + p + q + 1
    return vcat(apply_lhs(solver, sol_in[1:n, :], sol_in[(n + 1):(n + p), :], sol_in[(n + p + 1):(n + p + q), :], sol_in[tau_row:tau_row, :], sol_in[tau_row .+ (1:q), :], sol_in[end:end, :])...)
end

# for iterative methods, build block matrix for efficient multiplication
function setup_block_lhs(solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = n + p + q + 1
    rc1 = 1:n
    rc2 = n .+ (1:p)
    rc3 = (n + p) .+ (1:q)
    rc4 = tau_row:tau_row
    rc5 = tau_row .+ (1:q)
    dim = tau_row + q + 1
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

    block_lhs = BlockMatrix{T}(dim, dim,
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

    return block_lhs
end

# TODO experimental for block LHS: if block is a Cone then define mul as hessian product, if block is solver then define mul by mu/tau/tau
# TODO optimize... maybe need for each cone a 5-arg hess prod
import LinearAlgebra.mul!

function mul!(y::AbstractVecOrMat{T}, A::Cones.Cone{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
    # TODO in-place
    ytemp = y * beta
    Cones.hess_prod!(y, x, A)
    rmul!(y, alpha)
    y .+= ytemp
    return y
end

function mul!(y::AbstractVecOrMat{T}, solver::Solvers.Solver{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
    rmul!(y, beta)
    @. y += alpha * x / solver.tau * solver.mu / solver.tau
    return y
end
