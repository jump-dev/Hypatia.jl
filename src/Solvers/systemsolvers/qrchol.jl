#=
QR+Cholesky linear system solver
requires precomputed QR factorization of A'

solves linear system in naive.jl by first eliminating s, kap, and tau via the method in the symindef solver, then reducing the 3x3 symmetric indefinite system to a series of low-dimensional operations via a procedure similar to that described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf (the dominating subroutine is a positive definite linear solve with RHS of dimension n-p x 3)

TODO iterative refinement for 3x3 solution
=#

abstract type QRCholSystemSolver{T <: Real} <: SystemSolver{T} end

function setup_rhs3(
    ::QRCholSystemSolver,
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
            z_temp_k = sol.z_views[k]
            @. z_temp_k = -rhs_z_k - rhs_s_k
            Cones.inv_hess_prod!(z3_k, z_temp_k, cone_k)
        else
            Cones.hess_prod!(z3_k, rhs_z_k, cone_k)
            axpby!(-1, rhs_s_k, -1, z3_k)
        end
    end
    return nothing
end

# function solve_subsystem4(
#     system_solver::QRCholSystemSolver{T},
#     solver::Solver{T},
#     sol::Point{T},
#     rhs::Point{T},
#     tau_scal::T,
#     ) where {T <: Real}
#     model = solver.model
#     x_rows = system_solver.x_rows
#     y_rows = system_solver.y_rows
#     z_rows = system_solver.z_rows
#
#     rhs_sub = system_solver.rhs_sub
#     dim3 = length(rhs_sub)
#
#     @. @views rhs_sub[x_rows] = rhs.x
#     @. @views rhs_sub[y_rows] = -rhs.y
#
#     setup_rhs3(system_solver, model, rhs, sol, rhs_sub)
#
#     sol_sub = solve_subsystem3(system_solver, solver, rhs_sub) # NOTE modifies and returns rhs_sub
#
#     # TODO refactor all below
#     # TODO maybe use higher precision here
#     const_sol = system_solver.const_sol
#
#     # lift to get tau
#     @views tau_num = rhs.tau[1] + rhs.kap[1] + dot(model.c, sol_sub[x_rows]) + dot(model.b, sol_sub[y_rows]) + dot(model.h, sol_sub[z_rows])
#     @views tau_denom = tau_scal - dot(model.c, const_sol[x_rows]) - dot(model.b, const_sol[y_rows]) - dot(model.h, const_sol[z_rows])
#     sol_tau = tau_num / tau_denom
#
#     @. sol.vec[1:dim3] = sol_sub + sol_tau * const_sol
#     sol.tau[1] = sol_tau
#
#     return sol
# end

function solve_subsystem3(
    system_solver::QRCholSystemSolver,
    solver::Solver,
    sol_sub::Vector,
    rhs_sub::Vector,
    ) where {T <: Real}
    model = solver.model
    (n, p) = (model.n, model.p)

    copyto!(sol_sub, rhs_sub)
    @views x = sol_sub[system_solver.x_rows]
    @views y = sol_sub[system_solver.y_rows]
    @views z = sol_sub[system_solver.z_rows]

    copyto!(system_solver.QpbxGHbz, x) 
    mul!(system_solver.QpbxGHbz, model.G', z, true, true)
    lmul!(solver.Ap_Q', system_solver.QpbxGHbz)

    if !iszero(p)
        ldiv!(solver.Ap_R', y)
        sol_sub[1:p] = y

        if !isempty(system_solver.Q2div)
            mul!(system_solver.GQ1x, system_solver.GQ1, y)
            block_hess_prod.(model.cones, system_solver.HGQ1x_k, system_solver.GQ1x_k)
            mul!(system_solver.Q2div, system_solver.GQ2', system_solver.HGQ1x, -1, true)
        end
    end

    if !isempty(system_solver.Q2div)
        @views x_sub2 = copyto!(sol_sub[(p + 1):n], system_solver.Q2div)
        inv_prod(system_solver.fact_cache, x_sub2)
    end

    lmul!(solver.Ap_Q, x)

    mul!(system_solver.Gx, model.G, x)
    block_hess_prod.(model.cones, system_solver.HGx_k, system_solver.Gx_k)

    @. z = system_solver.HGx - z

    if !iszero(p)
        copyto!(y, system_solver.Q1pbxGHbz)
        mul!(y, system_solver.GQ1', system_solver.HGx, -1, true)
        ldiv!(solver.Ap_R, y)
    end

    return sol_sub
end

function block_hess_prod(cone_k::Cones.Cone{T}, prod_k::AbstractVecOrMat{T}, arr_k::AbstractVecOrMat{T}) where {T <: Real}
    if Cones.use_dual_barrier(cone_k)
        Cones.inv_hess_prod!(prod_k, arr_k, cone_k)
    else
        Cones.hess_prod!(prod_k, arr_k, cone_k)
    end
    return
end

#=
direct dense
=#

mutable struct QRCholDenseSystemSolver{T <: Real} <: QRCholSystemSolver{T}
    x_rows::UnitRange{Int}
    y_rows::UnitRange{Int}
    z_rows::UnitRange{Int}
    rhs_sub::Vector{T}
    sol_sub::Vector{T}
    const_sol::Vector{T}
    lhs1::Symmetric{T, Matrix{T}}
    inv_hess_cones::Vector{Int}
    inv_hess_sqrt_cones::Vector{Int}
    hess_cones::Vector{Int}
    hess_sqrt_cones::Vector{Int}
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
    fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} # can use BunchKaufman or Cholesky
    function QRCholDenseSystemSolver{T}(;
        fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} = DensePosDefCache{T}(), # NOTE or DenseSymCache{T}()
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::QRCholDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    nmp = n - p
    cone_idxs = model.cone_idxs

    system_solver.x_rows = 1:n
    system_solver.y_rows = n .+ (1:p)
    system_solver.z_rows = (n + p) .+ (1:q)

    dim3 = n + p + q
    system_solver.rhs_sub = zeros(T, dim3)
    system_solver.sol_sub = zeros(T, dim3)
    system_solver.lhs1 = Symmetric(zeros(T, nmp, nmp), :U)

    num_cones = length(cone_idxs)
    system_solver.inv_hess_cones = sizehint!(Int[], num_cones)
    system_solver.inv_hess_sqrt_cones = sizehint!(Int[], num_cones)
    system_solver.hess_cones = sizehint!(Int[], num_cones)
    system_solver.hess_sqrt_cones = sizehint!(Int[], num_cones)

    # NOTE very inefficient method used for sparse G * QRSparseQ : see https://github.com/JuliaLang/julia/issues/31124#issuecomment-501540818
    @timeit solver.timer "mul_G_Q" GQ = model.G * solver.Ap_Q

    system_solver.GQ2 = GQ[:, (p + 1):end]
    system_solver.HGQ2 = zeros(T, q, nmp)
    system_solver.QpbxGHbz = Vector{T}(undef, n)
    system_solver.Q2div = view(system_solver.QpbxGHbz, (p + 1):n)
    system_solver.Gx = zeros(T, q)
    system_solver.HGx = zeros(T, q)
    system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in cone_idxs]
    system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in cone_idxs]
    system_solver.HGx_k = [view(system_solver.HGx, idxs, :) for idxs in cone_idxs]
    system_solver.Gx_k = [view(system_solver.Gx, idxs, :) for idxs in cone_idxs]

    if !iszero(p)
        system_solver.GQ1 = GQ[:, 1:p]
        system_solver.Q1pbxGHbz = view(system_solver.QpbxGHbz, 1:p)
        system_solver.GQ1x = zeros(T, q)
        system_solver.HGQ1x = zeros(T, q)
        system_solver.HGQ1x_k = [view(system_solver.HGQ1x, idxs, :) for idxs in cone_idxs]
        system_solver.GQ1x_k = [view(system_solver.GQ1x, idxs, :) for idxs in cone_idxs]
    end

    load_matrix(system_solver.fact_cache, system_solver.lhs1)

    system_solver.const_sol = zeros(T, length(system_solver.rhs_sub))

    return system_solver
end

# NOTE move to dense.jl if useful elsewhere
outer_prod(A::AbstractMatrix{T}, B::AbstractMatrix{T}, alpha::Real, beta::Real) where {T <: LinearAlgebra.BlasReal} = BLAS.syrk!('U', 'T', alpha, A, beta, B)
outer_prod(A::AbstractMatrix{T}, B::AbstractMatrix{T}, alpha::Real, beta::Real) where {T <: Real} = mul!(B, A', A, alpha, beta)

function update_lhs(system_solver::QRCholDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    lhs = system_solver.lhs1.data

    if !isempty(system_solver.Q2div)
        inv_hess_cones = empty!(system_solver.inv_hess_cones)
        inv_hess_sqrt_cones = empty!(system_solver.inv_hess_sqrt_cones)
        hess_cones = empty!(system_solver.hess_cones)
        hess_sqrt_cones = empty!(system_solver.hess_sqrt_cones)

        # update hessian factorizations and partition of cones
        for (k, cone_k) in enumerate(model.cones)
            if Cones.use_sqrt_oracles(cone_k)
                cones_list = Cones.use_dual_barrier(cone_k) ? inv_hess_sqrt_cones : hess_sqrt_cones
            else
                cones_list = Cones.use_dual_barrier(cone_k) ? inv_hess_cones : hess_cones
            end
            push!(cones_list, k)
        end

        # do inv_hess and inv_hess_sqrt cones
        if isempty(inv_hess_sqrt_cones)
            lhs .= 0
        else
            idx = 1
            for k in inv_hess_sqrt_cones
                arr_k = system_solver.GQ2_k[k]
                q_k = size(arr_k, 1)
                @views prod_k = system_solver.HGQ2[idx:(idx + q_k - 1), :]
                Cones.inv_hess_sqrt_prod!(prod_k, arr_k, model.cones[k])
                idx += q_k
            end
            @views HGQ2_sub = system_solver.HGQ2[1:(idx - 1), :]
            outer_prod(HGQ2_sub, lhs, true, false)
        end

        for k in inv_hess_cones
            arr_k = system_solver.GQ2_k[k]
            prod_k = system_solver.HGQ2_k[k]
            Cones.inv_hess_prod!(prod_k, arr_k, model.cones[k])
            mul!(lhs, arr_k', prod_k, true, true)
        end

        # do hess and hess_sqrt cones
        if !isempty(hess_sqrt_cones)
            idx = 1
            for k in hess_sqrt_cones
                arr_k = system_solver.GQ2_k[k]
                q_k = size(arr_k, 1)
                @views prod_k = system_solver.HGQ2[idx:(idx + q_k - 1), :]
                Cones.hess_sqrt_prod!(prod_k, arr_k, model.cones[k])
                idx += q_k
            end
            @views HGQ2_sub = system_solver.HGQ2[1:(idx - 1), :]
            outer_prod(HGQ2_sub, lhs, true, true)
        end

        for k in hess_cones
            arr_k = system_solver.GQ2_k[k]
            prod_k = system_solver.HGQ2_k[k]
            Cones.hess_prod!(prod_k, arr_k, model.cones[k])
            mul!(lhs, arr_k', prod_k, true, true)
        end
    end

    # TODO refactor below
    if !isempty(system_solver.lhs1) && !update_fact(system_solver.fact_cache, system_solver.lhs1)
        # @warn("QRChol factorization failed")
        if T <: LinearAlgebra.BlasReal && system_solver.fact_cache isa DensePosDefCache{T}
            # @warn("switching QRChol solver from Cholesky to Bunch Kaufman")
            system_solver.fact_cache = DenseSymCache{T}()
            load_matrix(system_solver.fact_cache, system_solver.lhs1)
        else
            system_solver.lhs1 += sqrt(eps(T)) * I # attempt recovery # TODO make more efficient
        end
        if !update_fact(system_solver.fact_cache, system_solver.lhs1)
            system_solver.lhs1 += sqrt(eps(T)) * I # attempt recovery # TODO make more efficient
            if !update_fact(system_solver.fact_cache, system_solver.lhs1)
                @warn("QRChol Bunch-Kaufman factorization failed after recovery")
                @show system_solver.lhs1
                @assert !any(isnan, system_solver.lhs1)
            end
        end
    end

    # update solution for fixed c,b,h part
    rhs_sub = system_solver.rhs_sub
    @. rhs_sub[system_solver.x_rows] = -model.c
    rhs_sub[system_solver.y_rows] = model.b
    @views rhs_sub_z = rhs_sub[system_solver.z_rows]
    for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
        @views block_hess_prod(cone_k, rhs_sub_z[idxs_k], model.h[idxs_k])
    end
    solve_subsystem3(system_solver, solver, system_solver.const_sol, rhs_sub)

    return system_solver
end
