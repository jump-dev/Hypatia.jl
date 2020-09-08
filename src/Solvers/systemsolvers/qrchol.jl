#=
QR+Cholesky linear system solver
requires precomputed QR factorization of A'

solves linear system in naive.jl by first eliminating s, kap, and tau via the method in the symindef solver, then reducing the 3x3 symmetric indefinite system to a series of low-dimensional operations via a procedure similar to that described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf (the dominating subroutine is a positive definite linear solve with RHS of dimension n-p x 3)

TODO iterative refinement for 3x3 solution
=#

abstract type QRCholSystemSolver{T <: Real} <: SystemSolver{T} end

function setup_rhs3(
    ::QRCholSystemSolver{T},
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
            z_temp_k = sol.z_views[k]
            @. z_temp_k = -rhs_z_k - rhs_s_k
            Cones.inv_hess_prod!(rhs_sub_z_k, z_temp_k, cone_k)
        else
            Cones.hess_prod!(rhs_sub_z_k, rhs_z_k, cone_k)
            axpby!(-1, rhs_s_k, -1, rhs_sub_z_k)
        end
    end
    return nothing
end

function solve_subsystem3(
    system_solver::QRCholSystemSolver,
    solver::Solver,
    sol_sub::Point,
    rhs_sub::Point,
    ) where {T <: Real}
    model = solver.model
    copyto!(sol_sub.vec, rhs_sub.vec)
    x = sol_sub.x
    y = sol_sub.y
    z = sol_sub.z

    copyto!(system_solver.QpbxGHbz, x)
    mul!(system_solver.QpbxGHbz, model.G', z, true, true)
    lmul!(solver.Ap_Q', system_solver.QpbxGHbz)

    if !iszero(model.p)
        ldiv!(solver.Ap_R', y)
        sol_sub[1:model.p] = y

        if !isempty(system_solver.Q2div)
            mul!(system_solver.GQ1x, system_solver.GQ1, y)
            block_hess_prod.(model.cones, system_solver.HGQ1x_k, system_solver.GQ1x_k)
            mul!(system_solver.Q2div, system_solver.GQ2', system_solver.HGQ1x, -1, true)
        end
    end

    if !isempty(system_solver.Q2div)
        @views x_sub2 = copyto!(sol_sub.vec[(model.p + 1):model.n], system_solver.Q2div)
        inv_prod(system_solver.fact_cache, x_sub2)
    end

    lmul!(solver.Ap_Q, x)

    mul!(system_solver.Gx, model.G, x)
    block_hess_prod.(model.cones, system_solver.HGx_k, system_solver.Gx_k)

    @. z = system_solver.HGx - z

    if !iszero(model.p)
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
    lhs1::Symmetric{T, Matrix{T}}
    inv_hess_cones::Vector{Int}
    inv_hess_sqrt_cones::Vector{Int}
    hess_cones::Vector{Int}
    hess_sqrt_cones::Vector{Int}
    rhs_sub::Point{T}
    sol_sub::Point{T}
    sol_const::Point{T}
    rhs_const::Point{T}
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

    setup_point_sub(system_solver, model)

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
    rhs_const = system_solver.rhs_const
    for (k, cone_k) in enumerate(model.cones)
        block_hess_prod(cone_k, rhs_const.z_views[k], model.h[model.cone_idxs[k]])
    end
    @timeit solver.timer "solve_subsystem3" solve_subsystem3(system_solver, solver, system_solver.sol_const, rhs_const)

    return system_solver
end
