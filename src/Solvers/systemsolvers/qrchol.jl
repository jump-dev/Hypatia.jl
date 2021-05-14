#=
QR+Cholesky linear system solver
requires precomputed QR factorization of A'

solves linear system in naive.jl by first eliminating s, kap, and tau via the
method in the symindef solver, then reducing the 3x3 symmetric indefinite system
to a series of low-dimensional operations via a procedure similar to that
described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

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
    return
end

function solve_subsystem3(
    syssolver::QRCholSystemSolver,
    solver::Solver,
    sol_sub::Point,
    rhs_sub::Point,
    ) where {T <: Real}
    model = solver.model
    copyto!(sol_sub.vec, rhs_sub.vec)
    x = sol_sub.x
    y = sol_sub.y
    z = sol_sub.z

    copyto!(syssolver.QpbxGHbz, x)
    mul!(syssolver.QpbxGHbz, model.G', z, true, true)
    lmul!(solver.Ap_Q', syssolver.QpbxGHbz)

    if !iszero(model.p)
        ldiv!(solver.Ap_R', y)
        sol_sub.vec[1:model.p] = y

        if !isempty(syssolver.Q2div)
            mul!(syssolver.GQ1x, syssolver.GQ1, y)
            block_hess_prod.(model.cones, syssolver.HGQ1x_k,
                syssolver.GQ1x_k)
            mul!(syssolver.Q2div, syssolver.GQ2', syssolver.HGQ1x,
                -1, true)
        end
    end

    if !isempty(syssolver.Q2div)
        @views x_sub2 = copyto!(sol_sub.vec[(model.p + 1):model.n],
            syssolver.Q2div)
        inv_prod(syssolver.fact_cache, x_sub2)
    end

    lmul!(solver.Ap_Q, x)

    mul!(syssolver.Gx, model.G, x)
    block_hess_prod.(model.cones, syssolver.HGx_k, syssolver.Gx_k)

    @. z = syssolver.HGx - z

    if !iszero(model.p)
        copyto!(y, syssolver.Q1pbxGHbz)
        mul!(y, syssolver.GQ1', syssolver.HGx, -1, true)
        ldiv!(solver.Ap_R, y)
    end

    return sol_sub
end

function block_hess_prod(
    cone_k::Cones.Cone{T},
    prod_k::AbstractVecOrMat{T},
    arr_k::AbstractVecOrMat{T},
    ) where {T <: Real}
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
    inv_sqrt_hess_cones::Vector{Int}
    hess_cones::Vector{Int}
    sqrt_hess_cones::Vector{Int}
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
    fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}}
    function QRCholDenseSystemSolver{T}(;
        fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} =
            DensePosDefCache{T}(), # or DenseSymCache{T}()
        ) where {T <: Real}
        syssolver = new{T}()
        syssolver.fact_cache = fact_cache
        return syssolver
    end
end

function load(
    syssolver::QRCholDenseSystemSolver{T},
    solver::Solver{T},
    ) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    nmp = n - p
    cone_idxs = model.cone_idxs

    syssolver.lhs1 = Symmetric(zeros(T, nmp, nmp), :U)

    num_cones = length(cone_idxs)
    syssolver.inv_hess_cones = sizehint!(Int[], num_cones)
    syssolver.inv_sqrt_hess_cones = sizehint!(Int[], num_cones)
    syssolver.hess_cones = sizehint!(Int[], num_cones)
    syssolver.sqrt_hess_cones = sizehint!(Int[], num_cones)

    # very inefficient method used for sparse G * QRSparseQ
    # see https://github.com/JuliaLang/julia/issues/31124#issuecomment-501540818
    GQ = model.G * solver.Ap_Q

    syssolver.GQ2 = GQ[:, (p + 1):end]
    syssolver.HGQ2 = zeros(T, q, nmp)
    syssolver.QpbxGHbz = zeros(T, n)
    syssolver.Q2div = view(syssolver.QpbxGHbz, (p + 1):n)
    syssolver.Gx = zeros(T, q)
    syssolver.HGx = zeros(T, q)
    syssolver.HGQ2_k = [view(syssolver.HGQ2, idxs, :) for idxs in cone_idxs]
    syssolver.GQ2_k = [view(syssolver.GQ2, idxs, :) for idxs in cone_idxs]
    syssolver.HGx_k = [view(syssolver.HGx, idxs, :) for idxs in cone_idxs]
    syssolver.Gx_k = [view(syssolver.Gx, idxs, :) for idxs in cone_idxs]

    if !iszero(p)
        syssolver.GQ1 = GQ[:, 1:p]
        syssolver.Q1pbxGHbz = view(syssolver.QpbxGHbz, 1:p)
        syssolver.GQ1x = zeros(T, q)
        syssolver.HGQ1x = zeros(T, q)
        syssolver.HGQ1x_k = [view(syssolver.HGQ1x, idxs, :) for idxs in cone_idxs]
        syssolver.GQ1x_k = [view(syssolver.GQ1x, idxs, :) for idxs in cone_idxs]
    end

    load_matrix(syssolver.fact_cache, syssolver.lhs1)

    setup_point_sub(syssolver, model)

    return syssolver
end

function update_lhs(
    syssolver::QRCholDenseSystemSolver{T},
    solver::Solver{T},
    ) where {T <: Real}
    model = solver.model
    lhs = syssolver.lhs1.data

    if !isempty(syssolver.Q2div)
        inv_hess_cones = empty!(syssolver.inv_hess_cones)
        inv_sqrt_hess_cones = empty!(syssolver.inv_sqrt_hess_cones)
        hess_cones = empty!(syssolver.hess_cones)
        sqrt_hess_cones = empty!(syssolver.sqrt_hess_cones)

        # update hessian factorizations and partition of cones
        for (k, cone_k) in enumerate(model.cones)
            if Cones.use_sqrt_hess_oracles(cone_k)
                cones_list = Cones.use_dual_barrier(cone_k) ?
                    inv_sqrt_hess_cones : sqrt_hess_cones
            else
                cones_list = Cones.use_dual_barrier(cone_k) ?
                    inv_hess_cones : hess_cones
            end
            push!(cones_list, k)
        end

        # do inv_hess and inv_sqrt_hess cones
        if isempty(inv_sqrt_hess_cones)
            lhs .= 0
        else
            idx = 1
            for k in inv_sqrt_hess_cones
                arr_k = syssolver.GQ2_k[k]
                q_k = size(arr_k, 1)
                @views prod_k = syssolver.HGQ2[idx:(idx + q_k - 1), :]
                Cones.inv_sqrt_hess_prod!(prod_k, arr_k, model.cones[k])
                idx += q_k
            end
            @views HGQ2_sub = syssolver.HGQ2[1:(idx - 1), :]
            outer_prod(HGQ2_sub, lhs, true, false)
        end

        for k in inv_hess_cones
            arr_k = syssolver.GQ2_k[k]
            prod_k = syssolver.HGQ2_k[k]
            Cones.inv_hess_prod!(prod_k, arr_k, model.cones[k])
            mul!(lhs, arr_k', prod_k, true, true)
        end

        # do hess and sqrt_hess cones
        if !isempty(sqrt_hess_cones)
            idx = 1
            for k in sqrt_hess_cones
                arr_k = syssolver.GQ2_k[k]
                q_k = size(arr_k, 1)
                @views prod_k = syssolver.HGQ2[idx:(idx + q_k - 1), :]
                Cones.sqrt_hess_prod!(prod_k, arr_k, model.cones[k])
                idx += q_k
            end
            @views HGQ2_sub = syssolver.HGQ2[1:(idx - 1), :]
            outer_prod(HGQ2_sub, lhs, true, true)
        end

        for k in hess_cones
            arr_k = syssolver.GQ2_k[k]
            prod_k = syssolver.HGQ2_k[k]
            Cones.hess_prod!(prod_k, arr_k, model.cones[k])
            mul!(lhs, arr_k', prod_k, true, true)
        end
    end

    start_time = time()
    # TODO refactor below
    if !isempty(syssolver.lhs1) &&
        !update_fact(syssolver.fact_cache, syssolver.lhs1)
        # @warn("QRChol factorization failed")
        if T <: LinearAlgebra.BlasReal &&
            syssolver.fact_cache isa DensePosDefCache{T}
            # @warn("switching QRChol solver from Cholesky to Bunch Kaufman")
            syssolver.fact_cache = DenseSymCache{T}()
            load_matrix(syssolver.fact_cache, syssolver.lhs1)
        else
            # attempt recovery
            increase_diag!(syssolver.lhs1.data)
        end
        if !update_fact(syssolver.fact_cache, syssolver.lhs1)
            # attempt recovery # TODO make more efficient
            syssolver.lhs1 += sqrt(eps(T)) * I
            if !update_fact(syssolver.fact_cache, syssolver.lhs1)
                @warn("QRChol Bunch-Kaufman factorization failed after recovery")
                @assert !any(isnan, syssolver.lhs1)
            end
        end
    end
    solver.time_upfact += time() - start_time

    # update solution for fixed c,b,h part
    rhs_const = syssolver.rhs_const
    for (k, cone_k) in enumerate(model.cones)
        @inbounds @views h_k = model.h[model.cone_idxs[k]]
        block_hess_prod(cone_k, rhs_const.z_views[k], h_k)
    end
    solve_subsystem3(syssolver, solver, syssolver.sol_const, rhs_const)

    return syssolver
end
