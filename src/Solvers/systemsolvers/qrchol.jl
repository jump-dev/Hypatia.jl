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
    syssolver::QRCholSystemSolver{T},
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    ) where {T <: Real}
    model = solver.model
    copyto!(sol.vec, rhs.vec)
    x = sol.x
    y = sol.y
    z = sol.z

    copyto!(syssolver.QpbxGHbz, x)
    mul!(syssolver.QpbxGHbz, model.G', z, true, true)
    lmul!(solver.Ap_Q', syssolver.QpbxGHbz)

    if !iszero(model.p)
        ldiv!(solver.Ap_R', y)
        @views copyto!(sol.vec[1:model.p], y)

        if !isempty(syssolver.Q2div)
            mul!(syssolver.GQ1x, syssolver.GQ1, y)
            block_hess_prod!.(syssolver.HGQ1x_k, syssolver.GQ1x_k, model.cones)
            mul!(syssolver.Q2div, syssolver.GQ2', syssolver.HGQ1x, -1, true)
        end
    end

    if !isempty(syssolver.Q2div)
        @views x_sub2 = sol.vec[(model.p + 1):model.n]
        ldiv!(x_sub2, syssolver.fact, syssolver.Q2div)
    end

    lmul!(solver.Ap_Q, x)

    mul!(syssolver.Gx, model.G, x)
    block_hess_prod!.(syssolver.HGx_k, syssolver.Gx_k, model.cones)

    axpby!(true, syssolver.HGx, -1, z)

    if !iszero(model.p)
        copyto!(y, syssolver.Q1pbxGHbz)
        mul!(y, syssolver.GQ1', syssolver.HGx, -1, true)
        ldiv!(solver.Ap_R, y)
    end

    return sol
end

function block_hess_prod!(
    prod_k::AbstractVecOrMat{T},
    arr_k::AbstractVecOrMat{T},
    cone_k::Cones.Cone{T},
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
    lhs_sub::Symmetric{T, Matrix{T}}
    lhs_sub_fact::Symmetric{T, Matrix{T}}
    fact::Factorization{T}

    rhs_sub::Point{T}
    sol_sub::Point{T}
    sol_const::Point{T}
    rhs_const::Point{T}

    use_sqrt_hess_cones::Vector{Bool}
    GQ1::AbstractMatrix{T}
    GQ2::AbstractMatrix{T}
    QpbxGHbz::Vector{T}
    Q1pbxGHbz::SubArray
    Q2div::SubArray
    GQ1x::Vector{T}
    HGQ1x::Vector{T}
    HGQ2::Matrix{T}
    Gx::Vector{T}
    HGx::Vector{T}
    HGQ1x_k::Vector
    GQ1x_k::Vector
    HGQ2_k::Vector
    GQ2_k::Vector
    HGx_k::Vector
    Gx_k::Vector

    function QRCholDenseSystemSolver{T}() where {T <: Real}
        syssolver = new{T}()
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

    syssolver.lhs_sub = Symmetric(zeros(T, nmp, nmp), :U)
    syssolver.lhs_sub_fact = zero(syssolver.lhs_sub)

    syssolver.use_sqrt_hess_cones = falses(length(cone_idxs))

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

    setup_point_sub(syssolver, model)

    return syssolver
end

function update_lhs(
    syssolver::QRCholDenseSystemSolver{T},
    solver::Solver{T},
    ) where {T <: Real}
    model = solver.model

    # update LHS and factorization
    isempty(syssolver.Q2div) || update_lhs_fact(syssolver, solver)

    # update solution for fixed c,b,h part
    rhs_const = syssolver.rhs_const
    @inbounds for (k, cone_k) in enumerate(model.cones)
        @views h_k = model.h[model.cone_idxs[k]]
        block_hess_prod!(rhs_const.z_views[k], h_k, cone_k)
    end
    solve_subsystem3(syssolver, solver, syssolver.sol_const, rhs_const)

    return syssolver
end

function update_lhs_fact(
    syssolver::QRCholDenseSystemSolver{T},
    solver::Solver{T},
    ) where {T <: Real}
    model = solver.model
    lhs = syssolver.lhs_sub.data
    cones = model.cones
    use_sqrt_hess_cones = syssolver.use_sqrt_hess_cones
    HGQ2 = syssolver.HGQ2
    GQ2_k = syssolver.GQ2_k
    HGQ2_k = syssolver.HGQ2_k

    nmp = size(lhs, 1)
    @inbounds for k in eachindex(cones)
        use_sqrt_hess_cones[k] = Cones.use_sqrt_hess_oracles(nmp, cones[k])
    end

    # sqrt cones
    if any(use_sqrt_hess_cones)
        idx = 1
        @inbounds for k in eachindex(cones)
            use_sqrt_hess_cones[k] || continue
            cone_k = cones[k]
            arr_k = GQ2_k[k]
            q_k = size(arr_k, 1)
            @views prod_k = HGQ2[idx:(idx + q_k - 1), :]
            if Cones.use_dual_barrier(cone_k)
                Cones.inv_sqrt_hess_prod!(prod_k, arr_k, cone_k)
            else
                Cones.sqrt_hess_prod!(prod_k, arr_k, cone_k)
            end
            idx += q_k
        end
        @views outer_prod!(HGQ2[1:(idx - 1), :], lhs, true, false)
    else
        fill!(lhs, 0)
    end

    # not sqrt cones
    @inbounds for k in eachindex(cones)
        use_sqrt_hess_cones[k] && continue
        arr_k = GQ2_k[k]
        prod_k = HGQ2_k[k]
        block_hess_prod!(prod_k, arr_k, cones[k])
        mul!(lhs, arr_k', prod_k, true, true)
    end

    # TODO try equilibration, iterative refinement etc like posvx/sysvx
    solver.time_upfact += @elapsed syssolver.fact =
        posdef_fact_copy!(syssolver.lhs_sub_fact, syssolver.lhs_sub)

    if !issuccess(syssolver.fact)
        println("positive definite linear system factorization failed")
    end

    return
end
