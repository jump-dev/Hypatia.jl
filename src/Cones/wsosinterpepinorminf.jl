#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

=#

mutable struct WSOSInterpEpiNormInf{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    point::AbstractVector{T}
    dual_point::AbstractVector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    inv_hess_prod_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    barrier::Function

    mats::Vector{Vector{Matrix{T}}}
    matfact::Vector{Vector}
    Λi_Λ::Vector{Vector{Matrix{T}}}
    Λ11::Vector{Matrix{T}}
    tmpΛ11::Vector{Matrix{T}}
    tmpLL::Vector{Matrix{T}}
    tmpLU::Vector{Matrix{T}}
    tmpLU2::Vector{Matrix{T}}
    tmpUU_vec::Vector{Matrix{T}} # reused in update_hess
    tmpUU::Matrix{T}
    tmpUU2::Matrix{T}
    tmpURU::Matrix{T}
    tmpURU2::Matrix{T}
    PΛiPs1::Vector{Vector{Matrix{T}}} # for each (2, 2)-block pertaining to (lambda_1, lambda_i), P * inv(Λ)[1, 1] * Ps = P * inv(Λ)i[2, 2] * Ps
    PΛiPs2::Vector{Vector{Matrix{T}}} # for each (2, 2)-block pertaining to (lambda_1, lambda_i), P * inv(Λ)[2, 1] * Ps = P * inv(Λ)[1, 2]' * Ps
    lambdafact::Vector
    hess_edge_blocks::Vector{Matrix{T}}
    hess_diag_blocks::Vector{Matrix{T}}
    hess_diag_facts::Vector
    hess_diags::Vector{Matrix{T}}
    hess_schur_fact
    point_views

    function WSOSInterpEpiNormInf{T}(
        R::Int,
        U::Int,
        Ps::Vector{Matrix{T}};
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        for Pk in Ps
            @assert size(Pk, 1) == U
        end
        cone = new{T}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = U * R
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_data(cone::WSOSInterpEpiNormInf{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = similar(cone.point)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)

    cone.mats = [[Matrix{Any}(undef, size(Pk, 2), size(Pk, 2)) for _ in 1:(R - 1)] for Pk in cone.Ps]
    cone.matfact = [[cholesky(hcat([one(T)])) for _ in 1:R] for _ in cone.Ps] # TODO preallocate better
    cone.hess_edge_blocks = [zeros(T, U, U) for _ in 1:(R - 1)]
    cone.hess_diag_blocks = [zeros(T, U, U) for _ in 1:R]
    cone.hess_diag_facts = Any[cholesky(hcat([one(T)])) for _ in 1:(R - 1)] # TODO preallocate better
    cone.hess_diags = [zeros(T, U, U) for _ in 1:R - 1]
    cone.Λi_Λ = [Vector{Matrix{T}}(undef, R - 1) for Psk in Ps]
    @inbounds for k in eachindex(Ps), r in 1:(R - 1)
        cone.Λi_Λ[k][r] = similar(cone.grad, size(Ps[k], 2), size(Ps[k], 2))
    end
    cone.Λ11 = [similar(cone.grad, size(Psk, 2), size(Psk, 2)) for Psk in Ps]
    cone.tmpΛ11 = [similar(cone.grad, size(Psk, 2), size(Psk, 2)) for Psk in Ps]
    cone.tmpLL = [similar(cone.grad, size(Psk, 2), size(Psk, 2)) for Psk in Ps]
    cone.tmpLU = [similar(cone.grad, size(Psk, 2), U) for Psk in Ps]
    cone.tmpLU2 = [similar(cone.grad, size(Psk, 2), U) for Psk in Ps]
    cone.tmpUU_vec = [similar(cone.grad, U, U) for _ in eachindex(Ps)]
    cone.tmpUU = zeros(T, U, U)
    cone.tmpUU2 = zeros(T, U, U)
    cone.tmpURU = zeros(T, U, U * (R - 1))
    cone.tmpURU2 = zeros(T, U * (R - 1), U)
    cone.PΛiPs1 = [Vector{Matrix{T}}(undef, R) for Psk in Ps]
    cone.PΛiPs2 = [Vector{Matrix{T}}(undef, R) for Psk in Ps]
    @inbounds for k in eachindex(Ps), r in 1:(R - 1)
        cone.PΛiPs1[k][r] = similar(cone.grad, U, U)
        cone.PΛiPs2[k][r] = similar(cone.grad, U, U)
    end
    cone.lambdafact = Vector{Any}(undef, length(Ps))
    cone.point_views = [view(cone.point, block_idxs(U, i)) for i in 1:R]
    return
end

reset_data(cone::WSOSInterpEpiNormInf) = (cone.feas_updated = cone.grad_updated = cone.hess_updated =
    cone.inv_hess_updated = cone.hess_fact_updated = cone.hess_prod_updated = cone.inv_hess_prod_updated = false)

get_nu(cone::WSOSInterpEpiNormInf) = cone.R * sum(size(Pk, 2) for Pk in cone.Ps)

use_sqrt_oracles(::WSOSInterpEpiNormInf) = false

function set_initial_point(arr::AbstractVector, cone::WSOSInterpEpiNormInf)
    arr[1:cone.U] .= 1
    arr[(cone.U + 1):end] .= 0
    return arr
end

function update_feas(cone::WSOSInterpEpiNormInf)
    @assert !cone.feas_updated
    U = cone.U
    R = cone.R
    lambdafact = cone.lambdafact
    point_views = cone.point_views

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ps)
        Psk = cone.Ps[k]
        Λ11j = cone.Λ11[k]
        tmpΛ11j = cone.tmpΛ11[k]
        LLk = cone.tmpLL[k]
        LUk = cone.tmpLU[k]
        Λi_Λ = cone.Λi_Λ[k]
        matsk = cone.mats[k]
        factk = cone.matfact[k]

        # first lambda
        @. LUk = Psk' * point_views[1]'
        mul!(tmpΛ11j, LUk, Psk)
        copyto!(Λ11j, tmpΛ11j)
        lambdafact[k] = cholesky!(Symmetric(tmpΛ11j, :U), check = false)
        if !isposdef(lambdafact[k])
            cone.is_feas = false
            break
        end

        uo = U + 1
        @inbounds for r in 2:R
            r1 = r - 1
            matr = matsk[r1]
            factr = factk[r1]
            @. LUk = Psk' * point_views[r]'
            mul!(LLk, LUk, Psk)

            # not using lambdafact.L \ lambda with an syrk because storing lambdafact \ lambda is useful later
            ldiv!(Λi_Λ[r1], lambdafact[k], LLk)
            copyto!(matr, Λ11j)
            mul!(matr, LLk, Λi_Λ[r1], -1, 1)

            # h = lambdafact[k].L \ LLk
            # matr = Λ11j - h' * h

            # ldiv!(lambdafact[k].L, LLk)
            # mat = Λ11j - LLk' * LLk
            factk[r1] = cholesky!(Symmetric(matr, :U), check = false)
            if !isposdef(factk[r1])
                cone.is_feas = false
                cone.feas_updated = true
                return cone.is_feas
            end
            uo += U
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::WSOSInterpEpiNormInf) = true

# TODO common code could be refactored with epinormeucl version
function update_grad(cone::WSOSInterpEpiNormInf)
    @assert cone.is_feas
    U = cone.U
    R = cone.R
    R2 = R - 2
    lambdafact = cone.lambdafact
    matfact = cone.matfact

    cone.grad .= 0
    @inbounds for k in eachindex(cone.Ps)
        Psk = cone.Ps[k]
        LUk = cone.tmpLU[k]
        LUk2 = cone.tmpLU2[k]
        UUk = cone.tmpUU_vec[k]
        PΛiPs1 = cone.PΛiPs1[k]
        PΛiPs2 = cone.PΛiPs2[k]
        Λi_Λ = cone.Λi_Λ[k]

        # P * inv(Λ_11) * P' for (1, 1) hessian block and adding to PΛiPs[r][r]
        ldiv!(LUk, cone.lambdafact[k].L, Psk')
        mul!(UUk, LUk', LUk)

        # prep PΛiPs
        # get all the PΛiPs that are in row one or on the diagonal
        @inbounds for r in 1:(R - 1)
            # block-(1,1) is P * inv(mat) * P'
            ldiv!(LUk, matfact[k][r].L, Psk')
            mul!(PΛiPs1[r], LUk', LUk)
            # block (1,2)
            ldiv!(LUk, matfact[k][r], Psk')
            mul!(LUk2, Λi_Λ[r], LUk)
            mul!(PΛiPs2[r], Psk, LUk2, -1, false)
            # Λr = Psk' * Diagonal(cone.point[block_idxs(U, r + 1)]) * Psk
            # lil = cone.lambdafact[k] \ Λr
            # PΛiPs2[r] = -Psk * (cone.lambdafact[k] \ (Λr * (matfact[k][r] \ Psk')))
        end

        # (1, 1)-block
        # gradient is diag of sum(-PΛiPs[i][i] for i in 1:R) + (R - 1) * Lambda_11 - Lambda_11
        @inbounds for i in 1:U
            cone.grad[i] += UUk[i, i] * R2
            @inbounds for r in 1:(R - 1)
                cone.grad[i] -= PΛiPs1[r][i, i] * 2
            end
        end
        idx = U + 1
        @inbounds for r in 1:(R - 1), i in 1:U
            cone.grad[idx] -= 2 * PΛiPs2[r][i, i]
            idx += 1
        end
    end # j

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpEpiNormInf)
    @timeit cone.timer "hess" begin
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    U = cone.U
    H = cone.hess.data
    H .= 0

    @. @views H[1:U, 1:U] = cone.hess_diag_blocks[1]
    @inbounds for r in 1:(cone.R - 1)
        idxs = block_idxs(U, r + 1)
        LinearAlgebra.copytri!(cone.hess_edge_blocks[r], 'U')
        @. @views H[1:U, idxs] = cone.hess_edge_blocks[r]
        @. @views H[idxs, idxs] = cone.hess_diag_blocks[r + 1]
    end

    cone.hess_updated = true
    end # timer
    return cone.hess
end

function update_hess_prod(cone::WSOSInterpEpiNormInf)
    @timeit cone.timer "updatehessprod" begin
    @assert cone.grad_updated

    U = cone.U
    R = cone.R
    R2 = R - 2

    @inbounds for r in 1:(R - 1)
        cone.hess_diag_blocks[r] .= 0
        cone.hess_edge_blocks[r] .= 0
    end
    cone.hess_diag_blocks[R] .= 0

    @inbounds for k in eachindex(cone.Ps)
        PΛiPs1 = cone.PΛiPs1[k]
        PΛiPs2 = cone.PΛiPs2[k]
        UUk = cone.tmpUU_vec[k]

        for r in 1:(R - 1), j in 1:U, i in 1:j
            ij1 = PΛiPs1[r][i, j]
            ij2 = (PΛiPs2[r][i, j] + PΛiPs2[r][j, i]) / 2 # NOTE PΛiPs2[r] should be symmetric
            uu = 2 * (abs2(ij1) + abs2(ij2))
            cone.hess_diag_blocks[1][i, j] += uu
            cone.hess_diag_blocks[r + 1][i, j] += uu
            cone.hess_edge_blocks[r][i, j] += 4 * (ij1 * ij2)
        end

        for j in 1:U, i in 1:j
            cone.hess_diag_blocks[1][i, j] -= abs2(UUk[i, j]) * R2
        end
    end

    end # timer
    cone.hess_prod_updated = true
    return prod
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::WSOSInterpEpiNormInf)
    @timeit cone.timer "hessprod" begin
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    U = cone.U
    R = cone.R
    prod .= 0

    @views mul!(prod[1:U, :], Symmetric(cone.hess_diag_blocks[1], :U), arr[1:U, :])
    @inbounds for r in 1:(R - 1)
        idxs = block_idxs(U, r + 1)
        edge_r = Symmetric(cone.hess_edge_blocks[r], :U)
        @views arr_r = arr[idxs, :]
        @views mul!(prod[1:U, :], edge_r, arr_r, true, true)
        @views mul!(prod[idxs, :], edge_r, arr[1:U, :])
        @views mul!(prod[idxs, :], Symmetric(cone.hess_diag_blocks[r + 1], :U), arr_r, true, true)
    end

    end # timer
    return prod
end

function update_inv_hess_prod(cone::WSOSInterpEpiNormInf)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    U = cone.U
    R = cone.R
    H = cone.hess.data
    schur = cone.tmpUU2
    @views copyto!(schur, H[1:U, 1:U])
    edge = cone.tmpURU
    @views copyto!(edge, H[1:U, (U + 1):end])
    Diz = cone.tmpURU2

    @inbounds for r in 2:R
        r1 = r - 1
        diag_r = cone.hess_diags[r - 1]
        idxs = block_idxs(U, r)
        idxs2 = block_idxs(U, r - 1)
        @views z = H[1:U, idxs]
        @views copyto!(diag_r, H[idxs, idxs])
        cone.hess_diag_facts[r1] = cholesky!(Symmetric(diag_r, :U), check = false)
        if !isposdef(cone.hess_diag_facts[r1])
            # cone.hess_diag_facts[r1] = cholesky!(Symmetric(diag_r + I, :U)) # TODO save graciously
            cone.hess_diag_facts[r1] = bunchkaufman!(Symmetric(diag_r, :U)) # TODO save graciously
        end
        @views ldiv!(Diz[idxs2, :], cone.hess_diag_facts[r - 1], z)
        @views mul!(schur, z', Diz[idxs2, :], -1, true)
    end

    s_fact = cone.hess_schur_fact = cholesky!(Symmetric(schur, :U), check = false)
    if !isposdef(s_fact)
        # s_fact = cone.hess_schur_fact = cholesky!(Symmetric(schur + I * 10, :U))
        s_fact = cone.hess_schur_fact = bunchkaufman!(Symmetric(schur, :U))
    end

    cone.inv_hess_prod_updated = true
    return
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::WSOSInterpEpiNormInf)
    @timeit cone.timer "ihessprod" begin
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    U = cone.U
    R = cone.R

    edge = cone.tmpURU
    @views copyto!(edge, cone.hess[1:U, (U + 1):end])
    Diz = cone.tmpURU2

    @inbounds for r in 2:R
        idxs = block_idxs(U, r)
        @views ldiv!(prod[idxs, :], cone.hess_diag_facts[r - 1], arr[idxs, :])
    end
    # prod += u * inv(schur) * u' * arr
    s_fact = cone.hess_schur_fact
    ldiv!(s_fact, edge)
    @inbounds for j in 1:size(arr, 2)
        @views Dix = prod[(U + 1):end, j]
        @views a1j = arr[1:U, j]
        @views ldiv!(prod[1:U, j], s_fact, a1j)
        @views mul!(prod[1:U, j], edge, Dix, -1, true)
        @views mul!(prod[(U + 1):end, j], Diz, prod[1:U, j], -1, true)
    end
    end # timer

    return prod
end

use_correction(::WSOSInterpEpiNormInf) = false
