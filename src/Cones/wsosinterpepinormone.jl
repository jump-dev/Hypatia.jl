#=
interpolation-based weighted-sum-of-squares polynomial ell-1 norm cone parametrized by interpolation matrices Ps
certifies that u(x) <= sum(abs.(w(x))) for all x in the domain described by input Ps
u(x), w_1(x), ...,  w_R(x) are polynomials with U coefficients
=#

mutable struct WSOSInterpEpiNormOne{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    inv_hess_prod_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    mats::Vector{Vector{Matrix{T}}}
    matfact::Vector{Vector}
    Λi_Λ::Vector{Vector{Matrix{T}}}
    Λ11::Vector{Matrix{T}}
    tempΛ11::Vector{Matrix{T}}
    tempLL::Vector{Matrix{T}}
    tempLU::Vector{Matrix{T}}
    tempLU2::Vector{Matrix{T}}
    tempUU_vec::Vector{Matrix{T}} # reused in update_hess
    tempUU::Matrix{T}
    tempUU2::Matrix{T}
    tempURU::Matrix{T}
    tempURU2::Matrix{T}
    PΛiPs1::Vector{Vector{Matrix{T}}} # for each (2, 2)-block pertaining to (lambda_1, lambda_i), P * inv(Λ)[1, 1] * Ps = P * inv(Λ)i[2, 2] * Ps
    PΛiPs2::Vector{Vector{Matrix{T}}} # for each (2, 2)-block pertaining to (lambda_1, lambda_i), P * inv(Λ)[2, 1] * Ps = P * inv(Λ)[1, 2]' * Ps
    lambdafact::Vector
    hess_edge_blocks::Vector{Matrix{T}}
    hess_diag_blocks::Vector{Matrix{T}}
    hess_diag_facts::Vector
    hess_diags::Vector{Matrix{T}}
    hess_schur_fact
    point_views::Vector
    Ps_times::Vector{Float64}
    Ps_order::Vector{Int}

    function WSOSInterpEpiNormOne{T}(
        R::Int,
        U::Int,
        Ps::Vector{Matrix{T}};
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        for Pk in Ps
            @assert size(Pk, 1) == U
        end
        cone = new{T}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.dim = U * R
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data(cone::WSOSInterpEpiNormOne{T}) where {T <: Real}
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    K = length(Ps)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.mats = [[zeros(T, size(Pk, 2), size(Pk, 2)) for _ in 1:(R - 1)] for Pk in cone.Ps]
    cone.matfact = [[cholesky(hcat([one(T)])) for _ in 1:R] for _ in cone.Ps] # TODO preallocate better
    cone.hess_edge_blocks = [zeros(T, U, U) for _ in 1:(R - 1)]
    cone.hess_diag_blocks = [zeros(T, U, U) for _ in 1:R]
    cone.hess_diag_facts = Any[cholesky(hcat([one(T)])) for _ in 1:(R - 1)] # TODO preallocate better
    cone.hess_diags = [zeros(T, U, U) for _ in 1:R - 1]
    cone.Λi_Λ = [Vector{Matrix{T}}(undef, R - 1) for Pk in Ps]
    @inbounds for k in eachindex(Ps), r in 1:(R - 1)
        cone.Λi_Λ[k][r] = zeros(T, size(Ps[k], 2), size(Ps[k], 2))
    end
    cone.Λ11 = [zeros(T, size(Pk, 2), size(Pk, 2)) for Pk in Ps]
    cone.tempΛ11 = [zeros(T, size(Pk, 2), size(Pk, 2)) for Pk in Ps]
    cone.tempLL = [zeros(T, size(Pk, 2), size(Pk, 2)) for Pk in Ps]
    cone.tempLU = [zeros(T, size(Pk, 2), U) for Pk in Ps]
    cone.tempLU2 = [zeros(T, size(Pk, 2), U) for Pk in Ps]
    cone.tempUU_vec = [zeros(T, U, U) for _ in eachindex(Ps)]
    cone.tempUU = zeros(T, U, U)
    cone.tempUU2 = zeros(T, U, U)
    cone.tempURU = zeros(T, U, U * (R - 1))
    cone.tempURU2 = zeros(T, U * (R - 1), U)
    cone.PΛiPs1 = [Vector{Matrix{T}}(undef, R) for Pk in Ps]
    cone.PΛiPs2 = [Vector{Matrix{T}}(undef, R) for Pk in Ps]
    @inbounds for k in eachindex(Ps), r in 1:(R - 1)
        cone.PΛiPs1[k][r] = zeros(T, U, U)
        cone.PΛiPs2[k][r] = zeros(T, U, U)
    end
    cone.lambdafact = Vector{Any}(undef, K)
    cone.point_views = [view(cone.point, block_idxs(U, i)) for i in 1:R]
    cone.Ps_times = zeros(K)
    cone.Ps_order = collect(1:K)
    return cone
end

reset_data(cone::WSOSInterpEpiNormOne) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = cone.hess_prod_updated = cone.inv_hess_prod_updated = false)

use_correction(::WSOSInterpEpiNormOne) = false

get_nu(cone::WSOSInterpEpiNormOne) = cone.R * sum(size(Pk, 2) for Pk in cone.Ps)

use_sqrt_oracles(::WSOSInterpEpiNormOne) = false

function set_initial_point(arr::AbstractVector, cone::WSOSInterpEpiNormOne)
    @views arr[1:cone.U] .= 1
    @views arr[(cone.U + 1):end] .= 0
    return arr
end

function update_feas(cone::WSOSInterpEpiNormOne)
    @assert !cone.feas_updated
    U = cone.U
    R = cone.R
    lambdafact = cone.lambdafact
    point_views = cone.point_views

    # order the Ps by how long it takes to check feasibility, to improve efficiency
    sortperm!(cone.Ps_order, cone.Ps_times, initialized = true) # NOTE stochastic

    cone.is_feas = true
    for k in cone.Ps_order
        cone.Ps_times[k] = @elapsed @inbounds begin
            Pk = cone.Ps[k]
            Λ11j = cone.Λ11[k]
            tempΛ11j = cone.tempΛ11[k]
            LLk = cone.tempLL[k]
            LUk = cone.tempLU[k]
            Λi_Λ = cone.Λi_Λ[k]
            matsk = cone.mats[k]
            factk = cone.matfact[k]

            # first lambda
            @. LUk = Pk' * point_views[1]'
            mul!(tempΛ11j, LUk, Pk)
            copyto!(Λ11j, tempΛ11j)
            lambdafact[k] = cholesky!(Symmetric(tempΛ11j, :U), check = false)
            if !isposdef(lambdafact[k])
                cone.is_feas = false
                break
            end

            uo = U + 1
            for r in 2:R
                r1 = r - 1
                matr = matsk[r1]
                factr = factk[r1]
                @. LUk = Pk' * point_views[r]'
                mul!(LLk, LUk, Pk)
                # not using lambdafact.L \ lambda with an syrk because storing lambdafact \ lambda is useful later
                ldiv!(Λi_Λ[r1], lambdafact[k], LLk)
                copyto!(matr, Λ11j)
                mul!(matr, LLk, Λi_Λ[r1], -1, 1)

                factk[r1] = cholesky!(Symmetric(matr, :U), check = false)
                if !isposdef(factk[r1])
                    cone.is_feas = false
                    cone.feas_updated = true
                    return cone.is_feas
                end
                uo += U
            end
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(::WSOSInterpEpiNormOne) = true

function update_grad(cone::WSOSInterpEpiNormOne)
    @assert cone.is_feas
    U = cone.U
    R = cone.R
    R2 = R - 2
    lambdafact = cone.lambdafact
    matfact = cone.matfact

    cone.grad .= 0
    @inbounds for k in eachindex(cone.Ps)
        Pk = cone.Ps[k]
        LUk = cone.tempLU[k]
        LUk2 = cone.tempLU2[k]
        UUk = cone.tempUU_vec[k]
        PΛiPs1 = cone.PΛiPs1[k]
        PΛiPs2 = cone.PΛiPs2[k]
        Λi_Λ = cone.Λi_Λ[k]

        # P * inv(Λ_11) * P' for (1, 1) hessian block and adding to PΛiPs[r][r]
        ldiv!(LUk, cone.lambdafact[k].L, Pk')
        mul!(UUk, LUk', LUk)

        # prep PΛiPs
        # get all the PΛiPs that are in row one or on the diagonal
        @inbounds for r in 1:(R - 1)
            # block-(1,1) is P * inv(mat) * P'
            ldiv!(LUk, matfact[k][r].L, Pk')
            mul!(PΛiPs1[r], LUk', LUk)
            # block (1,2)
            ldiv!(LUk, matfact[k][r], Pk')
            mul!(LUk2, Λi_Λ[r], LUk)
            mul!(PΛiPs2[r], Pk, LUk2, -1, false)
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
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpEpiNormOne)
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
    return cone.hess
end

function update_hess_prod(cone::WSOSInterpEpiNormOne)
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

        UUk = cone.tempUU_vec[k]
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

    cone.hess_prod_updated = true
    return prod
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::WSOSInterpEpiNormOne)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    U = cone.U
    R = cone.R
    prod .= 0

    @views mul!(prod[1:U, :], Symmetric(cone.hess_diag_blocks[1], :U), arr[1:U, :])
    @inbounds @views for r in 1:(R - 1)
        idxs = block_idxs(U, r + 1)
        edge_r = Symmetric(cone.hess_edge_blocks[r], :U)
        arr_r = arr[idxs, :]
        mul!(prod[1:U, :], edge_r, arr_r, true, true)
        mul!(prod[idxs, :], edge_r, arr[1:U, :])
        mul!(prod[idxs, :], Symmetric(cone.hess_diag_blocks[r + 1], :U), arr_r, true, true)
    end

    return prod
end

function update_inv_hess_prod(cone::WSOSInterpEpiNormOne{T}) where T
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    U = cone.U
    R = cone.R
    schur = cone.tempUU2
    hess_diag_facts = cone.hess_diag_facts
    Diz = cone.tempURU2
    schur_backup = cone.tempUU

    copyto!(schur, Symmetric(cone.hess_diag_blocks[1], :U))

    @inbounds for r in 2:R
        r1 = r - 1
        diag_r = cone.hess_diags[r1]

        copyto!(diag_r, cone.hess_diag_blocks[r])
        r_fact = hess_diag_facts[r1] = cholesky!(Symmetric(diag_r, :U), check = false)
        if !isposdef(r_fact)
            # attempt recovery NOTE can do what hessian factorization fallback does
            copyto!(diag_r, cone.hess_diag_blocks[r])
            increase_diag!(diag_r)
            r_fact = hess_diag_facts[r1] = cholesky!(Symmetric(diag_r, :U), check = false)
            if !isposdef(r_fact)
                copyto!(diag_r, cone.hess_diag_blocks[r])
                if T <: BlasReal # TODO refac
                    hess_diag_facts[r1] = bunchkaufman!(Symmetric(diag_r, :U), true)
                else
                    hess_diag_facts[r1] = lu!(Symmetric(diag_r, :U))
                end
            end
        end

        z = cone.hess_edge_blocks[r1]
        LinearAlgebra.copytri!(z, 'U')
        idxs2 = block_idxs(U, r1)
        @views Dizi = Diz[idxs2, :]
        ldiv!(Dizi, hess_diag_facts[r1], z)
        mul!(schur, z', Dizi, -1, true)
    end

    copyto!(schur_backup, schur)
    s_fact = cone.hess_schur_fact = cholesky!(Symmetric(schur_backup, :U), check = false)
    if !isposdef(s_fact)
        # attempt recovery NOTE: can do what hessian factorization fallback
        copyto!(schur_backup, schur)
        increase_diag!(schur_backup)
        s_fact = cone.hess_schur_fact = cholesky!(Symmetric(schur_backup, :U), check = false)
        if !isposdef(s_fact)
            copyto!(schur_backup, schur)
                if T <: BlasReal # TODO refac
                cone.hess_schur_fact = bunchkaufman!(Symmetric(schur_backup, :U), true)
            else
                cone.hess_schur_fact = lu!(Symmetric(schur_backup, :U))
            end
        end
    end

    cone.inv_hess_prod_updated = true
    return
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::WSOSInterpEpiNormOne)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    U = cone.U
    R = cone.R
    edge = cone.tempURU
    Diz = cone.tempURU2
    s_fact = cone.hess_schur_fact

    @inbounds for r in 1:(R - 1)
        idxs = block_idxs(U, r)
        @views copyto!(edge[:, idxs], cone.hess_edge_blocks[r])
    end
    @inbounds for r in 2:R
        idxs = block_idxs(U, r)
        @views ldiv!(prod[idxs, :], cone.hess_diag_facts[r - 1], arr[idxs, :])
    end
    # prod += u * inv(schur) * u' * arr
    ldiv!(s_fact, edge)
    @inbounds @views for j in 1:size(arr, 2)
        Dix = prod[(U + 1):end, j]
        p1j = prod[1:U, j]
        ldiv!(p1j, s_fact, arr[1:U, j])
        mul!(p1j, edge, Dix, -1, true)
        mul!(Dix, Diz, p1j, -1, true)
    end

    return prod
end
