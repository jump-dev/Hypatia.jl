#=
interpolation-based weighted-sum-of-squares polynomial ell-1 norm cone parametrized by interpolation matrices Ps
certifies that u(x) <= sum(abs.(w(x))) for all x in the domain described by input Ps
u(x), w_1(x), ...,  w_R(x) are polynomials with U coefficients
=#

mutable struct WSOSInterpEpiNormOne{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    nu::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    correction::Vector{T}
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
    ΛLi_Λ::Vector{Vector{Matrix{T}}}
    Λ11::Vector{Matrix{T}}
    tempΛ11::Vector{Matrix{T}}
    tempLL::Vector{Matrix{T}}
    tempLU::Vector{Matrix{T}}
    tempLU2::Vector{Matrix{T}}
    Λ11LiP::Vector{Matrix{T}}
    tempUU_vec::Vector{Matrix{T}} # reused in update_hess
    tempUU::Matrix{T}
    tempUU2::Matrix{T}
    tempURU::Matrix{T}
    tempURU2::Matrix{T}
    PΛiPs1::Vector{Vector{Matrix{T}}} # for each (2, 2)-block pertaining to (lambda_1, lambda_i), P * inv(Λ)[1, 1] * Ps = P * inv(Λ)i[2, 2] * Ps
    PΛiPs2::Vector{Vector{Matrix{T}}} # for each (2, 2)-block pertaining to (lambda_1, lambda_i), P * inv(Λ)[2, 1] * Ps = P * inv(Λ)[1, 2]' * Ps
    ΛLiPs11::Vector{Vector{Matrix{T}}}
    ΛLiPs12::Vector{Vector{Matrix{T}}}
    ΛLiP_dir11::Vector{Vector{Matrix{T}}}
    ΛLiP_dir12::Vector{Vector{Matrix{T}}}
    ΛLiP_dir21::Vector{Vector{Matrix{T}}}
    corr_half::Vector{Vector{Matrix}}
    Λfact::Vector
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
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert R >= 2
        for Pk in Ps
            @assert size(Pk, 1) == U
        end
        cone = new{T}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.dim = U * R
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        cone.nu = R * sum(size(Pk, 2) for Pk in Ps)
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
    Ls = [size(Pk, 2) for Pk in cone.Ps]
    cone.mats = [[zeros(T, L, L) for _ in 1:(R - 1)] for L in Ls]
    cone.matfact = [[cholesky(hcat([one(T)])) for _ in 1:R] for _ in cone.Ps] # TODO preallocate better
    cone.hess_edge_blocks = [zeros(T, U, U) for _ in 1:(R - 1)]
    cone.hess_diag_blocks = [zeros(T, U, U) for _ in 1:R]
    cone.hess_diag_facts = Any[cholesky(hcat([one(T)])) for _ in 1:(R - 1)] # TODO preallocate better
    cone.hess_diags = [zeros(T, U, U) for _ in 1:R - 1]
    cone.ΛLi_Λ = [[zeros(T, L, L) for _ in 1:(R - 1)] for L in Ls]
    cone.Λ11 = [zeros(T, L, L) for L in Ls]
    cone.tempΛ11 = [zeros(T, L, L) for L in Ls]
    cone.tempLL = [zeros(T, L, L) for L in Ls]
    cone.tempLU = [zeros(T, L, U) for L in Ls]
    cone.tempLU2 = [zeros(T, L, U) for L in Ls]
    cone.Λ11LiP = [zeros(T, L, U) for L in Ls]
    cone.tempUU_vec = [zeros(T, U, U) for _ in Ps]
    cone.tempUU = zeros(T, U, U)
    cone.tempUU2 = zeros(T, U, U)
    cone.tempURU = zeros(T, U, U * (R - 1))
    cone.tempURU2 = zeros(T, U * (R - 1), U)
    cone.PΛiPs1 = [Vector{Matrix{T}}(undef, R) for Pk in Ps]
    cone.PΛiPs2 = [Vector{Matrix{T}}(undef, R) for Pk in Ps]
    cone.PΛiPs1 = [[zeros(T, U, U) for _ in 1:(R - 1)] for Pk in Ps]
    cone.PΛiPs2 = [[zeros(T, U, U) for _ in 1:(R - 1)] for Pk in Ps]
    cone.ΛLiPs11 = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.ΛLiPs12 = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.ΛLiP_dir11 = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.ΛLiP_dir12 = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.ΛLiP_dir21 = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.corr_half = [[zeros(T, 2 * L, 2 * U) for _ in 1:(R - 1)] for L in Ls]
    cone.Λfact = Vector{Any}(undef, K)
    cone.point_views = [view(cone.point, block_idxs(U, i)) for i in 1:R]
    cone.Ps_times = zeros(K)
    cone.Ps_order = collect(1:K)
    return cone
end

reset_data(cone::WSOSInterpEpiNormOne) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = cone.hess_prod_updated = cone.inv_hess_prod_updated = false)

use_sqrt_hess_oracles(::WSOSInterpEpiNormOne) = false # Hessian is block sparse

function set_initial_point(arr::AbstractVector, cone::WSOSInterpEpiNormOne)
    @views arr[1:cone.U] .= 1
    @views arr[(cone.U + 1):end] .= 0
    return arr
end

function update_feas(cone::WSOSInterpEpiNormOne)
    @assert !cone.feas_updated
    U = cone.U
    R = cone.R
    Λfact = cone.Λfact
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
            ΛLi_Λ = cone.ΛLi_Λ[k]
            matsk = cone.mats[k]
            factk = cone.matfact[k]

            # first lambda
            @. LUk = Pk' * point_views[1]'
            mul!(tempΛ11j, LUk, Pk)
            copyto!(Λ11j, tempΛ11j)
            Λfact[k] = cholesky!(Symmetric(tempΛ11j, :U), check = false)
            if !isposdef(Λfact[k])
                cone.is_feas = false
                break
            end

            uo = U + 1
            @inbounds for r in 2:R
                r1 = r - 1
                matr = matsk[r1]
                factr = factk[r1]
                @. LUk = Pk' * point_views[r]'
                mul!(LLk, LUk, Pk)

                ldiv!(ΛLi_Λ[r1], Λfact[k].L, LLk)
                copyto!(matr, Λ11j)
                mul!(matr, ΛLi_Λ[r1]', ΛLi_Λ[r1], -1, 1)

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
    Λfact = cone.Λfact
    matfact = cone.matfact
    grad = cone.grad
    grad .= 0

    @inbounds for k in eachindex(cone.Ps)
        Pk = cone.Ps[k]
        Λ11LiP = cone.Λ11LiP[k]
        ΛLi_Λ = cone.ΛLi_Λ[k]
        factk = cone.matfact[k]
        ΛLiPs11 = cone.ΛLiPs11[k]
        ΛLiPs12 = cone.ΛLiPs12[k]

        # prep ΛLiPs
        ldiv!(Λ11LiP, cone.Λfact[k].L, Pk')

        for r in 1:(R - 1)
            mul!(ΛLiPs12[r], ΛLi_Λ[r]', Λ11LiP, -1, false)
            ldiv!(factk[r].L, ΛLiPs12[r])
            ldiv!(ΛLiPs11[r], factk[r].L, Pk')
        end

        @views for u in 1:U
            grad[u] -= sum(abs2, Λ11LiP[:, u]) * R
            idx = u + U
            for r in 1:(R - 1)
                grad[u] -= 2 * sum(abs2, ΛLiPs12[r][:, u])
                grad[idx] -= 2 * dot(ΛLiPs12[r][:, u], ΛLiPs11[r][:, u])
                idx += U
            end
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
        Λ11LiP = cone.Λ11LiP[k]
        PΛ11iP = cone.tempUU_vec[k]
        PΛiPs1 = cone.PΛiPs1[k]
        PΛiPs2 = cone.PΛiPs2[k]
        ΛLiPs11 = cone.ΛLiPs11[k]
        ΛLiPs12 = cone.ΛLiPs12[k]

        # P * inv(Λ_11) * P' for (1, 1) hessian block and adding to PΛiPs[r][r]
        mul!(PΛ11iP, Λ11LiP', Λ11LiP)
        for r in 1:(R - 1)
            mul!(PΛiPs2[r], ΛLiPs12[r]', ΛLiPs11[r])
            mul!(PΛiPs1[r], ΛLiPs12[r]', ΛLiPs12[r])
            @. PΛiPs1[r] += PΛ11iP
            for j in 1:U, i in 1:j
                ij1 = PΛiPs1[r][i, j]
                ij2 = PΛiPs2[r][i, j]
                uu = 2 * (abs2(ij1) + abs2(ij2))
                cone.hess_diag_blocks[1][i, j] += uu
                cone.hess_diag_blocks[r + 1][i, j] += uu
                cone.hess_edge_blocks[r][i, j] += 4 * (ij1 * ij2)
            end
        end
        for j in 1:U, i in 1:j
            cone.hess_diag_blocks[1][i, j] -= abs2(PΛ11iP[i, j]) * R2
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

function correction(cone::WSOSInterpEpiNormOne, primal_dir::AbstractVector)
    @assert cone.grad_updated
    corr = cone.correction
    corr .= 0
    R = cone.R
    U = cone.U

    @inbounds for k in eachindex(cone.Ps)
        @views mul!(cone.tempLU[k], cone.Λ11LiP[k], Diagonal(primal_dir[1:U]))
        tempLU2 = mul!(cone.tempLU2[k], cone.tempLU[k], cone.tempUU_vec[k])
        @views for u in 1:U
            corr[u] += sum(abs2, tempLU2[:, u])
        end
    end
    @. @views corr[1:U] *= 2 - R

    @inbounds for k in eachindex(cone.Ps)
        L = size(cone.Ps[k], 2)
        PΛiPs1 = cone.PΛiPs1[k]
        PΛiPs2 = cone.PΛiPs2[k]
        ΛLiPs11 = cone.ΛLiPs11[k]
        ΛLiPs12 = cone.ΛLiPs12[k]
        Λ11LiP = cone.Λ11LiP[k]
        tempLU = cone.tempLU[k]
        ΛLiP_dir11 = cone.ΛLiP_dir11[k]
        ΛLiP_dir12 = cone.ΛLiP_dir12[k]
        ΛLiP_dir21 = cone.ΛLiP_dir21[k]
        factk = cone.matfact[k]
        corr_half = cone.corr_half[k]

        @views ΛLiP_dir22 = mul!(tempLU, Λ11LiP, Diagonal(primal_dir[1:U]))
        @views for r in 2:R
            mul!(ΛLiP_dir11[r - 1], ΛLiPs11[r - 1], Diagonal(primal_dir[1:U]))
            mul!(ΛLiP_dir11[r - 1], ΛLiPs12[r - 1], Diagonal(primal_dir[block_idxs(U, r)]), true, true)
            mul!(ΛLiP_dir12[r - 1], ΛLiPs11[r - 1], Diagonal(primal_dir[block_idxs(U, r)]))
            mul!(ΛLiP_dir12[r - 1], ΛLiPs12[r - 1], Diagonal(primal_dir[1:U]), true, true)
            mul!(ΛLiP_dir21[r - 1], Λ11LiP, Diagonal(primal_dir[block_idxs(U, r)]))
        end

        @views for s in 1:(R - 1)
            mul!(corr_half[s][1:L, 1:U], ΛLiP_dir11[s], PΛiPs1[s])
            mul!(corr_half[s][1:L, 1:U], ΛLiP_dir12[s], PΛiPs2[s], true, true)
            mul!(corr_half[s][1:L, (U + 1):(2 * U)], ΛLiP_dir11[s], PΛiPs2[s])
            mul!(corr_half[s][1:L, (U + 1):(2 * U)], ΛLiP_dir12[s], PΛiPs1[s], true, true)
            mul!(corr_half[s][(L + 1):(2 * L), 1:U], ΛLiP_dir21[s], PΛiPs1[s])
            mul!(corr_half[s][(L + 1):(2 * L), 1:U], ΛLiP_dir22, PΛiPs2[s], true, true)
            mul!(corr_half[s][(L + 1):(2 * L), (U + 1):(2 * U)], ΛLiP_dir21[s], PΛiPs2[s])
            mul!(corr_half[s][(L + 1):(2 * L), (U + 1):(2 * U)], ΛLiP_dir22, PΛiPs1[s], true, true)
        end

        idx = U + 1
        @views for s in 1:(R - 1), u in 1:U
            corr[u] += sum(abs2, corr_half[s][:, u])
            corr[u] += sum(abs2, corr_half[s][:, U + u])
            corr[idx] += 2 * dot(corr_half[s][:, U + u], corr_half[s][:, u])
            idx += 1
        end

    end

    return corr
end
