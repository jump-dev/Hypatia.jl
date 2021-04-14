#=
interpolation-based weighted-sum-of-squares (multivariate) polynomial epinormeucl (AKA second-order cone) parametrized by interpolation matrices Ps
certifies that u(x)^2 <= sum(w_i(x)^2) for all x in the domain described by input Ps
u(x), w_1(x), ...,  w_R(x) are polynomials with U coefficients

dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
and "Semidefinite Characterization of Sum-of-Squares Cones in Algebras" by D. Papp and F. Alizadeh
-logdet(schur(Lambda)) - logdet(Lambda_11)
note that if schur(M) = A - B * inv(D) * C then
logdet(schur) = logdet(M) - logdet(D) = logdet(Lambda) - (R - 1) * logdet(Lambda_11)
since our D is an (R - 1) x (R - 1) block diagonal matrix
=#

mutable struct WSOSInterpEpiNormEucl{T <: Real} <: Cone{T}
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
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    mat::Vector{Matrix{T}}
    matfact::Vector
    ΛLi_Λ::Vector{Vector{Matrix{T}}}
    Λ11::Vector{Matrix{T}}
    tempLL::Vector{Matrix{T}}
    tempLU::Vector{Matrix{T}}
    tempLU2::Vector{Matrix{T}}
    tempLU_vec::Vector{Vector{Matrix{T}}}
    tempLU_vec2::Vector{Vector{Matrix{T}}}
    tempLU_vec3::Vector{Matrix{T}}
    tempLL_vec::Vector{Vector{Matrix{T}}}
    tempLL_vec2::Vector{Matrix{T}}
    tempLRUR::Vector{Matrix{T}}
    tempUU::Matrix{T}
    ΛLiPs_edge::Vector{Vector{Matrix{T}}}
    PΛiPs::Vector{Matrix{T}}
    PΛiP_blocks_U::Vector
    Λ11LiP::Vector{Matrix{T}} # also equal to the block on the diagonal of ΛLiP
    matLiP::Vector{Matrix{T}} # also equal to block (1, 1) of ΛLiP
    PΛ11iP::Vector{Matrix{T}}
    Λfact::Vector
    point_views::Vector
    Ps_times::Vector{Float64}
    Ps_order::Vector{Int}

    function WSOSInterpEpiNormEucl{T}(
        R::Int,
        U::Int,
        Ps::Vector{Matrix{T}};
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        for Pj in Ps
            @assert size(Pj, 1) == U
        end
        cone = new{T}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.dim = U * R
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        cone.nu = 2 * sum(size(Pk, 2) for Pk in Ps)
        return cone
    end
end

function setup_extra_data(cone::WSOSInterpEpiNormEucl{T}) where {T <: Real}
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    K = length(Ps)
    Ls = [size(Pk, 2) for Pk in cone.Ps]
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.mat = [zeros(T, L, L) for L in Ls]
    cone.matfact = Vector{Any}(undef, K)
    cone.ΛLi_Λ = [[zeros(T, L, L) for _ in 1:(R - 1)] for L in Ls]
    cone.Λ11 = [zeros(T, L, L) for L in Ls]
    cone.tempLL = [zeros(T, L, L) for L in Ls]
    cone.tempLU = [zeros(T, L, U) for L in Ls]
    cone.tempLU2 = [zeros(T, L, U) for L in Ls]
    cone.tempLU_vec = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.tempLU_vec2 = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.tempLU_vec3 = [zeros(T, L, U * R) for L in Ls]
    cone.tempLL_vec = [[zeros(T, L, L) for _ in 1:(R - 1)] for L in Ls]
    cone.tempLL_vec2 = [zeros(R * L, L) for L in Ls]
    cone.tempLRUR = [zeros(T, L * R, U * R) for L in Ls]
    cone.tempUU = zeros(T, U, U)
    cone.ΛLiPs_edge = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.matLiP = [zeros(T, L, U) for L in Ls]
    cone.PΛiPs = [zeros(T, R * U, R * U) for _ in eachindex(Ls)]
    cone.Λ11LiP = [zeros(T, L, U) for L in Ls]
    cone.PΛ11iP = [zeros(T, U, U) for _ in eachindex(Ps)]
    cone.PΛiP_blocks_U = [[view(PΛiPk, block_idxs(U, r), block_idxs(U, s)) for r in 1:R, s in 1:R] for PΛiPk in cone.PΛiPs]
    cone.Λfact = Vector{Any}(undef, K)
    cone.point_views = [view(cone.point, block_idxs(U, i)) for i in 1:R]
    cone.Ps_times = zeros(K)
    cone.Ps_order = collect(1:K)
    return cone
end

function set_initial_point(arr::AbstractVector, cone::WSOSInterpEpiNormEucl)
    @views arr[1:cone.U] .= 1
    @views arr[(cone.U + 1):end] .= 0
    return arr
end

function update_feas(cone::WSOSInterpEpiNormEucl)
    @assert !cone.feas_updated
    U = cone.U
    Λfact = cone.Λfact
    matfact = cone.matfact
    point_views = cone.point_views

    # order the Ps by how long it takes to check feasibility, to improve efficiency
    sortperm!(cone.Ps_order, cone.Ps_times, initialized = true) # NOTE stochastic

    cone.is_feas = true
    for k in cone.Ps_order
        cone.Ps_times[k] = @elapsed @inbounds begin
            Pk = cone.Ps[k]
            Λ11k = cone.Λ11[k]
            LUk = cone.tempLU[k]
            ΛLi_Λ = cone.ΛLi_Λ[k]
            mat = cone.mat[k]

            # first lambda
            @. LUk = Pk' * point_views[1]'
            mul!(Λ11k, LUk, Pk)
            copyto!(mat, Λ11k)
            Λfact[k] = cholesky!(Symmetric(Λ11k, :U), check = false)
            if !isposdef(Λfact[k])
                cone.is_feas = false
                break
            end

            # subtract others
            uo = U + 1
            for r in 1:(cone.R - 1)
                @. LUk = Pk' * point_views[r + 1]'
                mul!(ΛLi_Λ[r], LUk, Pk)
                ldiv!(Λfact[k].L, ΛLi_Λ[r])
                mul!(mat, ΛLi_Λ[r]', ΛLi_Λ[r], -1, true)
                uo += U
            end

            matfact[k] = cholesky!(Symmetric(mat, :U), check = false)
            if !isposdef(matfact[k])
                cone.is_feas = false
                break
            end
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSInterpEpiNormEucl)
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
        ΛLiP_edge = cone.ΛLiPs_edge[k]
        matLiP = cone.matLiP[k]
        ΛLi_Λ = cone.ΛLi_Λ[k]

        ldiv!(Λ11LiP, cone.Λfact[k].L, Pk') # TODO may be more efficient to do ldiv(fact.U', B) than ldiv(fact.L, B) here and elsewhere since the factorizations are of symmetric :U matrices

        # prep PΛiP halves
        ldiv!(matLiP, matfact[k].L, Pk')
        # top edge of ΛLiP
        for r in 1:(R - 1)
            mul!(ΛLiP_edge[r], ΛLi_Λ[r]', Λ11LiP, -1, false)
            ldiv!(matfact[k].L, ΛLiP_edge[r])
        end

        @views for u in 1:U
            grad[u] -= sum(abs2, Λ11LiP[:, u]) + sum(abs2, matLiP[:, u])
            idx = U + u
            for r in 2:R
                grad[idx] -= 2 * dot(ΛLiP_edge[r - 1][:, u], matLiP[:, u])
                grad[u] -= sum(abs2, ΛLiP_edge[r - 1][:, u])
                idx += U
            end
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpEpiNormEucl)
    @assert cone.grad_updated
    U = cone.U
    R = cone.R
    R2 = R - 2
    hess = cone.hess.data
    UU = cone.tempUU
    matfact = cone.matfact

    hess .= 0
    @inbounds for k in eachindex(cone.Ps)
        PΛiPs = cone.PΛiP_blocks_U[k]
        PΛ11iP = cone.PΛ11iP[k]
        ΛLiP_edge = cone.ΛLiPs_edge[k]
        matLiP = cone.matLiP[k]
        Λ11LiP = cone.Λ11LiP[k]

        # prep PΛiPs
        # P * inv(Λ_11) * P' for (1, 1) hessian block and adding to PΛiPs[r][r]
        mul!(PΛ11iP, Λ11LiP', Λ11LiP)
        # block-(1,1) is P * inv(mat) * P'
        mul!(PΛiPs[1, 1], matLiP', matLiP)
        # get all the PΛiPs that are in row one or on the diagonal
        for r in 2:R
            mul!(PΛiPs[r, 1], ΛLiP_edge[r - 1]', matLiP)
            mul!(PΛiPs[r, r], ΛLiP_edge[r - 1]', ΛLiP_edge[r - 1])
            @. PΛiPs[r, r] += PΛ11iP
            for r2 in 2:(r - 1)
                mul!(PΛiPs[r, r2], ΛLiP_edge[r - 1]', ΛLiP_edge[r2 - 1])
            end
        end
        copytri!(cone.PΛiPs[k], 'L')

        for i in 1:U, k in 1:i
            hess[k, i] -= abs2(PΛ11iP[k, i]) * R2
        end

        @. @views hess[1:U, 1:U] += abs2(PΛiPs[1, 1])
        for r in 2:R
            idxs = block_idxs(U, r)
            for s in 1:(r - 1)
                # block (1,1)
                @. UU = abs2(PΛiPs[r, s])
                @. @views hess[1:U, 1:U] += UU
                @. @views hess[1:U, 1:U] += UU'
                # blocks (1,r)
                @. @views hess[1:U, idxs] += PΛiPs[s, 1] * PΛiPs[s, r]
            end
            # block (1,1)
            @. @views hess[1:U, 1:U] += abs2(PΛiPs[r, r])
            # blocks (1,r)
            @. @views hess[1:U, idxs] += PΛiPs[r, 1] * PΛiPs[r, r]
            # blocks (1,r)
            for s in (r + 1):R
                @. @views hess[1:U, idxs] += PΛiPs[s, 1] * PΛiPs[s, r]
            end

            # blocks (r, r2)
            # NOTE for hess[idxs, idxs], UU are symmetric
            @. UU = PΛiPs[r, 1] * PΛiPs[r, 1]'
            @. @views hess[idxs, idxs] += UU
            @. UU = PΛiPs[1, 1] * PΛiPs[r, r]
            @. @views hess[idxs, idxs] += UU
            for r2 in (r + 1):R
                idxs2 = block_idxs(U, r2)
                @. UU = PΛiPs[r, 1] * PΛiPs[r2, 1]'
                @. @views hess[idxs, idxs2] += UU
                @. UU = PΛiPs[1, 1] * PΛiPs[r2, r]'
                @. @views hess[idxs, idxs2] += UU
            end
        end
    end
    @. @views hess[:, (U + 1):cone.dim] *= 2

    cone.hess_updated = true
    return cone.hess
end

function correction(cone::WSOSInterpEpiNormEucl, primal_dir::AbstractVector)
    @assert cone.hess_updated
    corr = cone.correction
    corr .= 0
    R = cone.R
    U = cone.U

    @inbounds for k in eachindex(cone.Ps)
        Pk = cone.Ps[k]
        LUk = cone.tempLU[k]
        LLk = cone.tempLL[k]
        ΛFLPk = cone.Λ11LiP[k]

        @views mul!(LUk, ΛFLPk, Diagonal(primal_dir[1:U]))
        mul!(LLk, LUk, ΛFLPk')
        mul!(LUk, Hermitian(LLk), ΛFLPk)
        @views for u in 1:U
            corr[u] += sum(abs2, LUk[:, u])
        end
    end
    @. @views corr[1:U] *= 2 - R

    @inbounds for k in eachindex(cone.Ps)
        L = size(cone.Ps[k], 2)
        ΛLiP_edge = cone.ΛLiPs_edge[k]
        matLiP = cone.matLiP[k]
        Λ11LiP = cone.Λ11LiP[k]
        scaled_row = cone.tempLU_vec[k]
        scaled_col = cone.tempLU_vec2[k]
        ΛLiP_edge_ctgs = cone.tempLU_vec3[k]
        ΛLiP_D_ΛLiPt_row = cone.tempLL_vec[k]
        ΛLiP_D_ΛLiPt_col = cone.tempLL_vec2[k]
        ΛLiP_D_ΛLiPt_diag = cone.tempLL[k]
        corr_half = cone.tempLRUR[k]
        corr_half .= 0

        # get ΛLiP * D * PΛiP where D is diagonalized primal_dir scattered in an arrow and ΛLiP is half an arrow
        # ΛLiP * D is an arrow matrix but row edge doesn't equal column edge
        @views scaled_diag = mul!(cone.tempLU[k], Λ11LiP, Diagonal(primal_dir[1:U]))
        @views scaled_pt = mul!(cone.tempLU2[k], matLiP, Diagonal(primal_dir[1:U]))
        @views for r in 2:R
            mul!(scaled_pt, ΛLiP_edge[r - 1], Diagonal(primal_dir[block_idxs(U, r)]), true, true)
            mul!(scaled_row[r - 1], matLiP, Diagonal(primal_dir[block_idxs(U, r)]))
            mul!(scaled_row[r - 1], ΛLiP_edge[r - 1], Diagonal(primal_dir[1:U]), true, true)
            mul!(scaled_col[r - 1], Λ11LiP, Diagonal(primal_dir[block_idxs(U, r)]))
        end

        # ΛLiP is half an arrow, multiplying an arrow by its transpose gives another arrow
        @views mul!(ΛLiP_D_ΛLiPt_col[1:L, :], scaled_pt, matLiP')
        @views for r in 2:R
            mul!(ΛLiP_D_ΛLiPt_col[1:L, :], scaled_row[r - 1], ΛLiP_edge[r - 1]', true, true)
            mul!(ΛLiP_D_ΛLiPt_row[r - 1], scaled_row[r - 1], Λ11LiP')
            mul!(ΛLiP_D_ΛLiPt_col[block_idxs(L, r), :], scaled_col[r - 1], matLiP')
            mul!(ΛLiP_D_ΛLiPt_col[block_idxs(L, r), :], scaled_diag, ΛLiP_edge[r - 1]', true, true)
            mul!(ΛLiP_D_ΛLiPt_diag, scaled_diag, Λ11LiP')
        end

        # TODO check that this is worthwhile
        @. @views ΛLiP_edge_ctgs[:, 1:U] = matLiP
        for r in 2:R
            @. @views ΛLiP_edge_ctgs[:, block_idxs(U, r)] = ΛLiP_edge[r - 1]
        end

        # multiply by ΛLiP
        mul!(corr_half, ΛLiP_D_ΛLiPt_col, ΛLiP_edge_ctgs)
        @views for r in 2:R
            mul!(corr_half[1:L, block_idxs(U, r)], ΛLiP_D_ΛLiPt_row[r - 1], Λ11LiP, true, true)
            mul!(corr_half[block_idxs(L, r), block_idxs(U, r)], ΛLiP_D_ΛLiPt_diag, Λ11LiP, true, true)
        end

        @views for u in 1:U
            corr[u] += sum(abs2, corr_half[:, u])
            idx = U + u
            for r in 2:R
                corr[idx] += 2 * dot(corr_half[:, idx], corr_half[:, u])
                corr[u] += sum(abs2, corr_half[:, idx])
                idx += U
            end
        end

    end

    return corr
end
