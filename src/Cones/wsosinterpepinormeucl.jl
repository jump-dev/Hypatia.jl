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
    use_heuristic_neighborhood::Bool
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}

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
    tempLU::Vector{Matrix{T}}
    tempLU2::Vector{Matrix{T}}
    tempLU_vec::Vector{Vector{Matrix{T}}}
    tempLU_vec2::Vector{Vector{Matrix{T}}}
    tempLRUR::Vector{Matrix{T}}
    tempUU_vec::Vector{Matrix{T}} # reused in update_hess
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
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        for Pj in Ps
            @assert size(Pj, 1) == U
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
    cone.tempLU = [zeros(T, L, U) for L in Ls]
    cone.tempLU2 = [zeros(T, L, U) for L in Ls]
    cone.tempLU_vec = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.tempLU_vec2 = [[zeros(T, L, U) for _ in 1:(R - 1)] for L in Ls]
    cone.tempLRUR = [zeros(T, L * R, U * R) for L in Ls]
    cone.tempUU_vec = [zeros(T, U, U) for _ in eachindex(Ps)]
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

get_nu(cone::WSOSInterpEpiNormEucl) = 2 * sum(size(Pk, 2) for Pk in cone.Ps)

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

is_dual_feas(cone::WSOSInterpEpiNormEucl) = true

# block version of arrow(L * L') where L is half an arrow
function arrow_outer_prod!(vect::Vector{T}, mat::Matrix{T}, U::Int, R::Int) where T
    @views for u in 1:U
        vect[u] += sum(abs2, mat[:, u])
        idx = U + u
        for r in 2:R
            vect[idx] += 2 * dot(mat[:, idx], mat[:, u])
            vect[u] += sum(abs2, mat[:, idx])
            idx += U
        end
    end
    return vect
end

function update_grad(cone::WSOSInterpEpiNormEucl{T}) where T
    @assert cone.is_feas
    U = cone.U
    R = cone.R
    Λfact = cone.Λfact
    matfact = cone.matfact
    grad = cone.grad

    grad .= 0
    g_try = copy(grad)
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

        # L = size(Pk, 2)
        # point_views = cone.point_views
        # lambda_full = zeros(L * R, L * R)
        # lambda_full[1:L, 1:L] = Pk' * Diagonal(point_views[1]) * Pk
        # for r in 2:R
        #     lambda_full[block_idxs(L, r), block_idxs(L, r)] = lambda_full[1:L, 1:L]
        #     lambda_full[block_idxs(L, r), 1:L] = lambda_full[1:L, block_idxs(L, r)] = Pk' * Diagonal(point_views[r]) * Pk
        # end
        #
        #
        # lambda_inv_half = zeros(L * R, L * R)
        # lambda_inv_half[1:L, 1:L] = cholesky(inv(matfact[k])).U
        # for r in 2:R
        #     lambda_inv_half[1:L, block_idxs(L, r)] = -inv(lambda_inv_half[1:L, 1:L])' * inv(matfact[k]) * lambda_full[1:L, block_idxs(L, r)] * inv(Λfact[k])
        #     lambda_inv_half[block_idxs(L, r), block_idxs(L, r)] = cholesky(inv(Λfact[k])).U
        #     # lambda_inv_half[block_idxs(L, r), 1:L] = -inv(Λfact[k]) * lambda_full[block_idxs(L, r), 1:L] * inv(matfact[k]) * inv(lambda_inv_half[1:L, 1:L])'
        #     # lambda_inv_half[block_idxs(L, r), block_idxs(L, r)] = cholesky(inv(Λfact[k])).L
        # end
        # # lambda_inv_half = matfact[k].L \ lambda_inv_half
        # # @show R
        # @show (lambda_inv_half' * lambda_inv_half) ./ inv(lambda_full)
        #
        # lam_half_P = lambda_inv_half * kron(I(R), Pk')
        # @views for u in 1:U
        #     g_try[u] -= sum(abs2, lam_half_P[:, u])
        #     idx = U + u
        #     for r in 2:R
        #         g_try[idx] -= 2 * dot(lam_half_P[:, idx], lam_half_P[:, u])
        #         g_try[u] -= sum(abs2, lam_half_P[:, idx])
        #         idx += U
        #     end
        # end
        # g_try[1:U] += (R - 2) * diag(Pk * inv(Λfact[k]) * Pk')



    end
    # @show g_try ./ cone.grad

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
        UUk = cone.tempUU_vec[k]
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
                @. UUk = UU + UU'
                @. @views hess[1:U, 1:U] += UUk
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
            # NOTE for hess[idxs, idxs], UU and UUk are symmetric
            @. UU = PΛiPs[r, 1] * PΛiPs[r, 1]'
            @. UUk = PΛiPs[1, 1] * PΛiPs[r, r]
            @. @views hess[idxs, idxs] += UU + UUk
            for r2 in (r + 1):R
                @. UU = PΛiPs[r, 1] * PΛiPs[r2, 1]'
                @. UUk = PΛiPs[1, 1] * PΛiPs[r2, r]'
                idxs2 = block_idxs(U, r2)
                @. @views hess[idxs, idxs2] += UU + UUk
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
    c2 = zero(corr)
    # @show R

    @inbounds for pk in eachindex(cone.Ps)
        @views mul!(cone.tempLU[pk], cone.Λ11LiP[pk], Diagonal(primal_dir[1:U]))
        tempLU2 = mul!(cone.tempLU2[pk], cone.tempLU[pk], cone.PΛ11iP[pk])
        @views for u in 1:U
            corr[u] += sum(abs2, tempLU2[:, u])
        end
    end
    @. @views corr[1:U] *= 2 - R

    @inbounds for k in eachindex(cone.Ps)
        L = size(cone.Ps[k], 2)
        ΛLiP_edge = cone.ΛLiPs_edge[k]
        matLiP = cone.matLiP[k]
        PΛiP = cone.PΛiPs[k]
        Λ11LiP = cone.Λ11LiP[k]
        scaled_row = cone.tempLU_vec[k]
        scaled_col = cone.tempLU_vec2[k]

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

        corr_half = cone.tempLRUR[k]
        corr_half .= 0
        @views for c in 1:R
            mul!(corr_half[1:L, block_idxs(U, c)], scaled_pt, PΛiP[1:U, block_idxs(U, c)])
            for r in 2:R
                mul!(corr_half[1:L, block_idxs(U, c)], scaled_row[r - 1], PΛiP[block_idxs(U, r), block_idxs(U, c)], true, true)
                mul!(corr_half[block_idxs(L, r), block_idxs(U, c)], scaled_col[r - 1], PΛiP[1:U, block_idxs(U, c)])
                mul!(corr_half[block_idxs(L, r), block_idxs(U, c)], scaled_diag, PΛiP[block_idxs(U, r), block_idxs(U, c)], true, true)
            end
        end

        arrow_outer_prod!(corr, corr_half, U, R)




        Pk = cone.Ps[k]
        Δ = zeros(L * R, L * R)
        Δ[1:L, 1:L] = Pk' * Diagonal(primal_dir[1:U]) * Pk
        for r in 2:R
            Δ[block_idxs(L, r), block_idxs(L, r)] = Δ[1:L, 1:L]
            Δ[block_idxs(L, r), 1:L] = Δ[1:L, block_idxs(L, r)] = Pk' * Diagonal(primal_dir[block_idxs(U, r)]) * Pk
        end

        L = size(Pk, 2)
        point_views = cone.point_views
        lambda_full = zeros(L * R, L * R)
        lambda_full[1:L, 1:L] = Pk' * Diagonal(point_views[1]) * Pk
        for r in 2:R
            lambda_full[block_idxs(L, r), block_idxs(L, r)] = lambda_full[1:L, 1:L]
            lambda_full[block_idxs(L, r), 1:L] = lambda_full[1:L, block_idxs(L, r)] = Pk' * Diagonal(point_views[r]) * Pk
        end

        lambda_inv_half = zeros(L * R, L * R)
        lambda_inv_half[1:L, 1:L] = cholesky(inv(cone.matfact[k])).U
        for r in 2:R
            lambda_inv_half[1:L, block_idxs(L, r)] = -inv(lambda_inv_half[1:L, 1:L])' * inv(cone.matfact[k]) * lambda_full[1:L, block_idxs(L, r)] * inv(cone.Λfact[k])
            lambda_inv_half[block_idxs(L, r), block_idxs(L, r)] = cholesky(inv(cone.Λfact[k])).U
        end

        chalf = lambda_inv_half * Δ * lambda_inv_half' * lambda_inv_half * kron(I(R), Pk')

        @views for u in 1:U
            c2[u] += sum(abs2, chalf[:, u])
            idx = U + u
            for r in 2:R
                c2[idx] += 2 * dot(chalf[:, idx], chalf[:, u])
                c2[u] += sum(abs2, chalf[:, idx])
                idx += U
            end
        end
        L11 = lambda_full[1:L, 1:L]
        Δ11 = Δ[1:L, 1:L]
        c2[1:U] -= (R - 2) * diag(Pk * inv(L11) * Δ11 * inv(L11) * Δ11 * inv(L11) * Pk')

        At = lambda_inv_half[1:L, 1:L]
        Bt = lambda_inv_half[1:L, (L + 1):end]
        Ct = lambda_inv_half[(L + 1):2L, (L + 1):2L]
        X = Δ[1:L, 1:L]
        Y = Δ[(L + 1):end, 1:L]
        Z = Δ[(L + 1):2L, (L + 1):2L]
        CtP = Ct * Pk'

        # lambda_inv_half * Δ * lambda_inv_half' is arrow
        LP = zeros(L * R, U * R)
        LΔL = zeros(L * R, L * R)
        BYA = Bt * Y * At'
        LΔL[1:L, 1:L] = At * X * At' + BYA + BYA' + sum(Bt[:, block_idxs(L, r)] * Z * Bt[:, block_idxs(L, r)]' for r in 1:(R - 1))
        LP[1:L, 1:U] = lambda_inv_half[1:L, 1:L] * Pk'
        for r in 2:R
            # s * L^3
            LΔL[block_idxs(L, r), 1:L] = Ct * (Y[block_idxs(L, r - 1), :] * At' + Z * Bt[:, block_idxs(L, r - 1)]')
            LΔL[1:L, block_idxs(L, r)] = LΔL[block_idxs(L, r), 1:L]'
            LΔL[block_idxs(L, r), block_idxs(L, r)] = Ct * Z * Ct'
            #
            # s * L^2*U
            LP[1:L, block_idxs(U, r)] = lambda_inv_half[1:L, block_idxs(L, r)] * Pk'
            LP[block_idxs(L, r), block_idxs(U, r)] = CtP
        end

        # s^2 * L^2 * U
        chalf2 = zeros(L * R, L * R)
        # chalf[1:L, 1:U] = LΔL[1:L, :] * LP[:, 1:L]
        chalf2 = LΔL[:, 1:L] * LP[1:L, :]
        for r in 2:R
            chalf2[1:L, block_idxs(U, r)] = LΔL[1:L, 1:L] * LP[1:L, block_idxs(U, r)] + LΔL[1:L, block_idxs(L, r)] * LP[block_idxs(L, r), block_idxs(U, r)]
            chalf2[block_idxs(L, r), block_idxs(U, r)] = LΔL[block_idxs(L, r), block_idxs(L, r)] * LP[block_idxs(L, r), block_idxs(U, r)]
        end


        # @show LΔL
        # @show LΔL ≈ (lambda_inv_half * Δ * lambda_inv_half')
        # @show LΔL * LP ≈ chalf
        @show R
        @show norm(chalf2 - chalf, Inf), norm(chalf, Inf)

        # @show norm(LΔL - (lambda_inv_half * Δ * lambda_inv_half'))


        #
        # lambda_full = zeros(L * R, L * R)
        # lambda_full[1:L, 1:L] = Pk' * Diagonal(cone.point_views[1]) * Pk
        # for r in 2:R
        #     # s t^2 U
        #     lambda_full[block_idxs(L, r), block_idxs(L, r)] = lambda_full[1:L, 1:L]
        #     lambda_full[block_idxs(L, r), 1:L] = lambda_full[1:L, block_idxs(L, r)] = Pk' * Diagonal(cone.point_views[r]) * Pk
        # end
        #
        # ΛLiP_edge = cone.ΛLiPs_edge[k]
        # matLiP = cone.matLiP[k]
        # LiP = zeros(L * R, U * R)
        # Li = zeros(L * R, L * R)
        # LiP[1:L, 1:U] .= Λ11LiP
        #
        # Li[1:L, 1:L] .= inv(cone.Λfact[k].L)
        # for r in 2:R
        #     LiP[block_idxs(L, r), block_idxs(U, r)] .= matLiP
        #     LiP[block_idxs(L, r), 1:U] .= ΛLiP_edge[r - 1]
        #
        #     Li[block_idxs(L, r), block_idxs(L, r)] .= inv(cone.matfact[k].L)
        #     Li[block_idxs(L, r), 1:L] .= -cone.matfact[k].L \ (cone.ΛLi_Λ[k][r - 1]' * inv(cone.Λfact[k].L))
        # end
        #
        # lam_sqrt = cholesky(Hermitian(lambda_full)).L
        #
        # # @show (Li' * Li) ./ inv(lambda_full)
        #  # @show (kron(I(R), Pk) * Li' * Li * kron(I(R), Pk')) ./ (kron(I(R), Pk) * inv(lambda_full) * kron(I(R), Pk'))
        # # @show Li * kron(I(R), Pk') ≈ LiP
        # @show (LiP' * LiP) ./ (kron(I(R), Pk) * inv(lambda_full) * kron(I(R), Pk'))
        #
        # # chalf = Li * Δ * Li' * LiP
        # chalf = inv(lam_sqrt) * Δ * inv(lambda_full) * kron(I(R), Pk')
        # # arrow_outer_prod!(c2, chalf, U, R)
        #
        # L11 = lambda_full[1:L, 1:L]
        # Δ11 = Δ[1:L, 1:L]
        #
        # @views for u in 1:U
        #     c2[u] += sum(abs2, chalf[:, u])
        #     idx = U + u
        #     for r in 2:R
        #         c2[idx] += 2 * dot(chalf[:, idx], chalf[:, u])
        #         c2[u] += sum(abs2, chalf[:, idx])
        #         idx += U
        #     end
        # end
        #
        #
        # c2[1:U] -= (R - 2) * diag(Pk * inv(L11) * Δ11 * inv(L11) * Δ11 * inv(L11) * Pk')
        #
        # # bigthing = kron(I(R), Pk) * inv(lambda_full) * Δ * inv(lambda_full) * Δ * inv(lambda_full) * kron(I(R), Pk')
        # # function lamadj(vec, mat)
        # #     idx = 1
        # #     vec[1:U] += sum(diag(mat[block_idxs(U, r), block_idxs(U, r)]) for r in 1:R)
        # #     for r in 2:R
        # #         vec[block_idxs(U, r)] += 2 * diag(mat[block_idxs(U, r), 1:U])
        # #     end
        # #     return vec
        # # end
        # # lamadj(c2, bigthing)
        # # L11 = lambda_full[1:L, 1:L]
        # # Δ11 = Δ[1:L, 1:L]
        # # c2[1:U] -= (R - 2) * diag(Pk * inv(L11) * Δ11 * inv(L11) * Δ11 * inv(L11) * Pk')
        #
        # # @show chalf

    end
    @show c2 ./ corr

    return corr
end
