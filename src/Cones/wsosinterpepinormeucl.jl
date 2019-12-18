#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

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
    use_dual::Bool
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    mat::Vector{Matrix{T}}
    matfact::Vector

    Λi_Λ::Vector{Vector{Matrix{T}}}
    Λ11::Vector{Matrix{T}}
    tmpLL::Vector{Matrix{T}}
    tmpLU::Vector{Matrix{T}}
    tmpLU2::Vector{Matrix{T}}
    tmpUU_vec::Vector{Matrix{T}} # reused in update_hess
    tmpUU::Matrix{T}
    PΛiPs::Vector{Vector{Vector{Matrix{T}}}}
    lambdafact::Vector
    point_views

    function WSOSInterpEpiNormEucl{T}(
        R::Int,
        U::Int,
        Ps::Vector{Matrix{T}},
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        for Pj in Ps
            @assert size(Pj, 1) == U
        end
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = U * R
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

WSOSInterpEpiNormEucl{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpEpiNormEucl{T}(R, U, Ps, false)

function setup_data(cone::WSOSInterpEpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.mat = [similar(cone.grad, size(Psk, 2), size(Psk, 2)) for Psk in Ps]
    cone.matfact = Vector{Any}(undef, length(Ps))
    cone.Λi_Λ = [Vector{Matrix{T}}(undef, R - 1) for Psk in Ps]
    @inbounds for k in eachindex(Ps), r in 1:(R - 1)
        cone.Λi_Λ[k][r] = similar(cone.grad, size(Ps[k], 2), size(Ps[k], 2))
    end
    cone.Λ11 = [similar(cone.grad, size(Psk, 2), size(Psk, 2)) for Psk in Ps]
    cone.tmpLL = [similar(cone.grad, size(Psk, 2), size(Psk, 2)) for Psk in Ps]
    cone.tmpLU = [similar(cone.grad, size(Psk, 2), U) for Psk in Ps]
    cone.tmpLU2 = [similar(cone.grad, size(Psk, 2), U) for Psk in Ps]
    cone.tmpUU_vec = [similar(cone.grad, U, U) for _ in eachindex(Ps)]
    cone.tmpUU = similar(cone.grad, U, U)
    cone.PΛiPs = [Vector{Vector{Matrix{T}}}(undef, R) for Psk in Ps]
    @inbounds for k in eachindex(Ps), r1 in 1:R
        cone.PΛiPs[k][r1] = Vector{Matrix{T}}(undef, r1)
        for r2 in 1:r1
            cone.PΛiPs[k][r1][r2] = similar(cone.grad, U, U)
        end
    end
    cone.lambdafact = Vector{Any}(undef, length(Ps))
    cone.point_views = [view(cone.point, block_idxs(U, i)) for i in 1:R]
    return
end

get_nu(cone::WSOSInterpEpiNormEucl) = 2 * sum(size(Psk, 2) for Psk in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSInterpEpiNormEucl)
    arr[1:cone.U] .= 1
    arr[(cone.U + 1):end] .= 0
    return arr
end

function update_feas(cone::WSOSInterpEpiNormEucl)
    @assert !cone.feas_updated
    lambdafact = cone.lambdafact
    matfact = cone.matfact
    point_views = cone.point_views

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ps)
        Psk = cone.Ps[k]
        Λ11j = cone.Λ11[k]
        LLk = cone.tmpLL[k]
        LUk = cone.tmpLU[k]
        Λi_Λ = cone.Λi_Λ[k]
        mat = cone.mat[k]

        # first lambda
        @. LUk = Psk' * point_views[1]'
        mul!(Λ11j, LUk, Psk)
        copyto!(mat, Λ11j)
        lambdafact[k] = cholesky!(Symmetric(Λ11j, :U), check = false)
        if !isposdef(lambdafact[k])
            cone.is_feas = false
            break
        end

        # subtract others
        uo = cone.U + 1
        @inbounds for r in 2:cone.R
            @. LUk = Psk' * point_views[r]'
            mul!(LLk, LUk, Psk)

            # not using lambdafact.L \ lambda with an syrk because storing lambdafact \ lambda is useful later
            copyto!(Λi_Λ[r - 1], LLk)
            ldiv!(lambdafact[k], Λi_Λ[r - 1])
            mul!(mat, LLk, Λi_Λ[r - 1], -1, true)
            uo += cone.U
        end

        matfact[k] = cholesky!(Symmetric(mat, :U), check = false)
        if !isposdef(matfact[k])
            cone.is_feas = false
            break
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSInterpEpiNormEucl{T}) where {T}
    @assert cone.is_feas
    lambdafact = cone.lambdafact
    matfact = cone.matfact

    cone.grad .= 0
    @inbounds for k in eachindex(cone.Ps)
        Psk = cone.Ps[k]
        LUk = cone.tmpLU[k]
        LUk2 = cone.tmpLU2[k]
        UUk = cone.tmpUU_vec[k]
        PΛiPs = cone.PΛiPs[k]
        Λi_Λ = cone.Λi_Λ[k]

        # P * inv(Λ_11) * P' for (1, 1) hessian block and adding to PΛiPs[r][r]
        copyto!(LUk, Psk')
        ldiv!(cone.lambdafact[k].L, LUk) # TODO may be more efficient to do ldiv(fact.U', B) than ldiv(fact.L, B) here and elsewhere since the factorizations are of symmetric :U matrices
        mul!(UUk, LUk', LUk)

        # prep PΛiPs
        # block-(1,1) is P * inv(mat) * P'
        copyto!(LUk, Psk')
        ldiv!(matfact[k].L, LUk)
        mul!(PΛiPs[1][1], LUk', LUk)
        # get all the PΛiPs that are in row one or on the diagonal
        @inbounds for r in 2:cone.R
            copyto!(LUk, Psk')
            ldiv!(matfact[k], LUk)
            mul!(LUk2, Λi_Λ[r - 1], LUk)
            mul!(PΛiPs[r][1], Psk, LUk2, -1, false)
            # PΛiPs[r][r] .= Symmetric(Psk * Λi_Λ[r - 1] * (matfact[k] \ (Λi_Λ[r - 1]' * Psk')), :U)
            mul!(LUk, Λi_Λ[r - 1]', Psk')
            ldiv!(matfact[k].L, LUk)
            mul!(PΛiPs[r][r], LUk', LUk)
            @. PΛiPs[r][r] += UUk
        end

        # (1, 1)-block
        # gradient is diag of sum(-PΛiPs[i][i] for i in 1:R) + (R - 1) * Lambda_11 - Lambda_11
        @inbounds for i in 1:cone.U
            cone.grad[i] += UUk[i, i] * (cone.R - 2)
            @inbounds for r in 1:cone.R
                cone.grad[i] -= PΛiPs[r][r][i, i]
            end
        end
        idx = cone.U + 1
        @inbounds for r in 2:cone.R, i in 1:cone.U
            cone.grad[idx] -= 2 * PΛiPs[r][1][i, i]
            idx += 1
        end
    end # j

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpEpiNormEucl)
    @assert cone.grad_updated
    hess = cone.hess.data
    UU = cone.tmpUU
    matfact = cone.matfact

    hess .= 0
    @inbounds for k in eachindex(cone.Ps)
        Psk = cone.Ps[k]
        PΛiPs = cone.PΛiPs[k]
        Λi_Λ = cone.Λi_Λ[k]
        UUk = cone.tmpUU_vec[k]
        LUk = cone.tmpLU[k]
        LUk2 = cone.tmpLU2[k]

        # get the PΛiPs not calculated in update_grad
        @inbounds for r in 2:cone.R, r2 in 2:(r - 1)
            mul!(LUk, Λi_Λ[r2 - 1]', Psk')
            ldiv!(matfact[k], LUk)
            mul!(LUk2, Λi_Λ[r - 1], LUk)
            mul!(PΛiPs[r][r2], Psk, LUk2)
        end

        @inbounds for i in 1:cone.U, k in 1:i
            hess[k, i] -= abs2(UUk[k, i]) * (cone.R - 2)
        end

        @. hess[1:cone.U, 1:cone.U] += abs2(PΛiPs[1][1])
        @inbounds for r in 2:cone.R
            idxs = block_idxs(cone.U, r)
            @inbounds for s in 1:(r - 1)
                # block (1,1)
                @. UU = abs2(PΛiPs[r][s])
                # safe to ovewrite UUk now
                @. UUk = UU + UU'
                @. hess[1:cone.U, 1:cone.U] += UUk
                # blocks (1,r)
                @. hess[1:cone.U, idxs] += PΛiPs[s][1] * PΛiPs[r][s]'
            end
            # block (1,1)
            @. hess[1:cone.U, 1:cone.U] += abs2(PΛiPs[r][r])
            # blocks (1,r)
            @. hess[1:cone.U, idxs] += PΛiPs[r][1] * PΛiPs[r][r]
            # blocks (1,r)
            @inbounds for s in (r + 1):cone.R
                @. hess[1:cone.U, idxs] += PΛiPs[s][1] * PΛiPs[s][r]
            end

            # blocks (r, r2)
            # NOTE for hess[idxs, idxs], UU and UUk are symmetric
            @. UU = PΛiPs[r][1] * PΛiPs[r][1]'
            @. UUk = PΛiPs[1][1] * PΛiPs[r][r]
            @. hess[idxs, idxs] += UU + UUk
            @inbounds for r2 in (r + 1):cone.R
                @. UU = PΛiPs[r][1] * PΛiPs[r2][1]'
                @. UUk = PΛiPs[1][1] * PΛiPs[r2][r]'
                idxs2 = block_idxs(cone.U, r2)
                @. hess[idxs, idxs2] += UU + UUk
            end
        end
    end
    @. hess[:, (cone.U + 1):cone.dim] *= 2

    cone.hess_updated = true
    return cone.hess
end
