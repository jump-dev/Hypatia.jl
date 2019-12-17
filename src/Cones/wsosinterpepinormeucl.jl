#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial epinormeucl (AKA second-order cone) parametrized by interpolation matrices Ps
certifies that u(x)^2 <= sum(w_i(x)^2) for all x in the domain described by input Ps
u(x), w_1(x), ...,  w_R(x) are polynomials with U coefficients

dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
-logdet(schur(Lambda)) - logdet(Lambda_11)
if schur(M) = A - B * inv(D) * C
logdet(schur) = logdet(M) - logdet(D) = logdet(Lambda) - (R - 1) * logdet(Lambda_11) since our D is an (R - 1)x(R - 1) block diagonal matrix
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
    tmp3::Matrix{T}

    Λi_Λ::Vector{Vector{Matrix{T}}}
    Λ11::Vector{Matrix{T}}
    tmpLL::Vector{Matrix{T}}
    tmpLU::Vector{Matrix{T}}
    tmpUU::Vector{Matrix{T}}
    PΛiPs::Vector{Vector{Vector{Matrix{T}}}}
    lambdafact::Vector

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
    cone.mat = [similar(cone.grad, size(Psj, 2), size(Psj, 2)) for Psj in Ps]
    cone.matfact = Vector{Any}(undef, length(Ps))
    cone.tmp3 = similar(cone.grad, U, U)
    cone.Λi_Λ = [Vector{Matrix{T}}(undef, R - 1) for Psj in Ps]
    @inbounds for j in eachindex(Ps), r in 1:(R - 1)
        cone.Λi_Λ[j][r] = similar(cone.grad, size(Ps[j], 2), size(Ps[j], 2))
    end
    cone.Λ11 = [similar(cone.grad, size(Psj, 2), size(Psj, 2)) for Psj in Ps]
    cone.tmpLL = [similar(cone.grad, size(Psj, 2), size(Psj, 2)) for Psj in Ps]
    cone.tmpLU = [similar(cone.grad, size(Psj, 2), U) for Psj in Ps]
    cone.tmpUU = [similar(cone.grad, U, U) for _ in eachindex(Ps)]
    cone.PΛiPs = [Vector{Vector{Matrix{T}}}(undef, R) for Psj in Ps]
    @inbounds for j in eachindex(Ps), r1 in 1:R
        cone.PΛiPs[j][r1] = Vector{Matrix{T}}(undef, r1)
        for r2 in 1:r1
            cone.PΛiPs[j][r1][r2] = similar(cone.grad, U, U)
        end
    end
    cone.lambdafact = Vector{Any}(undef, length(Ps))
    return
end

get_nu(cone::WSOSInterpEpiNormEucl) = 2 * sum(size(Psj, 2) for Psj in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSInterpEpiNormEucl)
    arr[1:cone.U] .= 1
    arr[(cone.U + 1):end] .= 0
    return arr
end

function update_feas(cone::WSOSInterpEpiNormEucl)
    @assert !cone.feas_updated

    cone.is_feas = true
    @inbounds for j in eachindex(cone.Ps)
        Psj = cone.Ps[j]
        Λ11j = cone.Λ11[j]
        LLj = cone.tmpLL[j]
        LUj = cone.tmpLU[j]
        Λi_Λ = cone.Λi_Λ[j]
        mat = cone.mat[j]
        lambdafact = cone.lambdafact
        matfact = cone.matfact

        # first lambda
        point_pq = cone.point[1:cone.U]
        @. LUj = Psj' * point_pq'
        mul!(Λ11j, LUj, Psj)
        copyto!(mat, Λ11j)
        lambdafact[j] = cholesky!(Symmetric(Λ11j, :U), check = false)
        if !isposdef(lambdafact[j])
            cone.is_feas = false
            break
        end

        # minus others
        uo = cone.U + 1
        @inbounds for r in 2:cone.R
            point_pq = cone.point[uo:(uo + cone.U - 1)] # TODO prealloc
            @. LUj = Psj' * point_pq'
            mul!(LLj, LUj, Psj)

            # avoiding lambdafact.L \ lambda because lambdafact \ lambda is useful later
            Λi_Λ[r - 1] .= lambdafact[j] \ LLj
            mat -= LLj * Λi_Λ[r - 1]
            uo += cone.U
        end

        matfact[j] = cholesky!(Symmetric(mat, :U), check = false)
        @inbounds if !isposdef(matfact[j])
            cone.is_feas = false
            break
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSInterpEpiNormEucl{T}) where {T}
    @assert cone.is_feas

    cone.grad .= 0
    @inbounds for j in eachindex(cone.Ps)
        Psj = cone.Ps[j]
        LUj = cone.tmpLU[j]
        UUj = cone.tmpUU[j]
        PΛiPs = cone.PΛiPs[j]
        Λi_Λ = cone.Λi_Λ[j]
        lambdafact = cone.lambdafact
        matfact = cone.matfact

        # P * inv(Λ_11) * P' for (1, 1) hessian block and adding to PΛiPs[r][r]
        copyto!(LUj, Psj')
        ldiv!(LowerTriangular(cone.lambdafact[j].L), LUj)
        mul!(UUj, LUj', LUj)

        # prep PΛiPs
        # block-(1,1) is P * inv(mat) * P'
        copyto!(LUj, Psj')
        ldiv!(LowerTriangular(matfact[j].L), LUj)
        mul!(PΛiPs[1][1], LUj', LUj)
        # get all the PΛiPs that are in row one or on the diagonal
        @inbounds for r in 2:cone.R
            PΛiPs[r][1] = -Psj * Λi_Λ[r - 1] * (matfact[j] \ Psj')
            # PΛiPs[r][r] .= Symmetric(Psj * Λi_Λ[r - 1] * (matfact[j] \ (Λi_Λ[r - 1]' * Psj')), :U)
            mul!(LUj, Λi_Λ[r - 1]', Psj')
            ldiv!(matfact[j].L, LUj)
            mul!(PΛiPs[r][r], LUj', LUj)
            PΛiPs[r][r] .+= UUj
        end

        # (1, 1)-block
        # gradient is diag of sum(-PΛiPs[i][i] for i in 1:R) + (R - 1) * Lambda_11 - Lambda_11
        # TODO use UUj
        @inbounds for i in 1:cone.U
            cone.grad[i] += UUj[i, i] * (cone.R - 2)
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

    hess .= 0
    @inbounds for j in eachindex(cone.Ps)
        Psj = cone.Ps[j]
        tmp3 = cone.tmp3
        PΛiPs = cone.PΛiPs[j]
        Λi_Λ = cone.Λi_Λ[j]
        matfact = cone.matfact
        UUj = cone.tmpUU[j]

        # get the PΛiPs not calculated in update_grad
        @inbounds for r in 2:cone.R, r2 in 2:(r - 1)
            PΛiPs[r][r2] .= Psj * Λi_Λ[r - 1] * (matfact[j] \ (Λi_Λ[r2 - 1]' * Psj'))
        end

        @inbounds for i in 1:cone.U, k in 1:i
            hess[k, i] -= abs2(UUj[k, i]) * (cone.R - 2)
        end

        @. hess[1:cone.U, 1:cone.U] += PΛiPs[1][1]^2
        @inbounds for r in 2:cone.R
            idxs2 = ((r - 1) * cone.U + 1):(r * cone.U)
            @inbounds for s in 1:(r - 1)
                # block (1,1)
                tmp3 .= PΛiPs[r][s].^2
                hess[1:cone.U, 1:cone.U] .+= Symmetric(tmp3 + tmp3', :U)
                # blocks (1,r)
                @. hess[1:cone.U, idxs2] += 2 * PΛiPs[s][1] * PΛiPs[r][s]'
            end
            # block (1,1)
            @. hess[1:cone.U, 1:cone.U] += PΛiPs[r][r]^2
            # blocks (1,r)
            @. hess[1:cone.U, idxs2] += 2 * PΛiPs[r][1] * PΛiPs[r][r]
            # blocks (1,r)
            @inbounds for s in (r + 1):cone.R
                @. hess[1:cone.U, idxs2] += 2 * PΛiPs[s][1] * PΛiPs[s][r]
            end

            # blocks (r, r2)
            idxs = ((r - 1) * cone.U + 1):(r * cone.U)
            hess[idxs, idxs2] .+= 2 * Symmetric(Symmetric(PΛiPs[1][1], :U) .* Symmetric(PΛiPs[r][r], :U) + PΛiPs[r][1] .* PΛiPs[r][1]', :U)
            @inbounds for r2 in (r + 1):cone.R
                idxs2 = ((r2 - 1) * cone.U + 1):(r2 * cone.U)
                hess[idxs, idxs2] .+= 2 * (PΛiPs[1][1] .* PΛiPs[r2][r]' + PΛiPs[r][1] .* PΛiPs[r2][1]')
            end
        end
    end

    cone.hess_updated = true
    return cone.hess
end
