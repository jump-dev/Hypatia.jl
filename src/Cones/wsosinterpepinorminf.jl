#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

=#
using ForwardDiff

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
    tmpLL::Vector{Matrix{T}}
    tmpLU::Vector{Matrix{T}}
    tmpLU2::Vector{Matrix{T}}
    tmpUU_vec::Vector{Matrix{T}} # reused in update_hess
    tmpUU::Matrix{T}
    PΛiPs::Vector{Vector{Vector{Matrix{T}}}}
    lambdafact::Vector
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

        # soc-based
        # function barrier(point)
        #      bar = zero(eltype(point))
        #      for P in cone.Ps
        #          lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)
        #          fact_1 = cholesky(lambda_1)
        #          for i in 2:R
        #              lambda_i = Symmetric(P' * Diagonal(point[block_idxs(U, i)]) * P)
        #              LL = fact_1.L \ lambda_i
        #              bar -= logdet(lambda_1 - LL' * LL)
        #              # bar -= logdet(lambda_1 - lambda_i * (fact_1 \ lambda_i))
        #          end
        #          bar -= logdet(fact_1)
        #      end
        #      return bar
        # end

        # orthant-based
        function barrier(point)
             bar = zero(eltype(point))
             for P in cone.Ps
                 lambda_1 = Hermitian(P' * Diagonal(point[1:U]) * P)
                 for i in 2:R
                     lambda_i = Hermitian(P' * Diagonal(point[block_idxs(U, i)]) * P)
                     bar -= logdet(lambda_1 - lambda_i) + logdet(lambda_1 + lambda_i)
                 end
                 bar += logdet(cholesky(lambda_1)) * (R - 2)
             end
             return bar
        end

        cone.barrier = barrier

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
    cone.matfact = [[cholesky(hcat([one(T)])) for _ in 1:R] for _ in cone.Ps]
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

get_nu(cone::WSOSInterpEpiNormInf) = cone.R * sum(size(Pk, 2) for Pk in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSInterpEpiNormInf)
    arr[1:cone.U] .= 1
    arr[(cone.U + 1):end] .= 0
    return arr
end

# function update_feas(cone::WSOSInterpEpiNormInf)
#     @assert !cone.feas_updated
#     U = cone.U
#     point = cone.point
#
#     # cone.is_feas = true
#     # @inbounds for k in eachindex(cone.Ps)
#     #     P = cone.Ps[k]
#     #     lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)
#     #     fact_1 = cholesky(lambda_1, check = false)
#     #     if isposdef(fact_1)
#     #         for i in 2:cone.R
#     #             lambda_i = Symmetric(P' * Diagonal(point[block_idxs(U, i)]) * P)
#     #             LL = fact_1.L \ lambda_i
#     #             if !isposdef(lambda_1 - LL' * LL)
#     #                 cone.is_feas = false
#     #                 break
#     #             end
#     #         end
#     #     else
#     #         cone.is_feas = false
#     #         break
#     #     end
#     # end
#
#     cone.is_feas = true
#     @inbounds for k in eachindex(cone.Ps)
#         P = cone.Ps[k]
#         lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)
#         fact_1 = cholesky(lambda_1, check = false)
#         if isposdef(fact_1)
#             for i in 2:cone.R
#                 lambda_i = Symmetric(P' * Diagonal(point[block_idxs(U, i)]) * P)
#                 if !isposdef(lambda_1 - lambda_i) || !isposdef(lambda_1 + lambda_i)
#                     cone.is_feas = false
#                     break
#                 end
#             end
#         else
#             cone.is_feas = false
#             break
#         end
#     end
#
#     cone.feas_updated = true
#     return cone.is_feas
# end

function update_feas(cone::WSOSInterpEpiNormInf)
    @assert !cone.feas_updated
    U = cone.U
    R = cone.R
    lambdafact = cone.lambdafact
    # mats = [[Matrix{Any}(undef, size(P, 2), size(P, 2)) for _ in 1:R] for P in cone.Ps]
    # facts = [Vector{Any}(undef, R) for _ in cone.Ps]
    point_views = cone.point_views

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ps)
        Psk = cone.Ps[k]
        Λ11j = cone.Λ11[k]
        LLk = cone.tmpLL[k]
        LUk = cone.tmpLU[k]
        Λi_Λ = cone.Λi_Λ[k]
        matsk = cone.mats[k]
        factk = cone.matfact[k]

        # first lambda
        @. LUk = Psk' * point_views[1]'
        mul!(matsk[1], LUk, Psk)
        copyto!(Λ11j, matsk[1])
        lambdafact[k] = cholesky!(Symmetric(matsk[1], :U), check = false)
        if !isposdef(lambdafact[k])
            cone.is_feas = false
            break
        end

        uo = U + 1
        @inbounds for r in 2:cone.R
            matr = matsk[r - 1]
            factr = factk[r - 1]
            @. LUk = Psk' * point_views[r]'
            mul!(LLk, LUk, Psk)

            # not using lambdafact.L \ lambda with an syrk because storing lambdafact \ lambda is useful later
            ldiv!(Λi_Λ[r - 1], lambdafact[k], LLk)
            matr = Symmetric(Λ11j, :U) - LLk * Λi_Λ[r - 1]
            # ldiv!(lambdafact[k].L, LLk)
            # mat = Λ11j - LLk' * LLk
            factk[r - 1] = cholesky(Symmetric(matr))
            if !isposdef(factk[r - 1])
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
    # fd_grad = ForwardDiff.gradient(cone.barrier, cone.point)

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
        PΛiPs = cone.PΛiPs[k]
        Λi_Λ = cone.Λi_Λ[k]

        # P * inv(Λ_11) * P' for (1, 1) hessian block and adding to PΛiPs[r][r]
        ldiv!(LUk, cone.lambdafact[k].L, Psk') # TODO may be more efficient to do ldiv(fact.U', B) than ldiv(fact.L, B) here and elsewhere since the factorizations are of symmetric :U matrices
        mul!(UUk, LUk', LUk)

        # prep PΛiPs
        # get all the PΛiPs that are in row one or on the diagonal
        @inbounds for r in 2:R
            # block-(1,1) is P * inv(mat) * P'
            ldiv!(LUk, matfact[k][r - 1].L, Psk')
            mul!(PΛiPs[r][r], LUk', LUk)
            # block (1,2)
            ldiv!(LUk, matfact[k][r - 1], Psk')
            mul!(LUk2, Λi_Λ[r - 1], LUk)
            mul!(PΛiPs[r][1], Psk, LUk2, -1, false)
            # PΛiPs[r][r] .= Symmetric(Psk * Λi_Λ[r - 1] * (matfact[k] \ (Λi_Λ[r - 1]' * Psk')), :U)
            # mul!(LUk, Λi_Λ[r - 1]', Psk')
            # ldiv!(matfact[k][r].L, LUk)
            # mul!(PΛiPs[r][r], LUk', LUk)
            # @. PΛiPs[r][r] += UUk
        end

        # (1, 1)-block
        # gradient is diag of sum(-PΛiPs[i][i] for i in 1:R) + (R - 1) * Lambda_11 - Lambda_11
        @inbounds for i in 1:U
            cone.grad[i] += UUk[i, i] * R2
            @inbounds for r in 2:R
                cone.grad[i] -= PΛiPs[r][r][i, i] * 2
            end
        end
        idx = U + 1
        @inbounds for r in 2:R, i in 1:U
            cone.grad[idx] -= 2 * PΛiPs[r][1][i, i]
            idx += 1
        end
    end # j
    # @show cone.grad ./ fd_grad

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpEpiNormInf)
    # fd_hess = ForwardDiff.hessian(cone.barrier, cone.point)

    @assert cone.grad_updated
    U = cone.U
    R = cone.R
    R2 = R - 2
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
        # @inbounds for r in 2:R, r2 in 2:(r - 1)
        #     mul!(LUk, Λi_Λ[r2 - 1]', Psk')
        #     ldiv!(matfact[k], LUk)
        #     mul!(LUk2, Λi_Λ[r - 1], LUk)
        #     mul!(PΛiPs[r][r2], Psk, LUk2)
        # end

        @inbounds for i in 1:U, k in 1:i
            hess[k, i] -= abs2(UUk[k, i]) * R2
        end

        @inbounds for r in 2:R
            @. hess[1:U, 1:U] += abs2(PΛiPs[r][r])
            idxs = block_idxs(U, r)
            # @inbounds for s in 1:(r - 1)
                # block (1,1)
                @. UU = abs2(PΛiPs[r][1])
                # safe to ovewrite UUk now
                @. UUk = UU + UU'
                @. hess[1:U, 1:U] += UUk
                # blocks (1,r)
                @. hess[1:U, idxs] += PΛiPs[r][r] * PΛiPs[r][1]'
            # end
            # block (1,1)
            @. hess[1:U, 1:U] += abs2(PΛiPs[r][r])
            # blocks (1,r)
            @. hess[1:U, idxs] += PΛiPs[r][1] * PΛiPs[r][r] + PΛiPs[r][r] * PΛiPs[r][1]

            # blocks (r, r2)
            # NOTE for hess[idxs, idxs], UU and UUk are symmetric
            @. UU = PΛiPs[r][1] * PΛiPs[r][1]'
            @. UUk = PΛiPs[r][r] * PΛiPs[r][r]
            @. hess[idxs, idxs] += UU + UUk
        end
    end
    @. hess[:, (U + 1):cone.dim] *= 2

    cone.hess_updated = true
    return cone.hess
end

use_correction(::WSOSInterpEpiNormInf) = false
