#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial positive semidefinite cone parametrized by interpolation matrices Ps
certifies that a polynomial valued R x R matrix is in the positive semidefinite cone for all x in the domain defined by Ps

dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
and "Semidefinite Characterization of Sum-of-Squares Cones in Algebras" by D. Papp and F. Alizadeh
=#

setprecision(128)
out_T = BigFloat
# out_T = Float64

mutable struct WSOSInterpPosSemidefTri{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    point::AbstractVector{T}
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
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    rt2::T
    rt2i::T
    tmpU::Vector{T}
    tmpLRLR::Vector{Symmetric{T, Matrix{T}}}
    ΛFL::Vector
    ΛFLP::Vector{Matrix{T}}
    tmpLU::Vector{Matrix{T}}
    PlambdaP::Vector{Matrix{T}}

    function WSOSInterpPosSemidefTri{T}(
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
        cone.dim = U * svec_length(R)
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_data(cone::WSOSInterpPosSemidefTri{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    cone.point = zeros(T, dim)
    cone.grad = similar(cone.point)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.rt2 = sqrt(T(2))
    cone.rt2i = inv(cone.rt2)
    cone.tmpU = Vector{T}(undef, U)
    cone.tmpLRLR = [Symmetric(zeros(T, size(Pk, 2) * R, size(Pk, 2) * R), :L) for Pk in Ps]
    cone.tmpLU = [Matrix{T}(undef, size(Pk, 2), U) for Pk in Ps]
    cone.ΛFL = Vector{Any}(undef, length(Ps))
    cone.ΛFLP = [Matrix{T}(undef, R * size(Pk, 2), R * U) for Pk in Ps]
    cone.PlambdaP = [zeros(T, R * U, R * U) for _ in eachindex(Ps)]
    return
end

get_nu(cone::WSOSInterpPosSemidefTri) = cone.R * sum(size(Pk, 2) for Pk in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSInterpPosSemidefTri)
    arr .= 0
    block = 1
    @inbounds for i in 1:cone.R
        arr[block_idxs(cone.U, block)] .= 1
        block += i + 1
    end
    return arr
end

function update_feas(cone::WSOSInterpPosSemidefTri)
    @assert !cone.feas_updated

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ps)
        Pk = out_T.(cone.Ps[k])
        LU = out_T.(cone.tmpLU[k])
        L = size(Pk, 2)
        Λ = Symmetric(zeros(out_T, L * cone.R, L * cone.R), :L)

        for p in 1:cone.R, q in 1:p
            @views tmpU = out_T.(cone.point[block_idxs(cone.U, svec_idx(p, q))])
            if p != q
                tmpU .*= inv(sqrt(out_T(2)))
            end
            mul!(LU, Pk', Diagonal(tmpU)) # TODO check efficiency
            # @show typeof(LU)
            @views mul!(Λ.data[block_idxs(L, p), block_idxs(L, q)], LU, Pk)
        end

        ΛFLk = cone.ΛFL[k] = cholesky!(Λ, check = false)
        if !isposdef(ΛFLk)
            cone.is_feas = false
            break
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSInterpPosSemidefTri)
    @assert is_feas(cone)
    U = cone.U
    R = cone.R
    cone.grad .= 0

    # update PlambdaP
    for k in eachindex(cone.PlambdaP)
        Pk = out_T.(cone.Ps[k])

        L = size(cone.Ps[k], 2)
        ΛFL = cone.ΛFL[k].L
        # @show typeof(ΛFL)
        ΛFLP = out_T.(cone.ΛFLP[k])

        # given cholesky L factor ΛFL, get ΛFLP = ΛFL * kron(I, P')
        @inbounds for p in 1:R
            block_U_p_idxs = block_idxs(U, p)
            block_L_p_idxs = block_idxs(L, p)
            @views ΛFLP_pp = ΛFLP[block_L_p_idxs, block_U_p_idxs]
            # ΛFLP_pp = ΛFL_pp \ P'
            @views ldiv!(ΛFLP_pp, LowerTriangular(ΛFL[block_L_p_idxs, block_L_p_idxs]), Pk')
            # @show typeof(ΛFLP_pp)
            # to get off-diagonals in ΛFLP, subtract known blocks aggregated in ΛFLP_qp
            @inbounds for q in (p + 1):R
                block_L_q_idxs = block_idxs(L, q)
                @views ΛFLP_qp = ΛFLP[block_L_q_idxs, block_U_p_idxs]
                ΛFLP_qp .= 0
                @inbounds for p2 in p:(q - 1)
                    block_L_p2_idxs = block_idxs(L, p2)
                    @views mul!(ΛFLP_qp, ΛFL[block_L_q_idxs, block_L_p2_idxs], ΛFLP[block_L_p2_idxs, block_U_p_idxs], -1, 1)
                end
                @views ldiv!(LowerTriangular(ΛFL[block_L_q_idxs, block_L_q_idxs]), ΛFLP_qp)
            end
        end

        # PlambdaP = ΛFLP' * ΛFLP
        PlambdaPk = out_T.(cone.PlambdaP[k])
        for p in 1:R, q in p:R
            block_p_idxs = block_idxs(U, p)
            block_q_idxs = block_idxs(U, q)
            # since ΛFLP is block lower triangular rows only from max(p,q) start making a nonzero contribution to the product
            row_range = ((q - 1) * L + 1):(L * R)
            @inbounds @views mul!(PlambdaPk[block_p_idxs, block_q_idxs], ΛFLP[row_range, block_p_idxs]', ΛFLP[row_range, block_q_idxs])
        end

        # update gradient
        for p in 1:cone.R, q in 1:p
            scal = (p == q ? -1 : -cone.rt2)
            idx = (svec_idx(p, q) - 1) * U
            block_p = (p - 1) * U
            block_q = (q - 1) * U
            for i in 1:U
                block_p_i = block_p + i
                block_q_i = block_q + i
                @inbounds cone.grad[idx + i] += scal * PlambdaPk[block_q_i, block_p_i]
            end
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpPosSemidefTri)
    @assert cone.grad_updated
    R = cone.R
    U = cone.U
    H = cone.hess.data

    H .= 0
    for k in eachindex(cone.Ps)
        Pk = out_T.(cone.Ps[k])

        L = size(cone.Ps[k], 2)
        ΛFL = cone.ΛFL[k].L
        ΛFLP = out_T.(cone.ΛFLP[k])

        # given cholesky L factor ΛFL, get ΛFLP = ΛFL * kron(I, P')
        @inbounds for p in 1:R
            block_U_p_idxs = block_idxs(U, p)
            block_L_p_idxs = block_idxs(L, p)
            @views ΛFLP_pp = ΛFLP[block_L_p_idxs, block_U_p_idxs]
            # ΛFLP_pp = ΛFL_pp \ P'
            @views ldiv!(ΛFLP_pp, LowerTriangular(ΛFL[block_L_p_idxs, block_L_p_idxs]), Pk')
            # to get off-diagonals in ΛFLP, subtract known blocks aggregated in ΛFLP_qp
            @inbounds for q in (p + 1):R
                block_L_q_idxs = block_idxs(L, q)
                @views ΛFLP_qp = ΛFLP[block_L_q_idxs, block_U_p_idxs]
                ΛFLP_qp .= 0
                @inbounds for p2 in p:(q - 1)
                    block_L_p2_idxs = block_idxs(L, p2)
                    @views mul!(ΛFLP_qp, ΛFL[block_L_q_idxs, block_L_p2_idxs], ΛFLP[block_L_p2_idxs, block_U_p_idxs], -1, 1)
                end
                @views ldiv!(LowerTriangular(ΛFL[block_L_q_idxs, block_L_q_idxs]), ΛFLP_qp)
            end
        end

        # PlambdaP = ΛFLP' * ΛFLP
        PlambdaPk = out_T.(cone.PlambdaP[k])
        for p in 1:R, q in p:R
            block_p_idxs = block_idxs(U, p)
            block_q_idxs = block_idxs(U, q)
            # since ΛFLP is block lower triangular rows only from max(p,q) start making a nonzero contribution to the product
            row_range = ((q - 1) * L + 1):(L * R)
            # @show typeof(ΛFLP)
            @inbounds @views mul!(PlambdaPk[block_p_idxs, block_q_idxs], ΛFLP[row_range, block_p_idxs]', ΛFLP[row_range, block_q_idxs])
        end

        for p in 1:R, q in 1:p
            block = svec_idx(p, q)
            idxs = block_idxs(U, block)
            block_p_idxs = block_idxs(U, p)
            block_q_idxs = block_idxs(U, q)

            for p2 in 1:R, q2 in 1:p2
                block2 = svec_idx(p2, q2)
                if block2 < block
                    continue
                end
                idxs2 = block_idxs(U, block2)
                block_p_idxs2 = block_idxs(U, p2)
                block_q_idxs2 = block_idxs(U, q2)

                @views Hview = H[idxs, idxs2]
                PlambdaPk = Symmetric(PlambdaPk, :U)
                    # PlambdaPk = Symmetric(cone.PlambdaP[k], :U)
                #     @inbounds @. @views Hview += PlambdaPk[block_p_idxs, block_p_idxs2] * PlambdaPk[block_q_idxs, block_q_idxs2]
                #     if (p != q) || (p2 != q2)
                #         @inbounds @. @views Hview += PlambdaPk[block_p_idxs, block_q_idxs2] * PlambdaPk[block_q_idxs, block_p_idxs2]
                #     end
                # if xor(p == q, p2 == q2)
                #     @. Hview *= cone.rt2i
                # end
                # @show typeof(PlambdaPk)
                @. @views Hview += (PlambdaPk[block_p_idxs, block_p_idxs2] * PlambdaPk[block_q_idxs, block_q_idxs2] + PlambdaPk[block_p_idxs, block_q_idxs2] * PlambdaPk[block_q_idxs, block_p_idxs2]) *
                    (p == q ? 1 : sqrt(out_T(2))) * (p2 == q2 ? 1 : sqrt(out_T(2))) / 2
            end
        end
    end

    cone.hess_updated = true
    return cone.hess
end
