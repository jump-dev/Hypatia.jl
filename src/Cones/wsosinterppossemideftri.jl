#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial positive semidefinite cone parametrized by interpolation matrices Ps
certifies that a polynomial valued R x R matrix is in the positive semidefinite cone for all x in the domain defined by Ps

dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
and "Semidefinite Characterization of Sum-of-Squares Cones in Algebras" by D. Papp and F. Alizadeh
=#

mutable struct WSOSInterpPosSemidefTri{T <: Real} <: Cone{T}
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
    cone.dual_point = zeros(T, dim)
    cone.grad = similar(cone.point)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
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

use_correction(cone::WSOSInterpPosSemidefTri) = true

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
        Pk = cone.Ps[k]
        LU = cone.tmpLU[k]
        L = size(Pk, 2)
        Λ = cone.tmpLRLR[k]

        for p in 1:cone.R, q in 1:p
            @. @views cone.tmpU = cone.point[block_idxs(cone.U, svec_idx(p, q))]
            if p != q
                cone.tmpU .*= cone.rt2i
            end
            mul!(LU, Pk', Diagonal(cone.tmpU)) # TODO check efficiency
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

update_dual_feas(cone::WSOSInterpPosSemidefTri) = true

function update_grad(cone::WSOSInterpPosSemidefTri)
    @assert is_feas(cone)
    U = cone.U
    R = cone.R

    # update PlambdaP
    for k in eachindex(cone.PlambdaP)
        L = size(cone.Ps[k], 2)
        ΛFL = cone.ΛFL[k].L
        ΛFLP = cone.ΛFLP[k]

        # given cholesky L factor ΛFL, get ΛFLP = ΛFL * kron(I, P')
        @inbounds for p in 1:R
            block_U_p_idxs = block_idxs(U, p)
            block_L_p_idxs = block_idxs(L, p)
            @views ΛFLP_pp = ΛFLP[block_L_p_idxs, block_U_p_idxs]
            # ΛFLP_pp = ΛFL_pp \ P'
            @views ldiv!(ΛFLP_pp, LowerTriangular(ΛFL[block_L_p_idxs, block_L_p_idxs]), cone.Ps[k]')
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
        PlambdaPk = cone.PlambdaP[k]
        for p in 1:R, q in p:R
            block_p_idxs = block_idxs(U, p)
            block_q_idxs = block_idxs(U, q)
            # since ΛFLP is block lower triangular rows only from max(p,q) start making a nonzero contribution to the product
            row_range = ((q - 1) * L + 1):(L * R)
            @inbounds @views mul!(PlambdaPk[block_p_idxs, block_q_idxs], ΛFLP[row_range, block_p_idxs]', ΛFLP[row_range, block_q_idxs])
        end
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
            @inbounds cone.grad[idx + i] = scal * sum(PlambdaPk[block_q_i, block_p_i] for PlambdaPk in cone.PlambdaP)
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
            for k in eachindex(cone.Ps)
                PlambdaPk = Symmetric(cone.PlambdaP[k], :U)
                @inbounds @. @views Hview += PlambdaPk[block_p_idxs, block_p_idxs2] * PlambdaPk[block_q_idxs, block_q_idxs2]
                if (p != q) && (p2 != q2)
                    @inbounds @. @views Hview += PlambdaPk[block_p_idxs, block_q_idxs2] * PlambdaPk[block_q_idxs, block_p_idxs2]
                end
            end
            if xor(p == q, p2 == q2)
                @. Hview *= cone.rt2
            end
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# this works better when point is shuffled. everything else works better when it is not. maybe save a permutation.
function hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::WSOSInterpPosSemidefTri{T}) where {T}
    @assert cone.grad_updated
    U = cone.U
    R = cone.R
    prod .= 0

    arr_shuf = similar(arr)
    prod_shuf = similar(prod)
    prod_shuf .= 0
    r_dim = svec_length(R)

    # permute array
    for k in 1:size(prod, 2)
        idx_new = 1
        for u in 1:cone.U
            r_idx = 1
            for j in 1:cone.R, i in 1:j
                arr_shuf[idx_new, k] = arr[block_idxs(U, r_idx)[u], k]
                r_idx += 1
                idx_new += 1
            end
        end
    end

    # get permuted product, for a vector requires O(U^2) outer products of size O(R), while hess side is O(R^2*U)
    for j in 1:size(prod, 2)
        arr_shufj = arr_shuf[:, j]
        for k in eachindex(cone.Ps)
            PlambdaPk = Symmetric(cone.PlambdaP[k], :U)
            for p in 1:U
                for q in 1:U
                    arr_mat = svec_to_smat!(zeros(T, R, R), arr_shufj[block_idxs(r_dim, q)], cone.rt2)
                    PlambdaPk_slice = [PlambdaPk[block_idxs(U, ii)[p], block_idxs(U, jj)[q]] for ii in 1:R, jj in 1:R]
                    prod_mat = PlambdaPk_slice * Symmetric(arr_mat) * PlambdaPk_slice'
                    tmp = smat_to_svec!(zeros(r_dim), prod_mat, cone.rt2)
                    prod_shuf[block_idxs(r_dim, p), j] += tmp
                end
            end
        end
    end

    # un-permute product
    # @show prod_shuf
    for k in 1:size(prod, 2)
        r_idx = 1
        idx = 1
        for j in 1:cone.R, i in 1:j
            for u in 1:U
                # @show block_idxs(r_dim, u), length(block_idxs(r_dim, u)), r_idx
                prod[idx, k] = prod_shuf[block_idxs(r_dim, u)[r_idx], k]
                idx += 1
            end
            r_idx += 1
        end
    end
    # @show prod

    return prod

end

function correction2(cone::WSOSInterpPosSemidefTri, primal_dir::AbstractVector)
    @assert cone.grad_updated
    corr = cone.correction
    U = cone.U
    R = cone.R

    s_shuf = similar(primal_dir)
    corr_shuf = similar(corr)
    corr_shuf .= 0
    r_dim = svec_length(R)

    # permute primal_dir
    idx_new = 1
    for u in 1:cone.U
        r_idx = 1
        for j in 1:cone.R, i in 1:j
            s_shuf[idx_new] = primal_dir[block_idxs(U, r_idx)[u]]
            r_idx += 1
            idx_new += 1
        end
    end

    tmp = similar(corr, r_dim)
    tmp .= 0
    for k in eachindex(cone.Ps)
        PlambdaPk = Symmetric(cone.PlambdaP[k], :U)
        for p in 1:U
            for q in 1:U, r in 1:U
                primal_dir_mat_q = Symmetric(svec_to_smat!(similar(corr, R, R), s_shuf[block_idxs(r_dim, q)], cone.rt2))
                primal_dir_mat_r = Symmetric(svec_to_smat!(similar(corr, R, R), s_shuf[block_idxs(r_dim, r)], cone.rt2))
                PlambdaPk_slice_pq = [PlambdaPk[block_idxs(U, ii)[p], block_idxs(U, jj)[q]] for ii in 1:R, jj in 1:R]
                PlambdaPk_slice_qr = [PlambdaPk[block_idxs(U, ii)[q], block_idxs(U, jj)[r]] for ii in 1:R, jj in 1:R]
                PlambdaPk_slice_rp = [PlambdaPk[block_idxs(U, ii)[r], block_idxs(U, jj)[p]] for ii in 1:R, jj in 1:R]
                prod_mat = PlambdaPk_slice_pq * primal_dir_mat_q * PlambdaPk_slice_qr * primal_dir_mat_r * PlambdaPk_slice_rp
                prod_mat += prod_mat'
                corr_shuf[block_idxs(r_dim, p)] .+= smat_to_svec!(similar(corr, r_dim), prod_mat, cone.rt2)
            end
        end
    end

    r_idx = 1
    idx = 1
    for j in 1:cone.R, i in 1:j
        for u in 1:U
            corr[idx] = corr_shuf[block_idxs(r_dim, u)[r_idx]]
            idx += 1
        end
        r_idx += 1
    end

    corr ./= 2

    return corr
end
