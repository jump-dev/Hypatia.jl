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
    tmpRR::Matrix{T}
    tmpRR2::Matrix{T}
    tmpRR3::Matrix{T}
    tmpRRUU::Vector{Vector{Matrix{T}}}
    ΛFL::Vector
    ΛFLP::Vector{Matrix{T}}
    tmpLU::Vector{Matrix{T}}
    PlambdaP::Vector{Matrix{T}}

    PlambdaP_blocks_U::Vector{Matrix{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}}
    PlambdaP_blocks_R::Vector{Matrix{Matrix{T}}}
    blocks_R_updated::Bool

    UU1::Matrix{T}
    UU2::Matrix{T}
    UU3::Matrix{T}
    UU4::Matrix{T}
    UU5::Matrix{T}
    UU6::Matrix{T}
    UU7::Matrix{T}
    UU8::Matrix{T}
    UU9::Matrix{T}
    UU10::Matrix{T}
    UU11::Matrix{T}
    UU12::Matrix{T}

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
    cone.tmpRR = zeros(T, R, R)
    cone.tmpRR2 = zeros(T, R, R)
    cone.tmpRR3 = zeros(T, R, R)
    cone.tmpRRUU = [[zeros(T, R, R) for _ in 1:U] for _ in 1:U]
    cone.ΛFL = Vector{Any}(undef, length(Ps))
    cone.ΛFLP = [Matrix{T}(undef, R * size(Pk, 2), R * U) for Pk in Ps]
    cone.PlambdaP = [zeros(T, R * U, R * U) for _ in eachindex(Ps)]
    cone.PlambdaP_blocks_U = [Matrix{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}(undef, R, R) for _ in eachindex(Ps)]
    @inbounds for k in eachindex(Ps), r in 1:R, s in 1:R
        cone.PlambdaP_blocks_U[k][r, s] = view(cone.PlambdaP[k], block_idxs(U, r), block_idxs(U, s))
    end
    cone.PlambdaP_blocks_R = [Matrix{Matrix{T}}(undef, U, U) for _ in eachindex(Ps)]
    @inbounds for k in eachindex(Ps), r in 1:U, s in 1:U
        cone.PlambdaP_blocks_R[k][r, s] = zeros(T, R, R)
    end
    cone.UU1 = zeros(T, U, U)
    cone.UU2 = zeros(T, U, U)
    cone.UU3 = zeros(T, U, U)
    cone.UU4 = zeros(T, U, U)
    cone.UU5 = zeros(T, U, U)
    cone.UU6 = zeros(T, U, U)
    cone.UU7 = zeros(T, U, U)
    cone.UU8 = zeros(T, U, U)
    cone.UU9 = zeros(T, U, U)
    cone.UU10 = zeros(T, U, U)
    cone.UU11 = zeros(T, U, U)
    cone.UU12 = zeros(T, U, U)
    return
end

reset_data(cone::WSOSInterpPosSemidefTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = cone.blocks_R_updated = false)

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

is_dual_feas(cone::WSOSInterpPosSemidefTri) = true

function update_grad(cone::WSOSInterpPosSemidefTri)
    @assert is_feas(cone)
    U = cone.U
    R = cone.R

    # update PlambdaP
    @inbounds for k in eachindex(cone.PlambdaP)
        L = size(cone.Ps[k], 2)
        ΛFL = cone.ΛFL[k].L
        ΛFLP = cone.ΛFLP[k]

        # given cholesky L factor ΛFL, get ΛFLP = ΛFL \ kron(I, P')
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
        @inbounds for p in 1:R, q in p:R
            block_p_idxs = block_idxs(U, p)
            block_q_idxs = block_idxs(U, q)
            # since ΛFLP is block lower triangular rows only from max(p,q) start making a nonzero contribution to the product
            row_range = ((q - 1) * L + 1):(L * R)
            @inbounds @views mul!(PlambdaPk[block_p_idxs, block_q_idxs], ΛFLP[row_range, block_p_idxs]', ΛFLP[row_range, block_q_idxs])
        end
        LinearAlgebra.copytri!(PlambdaPk, 'U')
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


function update_blocks_R(cone::WSOSInterpPosSemidefTri)
    @assert cone.grad_updated
    U = cone.U
    # TODO only upper triangle
    @inbounds for k in eachindex(cone.Ps), q in 1:U, p in 1:U
        @views copyto!(cone.PlambdaP_blocks_R[k][p, q], cone.PlambdaP[k][p:U:(U * cone.R), q:U:(U * cone.R)])
    end
    cone.blocks_R_updated = true
    return
end

function update_hess(cone::WSOSInterpPosSemidefTri)
    @assert cone.grad_updated
    R = cone.R
    U = cone.U
    H = cone.hess.data
    H .= 0
    @inbounds for p in 1:R, q in 1:p
        block = svec_idx(p, q)
        idxs = block_idxs(U, block)

        for p2 in 1:R, q2 in 1:p2
            block2 = svec_idx(p2, q2)
            if block2 < block
                continue
            end
            idxs2 = block_idxs(U, block2)

            @views Hview = H[idxs, idxs2]
            for k in eachindex(cone.Ps)
                PlambdaPk = cone.PlambdaP_blocks_U[k]
                @inbounds @. @views Hview += PlambdaPk[p, p2] * PlambdaPk[q, q2]
                if (p != q) && (p2 != q2)
                    @inbounds @. @views Hview += PlambdaPk[p, q2] * PlambdaPk[q, p2]
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

function use_correction(cone::WSOSInterpPosSemidefTri)
    if cone.U < 30
        return true
    else
        return false
    end
end

function correction(cone::WSOSInterpPosSemidefTri{T}, primal_dir::AbstractVector{T}) where {T}
    if cone.U < 30
        correction1(cone, primal_dir)
    else
        correction2(cone, primal_dir)
    end
end

function correction1(cone::WSOSInterpPosSemidefTri{T}, primal_dir::AbstractVector{T}) where {T}
    @assert cone.grad_updated
    if !cone.blocks_R_updated
        @timeit cone.timer "update_blocks" update_blocks_R(cone)
    end
    corr = cone.correction
    corr .= 0
    U = cone.U
    UR = U * cone.R
    dim = cone.dim
    PlambdaP_dirs = cone.tmpRRUU
    matRR = cone.tmpRR
    matRR2 = cone.tmpRR2
    matRR3 = cone.tmpRR3

    @inbounds for k in eachindex(cone.Ps)
        PlambdaPk = cone.PlambdaP_blocks_R[k]

        @inbounds for p in 1:U, q in 1:U
            @views primal_dir_mat_q = Symmetric(svec_to_smat!(matRR, primal_dir[q:U:dim], cone.rt2))
            PlambdaPk_slice_pq = PlambdaPk[p, q]
            mul!(PlambdaP_dirs[p][q], PlambdaPk_slice_pq, primal_dir_mat_q)
        end

        @inbounds for p in 1:U
            @views primal_dir_mat_p = Symmetric(svec_to_smat!(matRR, primal_dir[p:U:dim], cone.rt2))
            @inbounds for q in 1:U
                pq_q = PlambdaP_dirs[p][q] # PlambdaPk_slice_pq * primal_dir_mat_q
                @timeit cone.timer "loop5" @inbounds for r in 1:q
                    PlambdaPk_slice_qr = PlambdaPk[q, r]
                    r_rp = PlambdaP_dirs[p][r]'

                    # O(R^3) done O(U^3) times
                    @timeit cone.timer "mul2" mul!(matRR2, PlambdaPk_slice_qr, r_rp)
                    @timeit cone.timer "mul3" mul!(matRR3, pq_q, matRR2)
                    @timeit cone.timer "axpy" axpy!(true, matRR3, matRR3')
                    if q != r
                        matRR3 .*= 2
                    end
                    @views smat_to_svec_add!(corr[p:U:dim], matRR3, cone.rt2)
                end
            end
        end
    end
    corr ./= 2

    return corr
end

function correction2(cone::WSOSInterpPosSemidefTri, primal_dir::AbstractVector)
    @assert cone.grad_updated
    corr = cone.correction
    U = cone.U
    R = cone.R
    corr .= 0
    rt2 = cone.rt2

    @inbounds for p in eachindex(cone.Ps)
        PlambdaPk = cone.PlambdaP_blocks_U[p]

        idx_kl = 1
        for l in 1:R, k in 1:l
            scal_kl = (k == l ? 1 : rt2)
            idx_mn = 1
            for n in 1:R, m in 1:n
                scal_mn = (m == n ? 1 : rt2)
                @views primal_dir_kl = Diagonal(primal_dir[block_idxs(U, idx_kl)])
                @views primal_dir_mn = Diagonal(primal_dir[block_idxs(U, idx_mn)])
                PlambdaPk_slice_km = PlambdaPk[k, m]
                PlambdaPk_slice_kn = PlambdaPk[k, n]
                PlambdaPk_slice_lm = PlambdaPk[l, m]
                PlambdaPk_slice_ln = PlambdaPk[l, n]

                @timeit cone.timer "muls_1" begin
                    mul!(cone.UU5, PlambdaPk_slice_kn, primal_dir_mn)
                    mid1 = mul!(cone.UU1, primal_dir_kl, cone.UU5)
                    mul!(cone.UU5, PlambdaPk_slice_ln, primal_dir_mn)
                    mid2 = mul!(cone.UU2, primal_dir_kl, cone.UU5)
                    mul!(cone.UU5, PlambdaPk_slice_km, primal_dir_mn)
                    mid3 = mul!(cone.UU3, primal_dir_kl, cone.UU5)
                    mul!(cone.UU5, PlambdaPk_slice_lm, primal_dir_mn)
                    mid4 = mul!(cone.UU4, primal_dir_kl, cone.UU5)
                end

                idx_ij = 1
                for j in 1:R, i in 1:j
                    scal_ij = (i == j ? 1 : rt2)
                    corr_ij = view(corr, block_idxs(U, idx_ij))

                    PlambdaPk_slice_ik = PlambdaPk[i, k]
                    PlambdaPk_slice_il = PlambdaPk[i, l]
                    PlambdaPk_slice_im = PlambdaPk[i, m]
                    PlambdaPk_slice_in = PlambdaPk[i, n]
                    PlambdaPk_slice_jk = PlambdaPk[j, k]
                    PlambdaPk_slice_jl = PlambdaPk[j, l]
                    PlambdaPk_slice_jm = PlambdaPk[j, m]
                    PlambdaPk_slice_jn = PlambdaPk[j, n]

                    @timeit cone.timer "muls_2" begin
                        right1 = mul!(cone.UU5, mid1, PlambdaPk_slice_jm')
                        right2 = mul!(cone.UU6, mid1, PlambdaPk_slice_im')
                        right3 = mul!(cone.UU7, mid2, PlambdaPk_slice_jm')
                        right4 = mul!(cone.UU8, mid2, PlambdaPk_slice_im')
                        right5 = mul!(cone.UU9, mid3, PlambdaPk_slice_jn')
                        right6 = mul!(cone.UU10, mid3, PlambdaPk_slice_in')
                        right7 = mul!(cone.UU11, mid4, PlambdaPk_slice_jn')
                        right8 = mul!(cone.UU12, mid4, PlambdaPk_slice_in')
                    end

                    scal = scal_ij * scal_kl * scal_mn / 4
                    for u in 1:U
                        @timeit cone.timer "dot" @views corr_ij[u] += scal * (
                            dot(PlambdaPk_slice_il[u, :], right1[:, u]) +
                            dot(PlambdaPk_slice_jl[u, :], right2[:, u]) +
                            dot(PlambdaPk_slice_ik[u, :], right3[:, u]) +
                            dot(PlambdaPk_slice_jk[u, :], right4[:, u]) +
                            dot(PlambdaPk_slice_il[u, :], right5[:, u]) +
                            dot(PlambdaPk_slice_jl[u, :], right6[:, u]) +
                            dot(PlambdaPk_slice_ik[u, :], right7[:, u]) +
                            dot(PlambdaPk_slice_jk[u, :], right8[:, u])
                            )
                    end
                    idx_ij += 1
                end
                idx_mn += 1
            end
            idx_kl += 1
        end
    end

    corr ./= 2

    return corr
end

# function correction2(cone::WSOSInterpPosSemidefTri, primal_dir::AbstractVector)
#     @assert cone.grad_updated
#     corr = cone.correction
#     U = cone.U
#     R = cone.R
#     corr .= 0
#     rt2 = cone.rt2
#
#     @inbounds for p in eachindex(cone.Ps)
#         PlambdaPk = cone.PlambdaP_blocks_U[p]
#
#         idx_kl = 1
#         for l in 1:R, k in 1:l
#             scal_kl = (k == l ? 1 : rt2)
#             idx_mn = 1
#             for n in 1:R, m in 1:n
#                 scal_mn = (m == n ? 1 : rt2)
#                 @views primal_dir_kl = Diagonal(primal_dir[block_idxs(U, idx_kl)])
#                 @views primal_dir_mn = Diagonal(primal_dir[block_idxs(U, idx_mn)])
#                 PlambdaPk_slice_km = PlambdaPk[k, m]
#                 PlambdaPk_slice_kn = PlambdaPk[k, n]
#                 PlambdaPk_slice_lm = PlambdaPk[l, m]
#                 PlambdaPk_slice_ln = PlambdaPk[l, n]
#
#                 idx_ij = 1
#                 for j in 1:R, i in 1:j
#                     scal_ij = (i == j ? 1 : rt2)
#                     corr_ij = view(corr, block_idxs(U, idx_ij))
#
#                     PlambdaPk_slice_ik = PlambdaPk[i, k]
#                     PlambdaPk_slice_il = PlambdaPk[i, l]
#                     PlambdaPk_slice_im = PlambdaPk[i, m]
#                     PlambdaPk_slice_in = PlambdaPk[i, n]
#                     PlambdaPk_slice_jk = PlambdaPk[j, k]
#                     PlambdaPk_slice_jl = PlambdaPk[j, l]
#                     PlambdaPk_slice_jm = PlambdaPk[j, m]
#                     PlambdaPk_slice_jn = PlambdaPk[j, n]
#
#                     @timeit cone.timer "muls_1" begin
#                         left1 = mul!(cone.UU1, PlambdaPk_slice_il, primal_dir_kl)
#                         left2 = mul!(cone.UU2, PlambdaPk_slice_jl, primal_dir_kl)
#                         left3 = mul!(cone.UU3, PlambdaPk_slice_ik, primal_dir_kl)
#                         left4 = mul!(cone.UU4, PlambdaPk_slice_jk, primal_dir_kl)
#
#                         right1 = mul!(cone.UU5, primal_dir_mn, PlambdaPk_slice_jm')
#                         right2 = mul!(cone.UU6, primal_dir_mn, PlambdaPk_slice_im')
#                         right3 = mul!(cone.UU7, primal_dir_mn, PlambdaPk_slice_jn')
#                         right4 = mul!(cone.UU8, primal_dir_mn, PlambdaPk_slice_in')
#                     end
#
#                     scal = scal_ij * scal_kl * scal_mn / 4
#                     for u in 1:U
#                         @timeit cone.timer "dot" @views corr_ij[u] += scal * (
#                             dot(left1[u, :], PlambdaPk_slice_kn, right1[:, u]) +
#                             dot(left2[u, :], PlambdaPk_slice_kn, right2[:, u]) +
#                             dot(left3[u, :], PlambdaPk_slice_ln, right1[:, u]) +
#                             dot(left4[u, :], PlambdaPk_slice_ln, right2[:, u]) +
#                             dot(left1[u, :], PlambdaPk_slice_km, right3[:, u]) +
#                             dot(left2[u, :], PlambdaPk_slice_km, right4[:, u]) +
#                             dot(left3[u, :], PlambdaPk_slice_lm, right3[:, u]) +
#                             dot(left4[u, :], PlambdaPk_slice_lm, right4[:, u])
#                             )
#                     end
#                     idx_ij += 1
#                 end
#                 idx_mn += 1
#             end
#             idx_kl += 1
#         end
#     end
#
#     corr ./= 2
#
#     return corr
# end
