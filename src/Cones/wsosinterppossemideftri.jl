#=
interpolation-based weighted-sum-of-squares (multivariate) polynomial positive semidefinite cone parametrized by interpolation matrices Ps
certifies that a polynomial valued R x R matrix is in the positive semidefinite cone for all x in the domain defined by Ps

dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
and "Semidefinite Characterization of Sum-of-Squares Cones in Algebras" by D. Papp and F. Alizadeh
=#

mutable struct WSOSInterpPosSemidefTri{T <: Real} <: Cone{T}
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

    rt2::T
    rt2i::T
    tempU::Vector{T}
    tempLRLR::Vector{Symmetric{T, Matrix{T}}}
    tempLRUR::Vector{Matrix{T}}
    tempLRUR2::Vector{Matrix{T}}
    ΛFL::Vector
    ΛFLP::Vector{Matrix{T}}
    tempLU::Vector{Matrix{T}}
    PΛiP::Vector{Matrix{T}}
    PΛiP_blocks_U::Vector
    Ps_times::Vector{Float64}
    Ps_order::Vector{Int}

    function WSOSInterpPosSemidefTri{T}(
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
        cone.dim = U * svec_length(R)
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data(cone::WSOSInterpPosSemidefTri{T}) where {T <: Real}
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    K = length(Ps)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.rt2 = sqrt(T(2))
    cone.rt2i = inv(cone.rt2)
    cone.tempU = zeros(T, U)
    Ls = [size(Pk, 2) for Pk in cone.Ps]
    cone.tempLRLR = [Symmetric(zeros(T, L * R, L * R), :L) for L in Ls]
    cone.tempLRUR = [zeros(T, L * R, U * R) for L in Ls]
    cone.tempLRUR2 = [zeros(T, L * R, U * R) for L in Ls]
    cone.tempLU = [zeros(T, L, U) for L in Ls]
    cone.ΛFL = Vector{Any}(undef, K)
    cone.ΛFLP = [zeros(T, R * L, R * U) for L in Ls]
    cone.PΛiP = [zeros(T, R * U, R * U) for _ in eachindex(Ps)]
    cone.PΛiP_blocks_U = [[view(PΛiPk, block_idxs(U, r), block_idxs(U, s)) for r in 1:R, s in 1:R] for PΛiPk in cone.PΛiP]
    cone.Ps_times = zeros(K)
    cone.Ps_order = collect(1:K)
    return cone
end

get_nu(cone::WSOSInterpPosSemidefTri) = cone.R * sum(size(Pk, 2) for Pk in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSInterpPosSemidefTri)
    arr .= 0
    block = 1
    @inbounds for i in 1:cone.R
        @views arr[block_idxs(cone.U, block)] .= 1
        block += i + 1
    end
    return arr
end

function update_feas(cone::WSOSInterpPosSemidefTri)
    @assert !cone.feas_updated

    # order the Ps by how long it takes to check feasibility, to improve efficiency
    sortperm!(cone.Ps_order, cone.Ps_times, initialized = true) # NOTE stochastic

    cone.is_feas = true
    for k in cone.Ps_order
        cone.Ps_times[k] = @elapsed @inbounds begin
            Pk = cone.Ps[k]
            LU = cone.tempLU[k]
            L = size(Pk, 2)
            Λ = cone.tempLRLR[k]

            for p in 1:cone.R, q in 1:p
                @. @views cone.tempU = cone.point[block_idxs(cone.U, svec_idx(p, q))]
                if p != q
                    cone.tempU .*= cone.rt2i
                end
                mul!(LU, Pk', Diagonal(cone.tempU)) # TODO check efficiency
                @views mul!(Λ.data[block_idxs(L, p), block_idxs(L, q)], LU, Pk)
            end

            ΛFLk = cone.ΛFL[k] = cholesky!(Λ, check = false)
            if !isposdef(ΛFLk)
                cone.is_feas = false
                break
            end
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::WSOSInterpPosSemidefTri) = true

# diagonal from each (i, j) block in mat' * mat
function block_diag_prod!(vect::Vector{T}, mat::Matrix{T}, U::Int, R::Int, rt2::T, scal::Int = 1) where T
    @inbounds for u in 1:U
        idx = u
        j_idx = u
        for j in 1:R
            i_idx = u
            for i in 1:(j - 1)
                @views vect[idx] += dot(mat[:, i_idx], mat[:, j_idx]) * rt2 * scal
                idx += U
                i_idx += U
            end
            @views vect[idx] += sum(abs2, mat[:, j_idx]) * scal
            j_idx += U
            idx += U
        end
    end
    return
end

function update_grad(cone::WSOSInterpPosSemidefTri)
    @assert is_feas(cone)
    U = cone.U
    R = cone.R
    cone.grad .= 0

    @inbounds for k in eachindex(cone.PΛiP)
        L = size(cone.Ps[k], 2)
        ΛFL = cone.ΛFL[k].L
        ΛFLP = cone.ΛFLP[k]

        # given cholesky L factor ΛFL, get ΛFLP = ΛFL \ kron(I, P')
        for p in 1:R
            block_U_p_idxs = block_idxs(U, p)
            block_L_p_idxs = block_idxs(L, p)
            @views ΛFLP_pp = ΛFLP[block_L_p_idxs, block_U_p_idxs]
            # ΛFLP_pp = ΛFL_pp \ P'
            @views ldiv!(ΛFLP_pp, LowerTriangular(ΛFL[block_L_p_idxs, block_L_p_idxs]), cone.Ps[k]')
            # to get off-diagonals in ΛFLP, subtract known blocks aggregated in ΛFLP_qp
            for q in (p + 1):R
                block_L_q_idxs = block_idxs(L, q)
                @views ΛFLP_qp = ΛFLP[block_L_q_idxs, block_U_p_idxs]
                ΛFLP_qp .= 0
                for p2 in p:(q - 1)
                    block_L_p2_idxs = block_idxs(L, p2)
                    @views mul!(ΛFLP_qp, ΛFL[block_L_q_idxs, block_L_p2_idxs], ΛFLP[block_L_p2_idxs, block_U_p_idxs], -1, 1)
                end
                @views ldiv!(LowerTriangular(ΛFL[block_L_q_idxs, block_L_q_idxs]), ΛFLP_qp)
            end
        end

        # update grad
        block_diag_prod!(cone.grad, ΛFLP, U, R, cone.rt2, -1)
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

    @inbounds for k in eachindex(cone.PΛiP)
        L = size(cone.Ps[k], 2)
        # PΛiP = ΛFLP' * ΛFLP
        PΛiPk = cone.PΛiP[k]
        ΛFLP = cone.ΛFLP[k]
        for p in 1:R, q in p:R
            block_p_idxs = block_idxs(U, p)
            block_q_idxs = block_idxs(U, q)
            # since ΛFLP is block lower triangular rows only from max(p,q) start making a nonzero contribution to the product
            row_range = ((q - 1) * L + 1):(L * R)
            @views mul!(PΛiPk[block_p_idxs, block_q_idxs], ΛFLP[row_range, block_p_idxs]', ΛFLP[row_range, block_q_idxs])
        end
        LinearAlgebra.copytri!(PΛiPk, 'U')
    end

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
                PΛiPk = cone.PΛiP_blocks_U[k]
                @inbounds @. Hview += PΛiPk[p, p2] * PΛiPk[q, q2]
                if (p != q) && (p2 != q2)
                    @inbounds @. Hview += PΛiPk[p, q2] * PΛiPk[q, p2]
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

function correction(cone::WSOSInterpPosSemidefTri{T}, primal_dir::AbstractVector{T}) where T
    @assert cone.grad_updated
    corr = cone.correction
    corr .= 0
    U = cone.U
    R = cone.R

    @inbounds for k in eachindex(cone.Ps)
        L = size(cone.Ps[k], 2)
        ΛFLP = cone.ΛFLP[k]
        # ΛFLP * scattered Diagonal of primal_dir
        ΛFLP_dir = cone.tempLRUR[k]
        ΛFLP_dir .= 0
        for i in 1:R
            for p in 1:i # only go up to i since ΛFLP is lower block triangular
                for j in 1:(p - 1)
                    sidx = svec_idx(p, j)
                    @views mul!(ΛFLP_dir[block_idxs(L, i), block_idxs(U, j)], ΛFLP[block_idxs(L, i), block_idxs(U, p)], Diagonal(primal_dir[block_idxs(U, sidx)]), cone.rt2i, true)
                end
                sidx = svec_idx(p, p)
                @views mul!(ΛFLP_dir[block_idxs(L, i), block_idxs(U, p)], ΛFLP[block_idxs(L, i), block_idxs(U, p)], Diagonal(primal_dir[block_idxs(U, sidx)]), true, true)
                for j in (p + 1):R
                    sidx = svec_idx(j, p)
                    @views mul!(ΛFLP_dir[block_idxs(L, i), block_idxs(U, j)], ΛFLP[block_idxs(L, i), block_idxs(U, p)], Diagonal(primal_dir[block_idxs(U, sidx)]), cone.rt2i, true)
                end
            end
        end

        big_mat_half = mul!(cone.tempLRUR2[k], ΛFLP_dir, Symmetric(cone.PΛiP[k], :U))
        block_diag_prod!(corr, big_mat_half, U, R, cone.rt2)
    end

    return corr
end
