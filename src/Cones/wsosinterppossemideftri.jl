"""
$(TYPEDEF)

Interpolant-basis weighted sum-of-squares polynomial (of dimension `U`) positive
semidefinite matrix (of side dimension `R`) cone, parametrized by vector of
matrices `Ps` derived from interpolant basis and polynomial domain constraints.

    $(FUNCTIONNAME){T}(R::Int, U::Int, Ps::Vector{Matrix{T}}, use_dual::Bool = false)
"""
mutable struct WSOSInterpPosSemidefTri{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    nu::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
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
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}
    use_hess_prod_slow::Bool
    use_hess_prod_slow_updated::Bool

    rt2::T
    rt2i::T
    tempU::Vector{T}
    tempLRLR::Vector{Symmetric{T, Matrix{T}}}
    tempLRLR2::Vector{Matrix{T}}
    tempLRUR::Vector{Matrix{T}}
    ΛFL::Vector
    ΛFLP::Vector{Matrix{T}}
    tempLU::Vector{Matrix{T}}
    PΛiP::Matrix{T}
    PΛiP_blocks_U
    Ps_times::Vector{Float64}
    Ps_order::Vector{Int}

    function WSOSInterpPosSemidefTri{T}(
        R::Int,
        U::Int,
        Ps::Vector{Matrix{T}};
        use_dual::Bool = false,
        ) where {T <: Real}
        for Pk in Ps
            @assert size(Pk, 1) == U
        end
        cone = new{T}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.dim = U * svec_length(R)
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.nu = R * sum(size(Pk, 2) for Pk in Ps)
        return cone
    end
end

reset_data(cone::WSOSInterpPosSemidefTri) = (cone.feas_updated =
    cone.grad_updated = cone.hess_updated = cone.inv_hess_updated =
    cone.hess_fact_updated = cone.use_hess_prod_slow =
    cone.use_hess_prod_slow_updated = false)

function setup_extra_data!(cone::WSOSInterpPosSemidefTri{T}) where {T <: Real}
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    K = length(Ps)
    cone.rt2 = sqrt(T(2))
    cone.rt2i = inv(cone.rt2)
    cone.tempU = zeros(T, U)
    Ls = [size(Pk, 2) for Pk in cone.Ps]
    cone.tempLRLR = [Symmetric(zeros(T, L * R, L * R), :L) for L in Ls]
    cone.tempLRLR2 = [zeros(T, L * R, L * R) for L in Ls]
    cone.tempLRUR = [zeros(T, L * R, U * R) for L in Ls]
    cone.tempLU = [zeros(T, L, U) for L in Ls]
    cone.ΛFL = Vector{Any}(undef, K)
    cone.ΛFLP = [zeros(T, R * L, R * U) for L in Ls]
    cone.PΛiP = zeros(T, R * U, R * U)
    cone.PΛiP_blocks_U = [view(cone.PΛiP, block_idxs(U, r), block_idxs(U, s))
        for r in 1:R, s in 1:R]
    cone.Ps_times = zeros(K)
    cone.Ps_order = collect(1:K)
    return cone
end

function set_initial_point!(arr::AbstractVector, cone::WSOSInterpPosSemidefTri)
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
    sortperm!(cone.Ps_order, cone.Ps_times, initialized = true) # stochastic

    cone.is_feas = true
    for k in cone.Ps_order
        cone.Ps_times[k] = @elapsed @inbounds begin
            Pk = cone.Ps[k]
            LU = cone.tempLU[k]
            L = size(Pk, 2)
            Λ = cone.tempLRLR[k]

            @views for p in 1:cone.R, q in 1:p
                @. cone.tempU = cone.point[block_idxs(cone.U, svec_idx(p, q))]
                if p != q
                    cone.tempU .*= cone.rt2i
                end
                mul!(LU, Pk', Diagonal(cone.tempU)) # TODO check efficiency
                mul!(Λ.data[block_idxs(L, p), block_idxs(L, q)], LU, Pk)
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

function update_grad(cone::WSOSInterpPosSemidefTri)
    @assert is_feas(cone)
    U = cone.U
    R = cone.R
    grad = cone.grad
    cone.grad .= 0

    @inbounds for k in eachindex(cone.Ps)
        L = size(cone.Ps[k], 2)
        ΛFL = cone.ΛFL[k].L
        ΛFLP = cone.ΛFLP[k]

        # given cholesky L factor ΛFL, get ΛFLP = ΛFL \ kron(I, P')
        for p in 1:R
            block_U_p_idxs = block_idxs(U, p)
            block_L_p_idxs = block_idxs(L, p)
            @views ΛFLP_pp = ΛFLP[block_L_p_idxs, block_U_p_idxs]
            # ΛFLP_pp = ΛFL_pp \ P'
            @views ldiv!(ΛFLP_pp, LowerTriangular(
                ΛFL[block_L_p_idxs, block_L_p_idxs]), cone.Ps[k]')
            # to get off-diagonals in ΛFLP, subtract known blocks aggregated in ΛFLP_qp
            for q in (p + 1):R
                block_L_q_idxs = block_idxs(L, q)
                @views ΛFLP_qp = ΛFLP[block_L_q_idxs, block_U_p_idxs]
                ΛFLP_qp .= 0
                for p2 in p:(q - 1)
                    block_L_p2_idxs = block_idxs(L, p2)
                    @views mul!(ΛFLP_qp, ΛFL[block_L_q_idxs, block_L_p2_idxs],
                        ΛFLP[block_L_p2_idxs, block_U_p_idxs], -1, 1)
                end
                @views ldiv!(LowerTriangular(ΛFL[block_L_q_idxs,
                    block_L_q_idxs]), ΛFLP_qp)
            end
        end

        # update grad
        block_diag_prod!(grad, ΛFLP, ΛFLP, cone)
    end
    grad .*= -1

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpPosSemidefTri{T}) where {T <: Real}
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    R = cone.R
    U = cone.U
    PΛiP_blocks = cone.PΛiP_blocks_U
    H .= 0

    @inbounds for k in eachindex(cone.Ps)
        L = size(cone.Ps[k], 2)
        # PΛiP = ΛFLP' * ΛFLP
        ΛFLP = cone.ΛFLP[k]
        for p in 1:R, q in p:R
            # since ΛFLP is block lower triangular rows only from max(p,q)
            # start making a nonzero contribution to the product
            row_range = ((q - 1) * L + 1):(L * R)
            @views mul!(PΛiP_blocks[p, q], ΛFLP[row_range, block_idxs(U, p)]',
                ΛFLP[row_range, block_idxs(U, q)])
        end
        LinearAlgebra.copytri!(cone.PΛiP, 'U')

        for p in 1:R, q in 1:p
            block = svec_idx(p, q)
            idxs = block_idxs(U, block)

            for p2 in 1:R, q2 in 1:p2
                block2 = svec_idx(p2, q2)
                if block2 < block
                    continue
                end
                idxs2 = block_idxs(U, block2)

                @views Hview = H[idxs, idxs2]
                scal = (xor(p == q, p2 == q2) ? cone.rt2 : one(T))
                @. Hview += PΛiP_blocks[p, p2] * PΛiP_blocks[q, q2] * scal
                if (p != q) && (p2 != q2)
                    @inbounds @. Hview += PΛiP_blocks[p, q2] * PΛiP_blocks[q, p2]
                end
            end
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod_slow!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::WSOSInterpPosSemidefTri,
    )
    cone.use_hess_prod_slow_updated || update_use_hess_prod_slow(cone)
    @assert cone.hess_updated
    cone.use_hess_prod_slow || return hess_prod!(prod, arr, cone)
    return partial_prod!(prod, arr, false, cone)
end

function dder3(cone::WSOSInterpPosSemidefTri, dir::AbstractVector)
    @assert cone.grad_updated
    return partial_prod!(cone.dder3, dir, true, cone)
end

# diagonal from each (i, j) block in mat1' * mat2
function block_diag_prod!(
    vect::AbstractVector{T},
    mat1::Matrix{T},
    mat2::Matrix{T},
    cone::WSOSInterpPosSemidefTri{T},
    ) where T
    U = cone.U
    @inbounds for u in 1:U
        idx = u
        j_idx = u
        for j in 1:cone.R
            i_idx = u
            for i in 1:(j - 1)
                @views vect[idx] += dot(mat1[:, i_idx], mat2[:, j_idx]) * cone.rt2
                idx += U
                i_idx += U
            end
            @views vect[idx] += dot(mat1[:, j_idx], mat2[:, j_idx])
            j_idx += U
            idx += U
        end
    end
    return
end

function partial_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    use_symm_prod::Bool,
    cone::WSOSInterpPosSemidefTri,
    )
    @assert cone.grad_updated
    prod .= 0
    U = cone.U
    R = cone.R
    tempU = cone.tempU

    @inbounds for (k, Pk) in enumerate(cone.Ps)
        Lk = size(Pk, 2)
        LUk = cone.tempLU[k]
        LRURk = cone.tempLRUR[k]
        LRLRk = cone.tempLRLR2[k]
        ΛFLk = cone.ΛFL[k]
        ΛFLPk = cone.ΛFLP[k]
        left_prod = (use_symm_prod ? LRURk : ΛFLPk)

        @views for j in 1:size(arr, 2)
            delta = arr[:, j]

            for q in 1:R, p in 1:q
                @. tempU = delta[block_idxs(U, svec_idx(q, p))]
                if p != q
                    # svec scaling
                    tempU .*= cone.rt2i
                end
                mul!(LUk, Pk', Diagonal(tempU))
                mul!(LRLRk[block_idxs(Lk, p), block_idxs(Lk, q)], LUk, Pk)
            end

            LinearAlgebra.copytri!(LRLRk, 'U')
            ldiv!(ΛFLk.L, LRLRk)
            rdiv!(LRLRk, ΛFLk.L')
            mul!(LRURk, Symmetric(LRLRk), ΛFLPk)

            block_diag_prod!(prod[:, j], left_prod, LRURk, cone)
        end
    end

    return prod
end
