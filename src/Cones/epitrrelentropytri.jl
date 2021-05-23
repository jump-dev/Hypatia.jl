# TODO reduce allocations, improve numerics, generalize for complex
# TODO don't use symm_kron_nonsymm!

"""
$(TYPEDEF)

Epigraph of matrix relative entropy cone of dimension `dim` in svec format.

    $(FUNCTIONNAME){T}(dim::Int, use_dual::Bool = false)
"""
mutable struct EpiTrRelEntropyTri{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int

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
    hess_fact_cache

    rt2::T
    V::Matrix{T}
    W::Matrix{T}
    Vi::Matrix{T}
    Wi::Matrix{T}
    V_idxs::UnitRange{Int}
    W_idxs::UnitRange{Int}
    vw_dim::Int
    z::T
    dzdV::Vector{T} # divided by z
    dzdW::Vector{T}
    W_sim::Matrix{T}
    mat::Matrix{T}
    matsdim1::Matrix{T}
    matsdim2::Matrix{T}
    tempsdim::Vector{T}
    Δ2_V::Matrix{T}
    Δ2_W::Matrix{T}
    Δ3_V::Array{T, 3}
    V_fact
    W_fact
    V_λ_log::Vector{T}
    W_λ_log::Vector{T}
    V_log::Matrix{T}
    W_log::Matrix{T}
    WV_log::Matrix{T}
    dz_sqr_dV_sqr::Matrix{T}
    dz_sqr_dW_sqr::Matrix{T}
    dz_sqr_dW_dV::Matrix{T}

    function EpiTrRelEntropyTri{T}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim > 2
        @assert isodd(dim)
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.vw_dim = div(dim - 1, 2)
        cone.d = svec_side(cone.vw_dim)
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data!(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    vw_dim = cone.vw_dim
    d = cone.d
    cone.rt2 = sqrt(T(2))
    cone.V = zeros(T, d, d)
    cone.W = zeros(T, d, d)
    cone.Vi = zeros(T, d, d)
    cone.Wi = zeros(T, d, d)
    cone.V_idxs = 2:(vw_dim + 1)
    cone.W_idxs = (vw_dim + 2):cone.dim
    cone.dzdV = zeros(T, vw_dim)
    cone.dzdW = zeros(T, vw_dim)
    cone.W_sim = zeros(T, d, d)
    cone.mat = zeros(T, d, d)
    cone.matsdim1 = zeros(T, vw_dim, vw_dim)
    cone.matsdim2 = zeros(T, vw_dim, vw_dim)
    cone.tempsdim = zeros(T, vw_dim)
    cone.Δ2_V = zeros(T, d, d)
    cone.Δ2_W = zeros(T, d, d)
    cone.Δ3_V = zeros(T, d, d, d)
    cone.V_λ_log = zeros(T, d)
    cone.W_λ_log = zeros(T, d)
    cone.V_log = zeros(T, d, d)
    cone.W_log = zeros(T, d, d)
    cone.WV_log = zeros(T, d, d)
    cone.dz_sqr_dV_sqr = zeros(T, vw_dim, vw_dim)
    cone.dz_sqr_dW_sqr = zeros(T, vw_dim, vw_dim)
    cone.dz_sqr_dW_dV = zeros(T, vw_dim, vw_dim)
    return
end

get_nu(cone::EpiTrRelEntropyTri) = 2 * cone.d + 1

function set_initial_point!(
    arr::AbstractVector,
    cone::EpiTrRelEntropyTri{T},
    ) where {T <: Real}
    arr .= 0
    # at the initial point V and W are diagonal, equivalent to epirelentropy
    (arr[1], v, w) = get_central_ray_epirelentropy(cone.d)
    k = 1
    for i in 1:cone.d
        arr[1 + k] = v
        arr[cone.vw_dim + 1 + k] = w
        k += i + 1
    end
    return arr
end

function update_feas(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    @assert !cone.feas_updated
    point = cone.point

    cone.is_feas = false
    for (X, idxs) in zip((cone.V, cone.W), (cone.V_idxs, cone.W_idxs))
        @views svec_to_smat!(X, point[idxs], cone.rt2)
    end
    VH = Hermitian(cone.V, :U)
    WH = Hermitian(cone.W, :U)
    if isposdef(VH) && isposdef(WH)
        # TODO use LAPACK syev! instead of syevr! for efficiency
        V_fact = cone.V_fact = eigen(VH)
        W_fact = cone.W_fact = eigen(WH)
        if isposdef(V_fact) && isposdef(W_fact)
            for (fact, λ_log, X_log) in zip((V_fact, W_fact),
                (cone.V_λ_log, cone.W_λ_log), (cone.V_log, cone.W_log))
                (λ, vecs) = fact
                @. λ_log = log(λ)
                mul!(cone.mat, vecs, Diagonal(λ_log))
                mul!(X_log, cone.mat, vecs')
            end
            @. cone.WV_log = cone.W_log - cone.V_log
            cone.z = point[1] - dot(WH, Hermitian(cone.WV_log, :U))
            cone.is_feas = (cone.z > 0)
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    @assert cone.is_feas
    d = cone.d
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    W = Hermitian(cone.W, :U)
    z = cone.z
    (V_λ, V_vecs) = cone.V_fact
    Vi = cone.Vi = inv(cone.V_fact)
    Wi = cone.Wi = inv(cone.W_fact)

    cone.grad[1] = -inv(z)

    @views smat_to_svec!(cone.grad[W_idxs], Wi, rt2)
    dzdW = smat_to_svec!(cone.dzdW, -(cone.WV_log + I) / z, rt2)
    @. @views cone.grad[W_idxs] += dzdW
    @. @views cone.grad[W_idxs] *= -1

    Δ2_V = cone.Δ2_V
    Δ2!(Δ2_V, V_λ, cone.V_λ_log)
    W_sim = cone.W_sim = V_vecs' * W * V_vecs
    temp = V_vecs * (W_sim .* Hermitian(Δ2_V, :U)) * V_vecs' / z
    grad_V = -temp - Vi
    @views smat_to_svec!(cone.grad[V_idxs], grad_V, rt2)
    @views smat_to_svec!(cone.dzdV, temp, rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
println("start")
@time begin
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    d = cone.d
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    z = cone.z
    (V_λ, V_vecs) = cone.V_fact
    (W_λ, W_vecs) = cone.W_fact
    Vi = cone.Vi
    Wi = cone.Wi
end

println("D2")
@time begin
    Δ2!(cone.Δ2_W, W_λ, cone.W_λ_log)
    Δ2_V = Hermitian(cone.Δ2_V, :U)
    Δ2_W = Hermitian(cone.Δ2_W, :U)
    Δ3_V = Δ3!(cone.Δ3_V, Δ2_V, V_λ)
end

    W_sim = cone.W_sim
println("trvv")
    @time dz_sqr_dV_sqr = hess_tr_logm!(cone.dz_sqr_dV_sqr, V_vecs, W_sim, Δ3_V, rt2)

println("vv2")
@time begin
    @. dz_sqr_dV_sqr *= -1
    @views Hvv = H[V_idxs, V_idxs]
    symm_kron!(Hvv, Vi, rt2)
    dzdV = cone.dzdV
    mul!(Hvv, dzdV, dzdV', true, true)
    @. Hvv += dz_sqr_dV_sqr / z
end

println("ww")
@time begin
    dz_sqr_dW_sqr = grad_logm!(cone.dz_sqr_dW_sqr, W_vecs, cone.matsdim1,
        cone.matsdim2, cone.tempsdim, Δ2_W, rt2)
    @views Hww = H[W_idxs, W_idxs]
    symm_kron!(Hww, Wi, rt2)
    dzdW = cone.dzdW
    mul!(Hww, dzdW, dzdW', true, true)
    @. Hww += dz_sqr_dW_sqr / z
end

println("vw")
@time begin
    dz_sqr_dW_dV = grad_logm!(cone.dz_sqr_dW_dV, V_vecs, cone.matsdim1,
        cone.matsdim2, cone.tempsdim, Δ2_V, rt2)
    @views Hvw = H[V_idxs, W_idxs]
    @. Hvw = -dz_sqr_dW_dV / z
    mul!(Hvw, dzdV, dzdW', true, true)
end

println("H1")
@time begin
    H[1, 1] = -cone.grad[1]
    @views H[1, V_idxs] .= dzdV
    @views H[1, W_idxs] .= dzdW
    @views H[1, :] ./= z
end

    cone.hess_updated = true
    return cone.hess
end

function dder3(cone::EpiTrRelEntropyTri{T}, dir::AbstractVector{T}) where T
println("start")
@time begin
    @assert cone.hess_updated
    d = cone.d
    dder3 = cone.dder3
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    z = cone.z
    (V_λ, V_vecs) = cone.V_fact
    (W_λ, W_vecs) = cone.W_fact
    Vi = Hermitian(cone.Vi)
    Wi = Hermitian(cone.Wi)
    dzdV = cone.dzdV * z
    dzdW = cone.dzdW * z
    Δ2_V = Hermitian(cone.Δ2_V, :U)
    Δ2_W = Hermitian(cone.Δ2_W, :U)
    Δ3_V = cone.Δ3_V
    W_sim = cone.W_sim
    dz_sqr_dV_sqr = cone.dz_sqr_dV_sqr
    dlogW_dW = cone.dz_sqr_dW_sqr
    dlogV_dV = cone.dz_sqr_dW_dV

    u_dir = dir[1]
    @views v_dir = dir[V_idxs]
    @views w_dir = dir[W_idxs]
    @views V_dir = Symmetric(svec_to_smat!(
        zeros(T, d, d), dir[V_idxs], cone.rt2), :U)
    @views W_dir = Symmetric(svec_to_smat!(
        zeros(T, d, d), dir[W_idxs], cone.rt2), :U)
    V_dir_sim = V_vecs' * V_dir * V_vecs
    W_dir_sim = W_vecs' * W_dir * W_vecs

    Δ3_W = zeros(T, d, d, d)
    Δ3!(Δ3_W, Δ2_W, W_λ)

    VW_dir_sim = V_vecs' * W_dir * V_vecs
    diff_dot_V_VV = [V_dir_sim[:, q]' * Diagonal(Δ3_V[:, p, q]) *
        V_dir_sim[:, p] for p in 1:d, q in 1:d]
    d2logV_dV2_VV = V_vecs * diff_dot_V_VV * V_vecs'
    diff_dot_V_VW = [V_dir_sim[:, q]' * Diagonal(Δ3_V[:, p, q]) *
        VW_dir_sim[:, p] for p in 1:d, q in 1:d]
    diff_dot_W_WW = [W_dir_sim[:, q]' * Diagonal(Δ3_W[:, p, q]) *
        W_dir_sim[:, p] for p in 1:d, q in 1:d]

    dlogV_dV_dw = Symmetric(svec_to_smat!(
        zeros(T, d, d), dlogV_dV * w_dir, cone.rt2), :U)
    dlogV_dV_dv = Symmetric(svec_to_smat!(
        zeros(T, d, d), dlogV_dV * v_dir, cone.rt2), :U)
    dlogW_dW_dw = Symmetric(svec_to_smat!(
        zeros(T, d, d), dlogW_dW * w_dir, cone.rt2), :U)
    dz_sqr_dV_sqr_dv = Symmetric(svec_to_smat!(
        zeros(T, d, d), dz_sqr_dV_sqr * v_dir, cone.rt2), :U)
    const0 = (u_dir + dot(dzdV, v_dir) + dot(dzdW, w_dir)) / z
    const1 = abs2(const0) + (dot(v_dir, dz_sqr_dV_sqr, v_dir) / 2 +
        dot(w_dir, dlogW_dW, w_dir) / 2 - dot(v_dir, dlogV_dV, w_dir)) / z

    # u
    dder3[1] = const1
end

println("v1")
@time begin
    # v
    Δ4_ij = zeros(T, d, d)
    d3WlogVdV = zeros(T, d, d)
    Vds = V_dir_sim
    Ws = W_sim
    @inbounds for j in 1:d, i in 1:j
        Δ4_ij!(Δ4_ij, i, j, Δ3_V, V_λ)
        t = zero(T)
        for l in 1:d
            for k in 1:(l - 1)
                t += Δ4_ij[k, l] * (VdWs(i, j, k, l, Vds, Ws) + VdWs(i, j, l, k, Vds, Ws))
            end
            t += Δ4_ij[l, l] * VdWs(i, j, l, l, Vds, Ws)
        end
        d3WlogVdV[i, j] = t
    end
    copytri!(d3WlogVdV, 'U')
end

println("end")
@time begin
    V_part_1 = (dz_sqr_dV_sqr_dv - dlogV_dV_dw) * const0
    sqrt_λ = sqrt.(V_λ)
    rdiv!(V_dir_sim, Diagonal(sqrt_λ))
    ldiv!(Diagonal(V_λ), V_dir_sim)
    V_part_2a = V_dir_sim * V_dir_sim'
    V_part_2 = V_vecs * (diff_dot_V_VW + diff_dot_V_VW' +
        d3WlogVdV + V_part_2a * z) * V_vecs'
    V_dder3 = V_part_1 + V_part_2
    @views smat_to_svec!(dder3[V_idxs], V_dder3, cone.rt2)
    @. @views dder3[V_idxs] += dzdV * const1

    # w
    W_part_1 = const0 * (dlogW_dW_dw - dlogV_dV_dv) + d2logV_dV2_VV
    sqrt_λ = sqrt.(W_λ)
    rdiv!(W_dir_sim, Diagonal(sqrt_λ))
    ldiv!(Diagonal(W_λ), W_dir_sim)
    W_part_2a = W_dir_sim * W_dir_sim'
    W_part_2 = W_vecs * (W_part_2a * z - diff_dot_W_WW) * W_vecs'
    W_dder3 = W_part_1 + W_part_2
    @views smat_to_svec!(dder3[W_idxs], W_dder3, cone.rt2)
    @. @views dder3[W_idxs] += dzdW * const1

    dder3 .*= inv(z)
end

    return dder3
end

function VdWs(i, j, k, l, Vds, Ws)
    @inbounds begin
        a = Vds[l, j] * Ws[i, k] + Vds[l, i] * Ws[j, k]
        b = Vds[k, i] * Vds[l, j] * Ws[k, l]
        c = Vds[k, l] * a + b
    end
    return c
end






function Δ2!(
    Δ2::Matrix{T},
    λ::Vector{T},
    log_λ::Vector{T},
    ) where {T <: Real}
    rteps = sqrt(eps(T))
    d = length(λ)

    @inbounds for j in 1:d
        λ_j = λ[j]
        lλ_j = log_λ[j]
        for i in 1:(j - 1)
            λ_i = λ[i]
            λ_ij = λ_i - λ_j
            if abs(λ_ij) < rteps
                Δ2[i, j] = 2 / (λ_i + λ_j)
            else
                Δ2[i, j] = (log_λ[i] - lλ_j) / λ_ij
            end
        end
        Δ2[j, j] = inv(λ_j)
    end

    return Δ2
end

function Δ3!(
    Δ3::Array{T, 3},
    Δ2::AbstractMatrix{T},
    λ::Vector{T},
    ) where {T <: Real}
    rteps = sqrt(eps(T))
    d = length(λ)

    @inbounds for k in 1:d, j in 1:k, i in 1:j
        λ_j = λ[j]
        λ_k = λ[k]
        λ_jk = λ_j - λ_k
        if abs(λ_jk) < rteps
            λ_i = λ[i]
            λ_ij = λ_i - λ_j
            if abs(λ_ij) < rteps
                t = abs2(3 / (λ_i + λ_j + λ_k)) / -2
            else
                t = (Δ2[i, j] - Δ2[j, k]) / λ_ij
            end
        else
            t = (Δ2[i, j] - Δ2[i, k]) / λ_jk
        end

        Δ3[i, j, k] = Δ3[i, k, j] = Δ3[j, i, k] =
            Δ3[j, k, i] = Δ3[k, i, j] = Δ3[k, j, i] = t
    end

    return Δ3
end

function Δ4_ij!(
    Δ4_ij::Matrix,
    i::Int,
    j::Int,
    Δ3::Array{T, 3},
    λ::Vector{T},
    ) where {T <: Real}
    rteps = sqrt(eps(T))
    d = length(λ)
    λ_i = λ[i]
    λ_j = λ[j]

    @inbounds for l in 1:d, k in 1:l # TODO not over d^2 elements?
        λ_k = λ[k]
        λ_l = λ[l]
        λ_ij = λ_i - λ_j
        λ_ik = λ_i - λ_k
        λ_il = λ_i - λ_l
        B_ik = (abs(λ_ik) < rteps)
        B_il = (abs(λ_il) < rteps)

        if (abs(λ_ij) < rteps) && B_ik && B_il
            t = λ_i^-3 / 3
        elseif B_ik && B_il
            t = (Δ3[i, i, i] - Δ3[i, i, j]) / λ_ij
        elseif B_il
            t = (Δ3[i, i, j] - Δ3[i, j, k]) / λ_ik
        else
            t = (Δ3[i, j, k] - Δ3[j, k, l]) / λ_il
        end

        Δ4_ij[k, l] = t
    end

    return Δ4_ij
end

function grad_logm!(
    mat::Matrix{T},
    vecs::Matrix{T},
    tempmat1::Matrix{T},
    tempmat2::Matrix{T},
    tempvec::Vector{T},
    Δ2::AbstractMatrix{T},
    rt2::T,
    ) where {T <: Real}
    veckron = symm_kron_nonsymm!(tempmat1, vecs, rt2)
    smat_to_svec!(tempvec, Δ2, one(T))
    mul!(tempmat2, veckron, Diagonal(tempvec))
    return mul!(mat, tempmat2, veckron')
end

function hess_tr_logm!(
    mat::Matrix{T},
    vecs::Matrix{T},
    mat_inner::Matrix{T},
    Δ3::Array{T, 3},
    rt2::T,
    ) where {T <: Real}
    d = size(vecs, 1)

println("tr1")
@time begin
    X = zeros(T, d, d, d)
    @inbounds @views for i in 1:d
        X[i, :, :] = vecs * (Δ3[i, :, :] .* mat_inner) * vecs'
    end
    temp = Symmetric(zeros(T, d^2, d^2), :U)
    @inbounds @views for j in 1:d, i in 1:j
        temp.data[block_idxs(d, i), block_idxs(d, j)] = vecs *
            Diagonal(X[:, i, j]) * vecs'
    end
end

println("tr2")
@time begin
    rt2i = inv(rt2)
    row_idx = 1
    @inbounds for j in 1:d, i in 1:j
        col_idx = 1
        ijeq = (i == j)
        di = d * (i - 1)
        dj = d * (j - 1)
        for l in 1:d, k in 1:l
            rho = (k == l ? (ijeq ? T(0.5) : rt2i) : (ijeq ? rt2i : one(T)))
            dlk = d * (l - 1) + k
            dkl = d * (k - 1) + l
            dji = dj + i
            dij = di + j
            mat[row_idx, col_idx] = rho * (
                temp[dji, dlk] + temp[dij, dlk] + temp[dji, dkl] + temp[dij, dkl])
            col_idx += 1
        end
        row_idx += 1
    end
end

    return mat
end

function symm_kron_nonsymm!(
    H::AbstractMatrix{T},
    mat::AbstractMatrix{T},
    rt2::T,
    ) where {T <: Real}
    side = size(mat, 1)

    col_idx = 1
    @inbounds for l in 1:side
        for k in 1:(l - 1)
            row_idx = 1
            for j in 1:side
                for i in 1:(j - 1)
                    H[row_idx, col_idx] = mat[i, k] * mat[j, l] +
                        mat[i, l] * mat[j, k]
                    row_idx += 1
                end
                H[row_idx, col_idx] = rt2 * mat[j, k] * mat[j, l]
                row_idx += 1
            end
            col_idx += 1
        end

        row_idx = 1
        for j in 1:side
            for i in 1:(j - 1)
                H[row_idx, col_idx] = rt2 * mat[i, l] * mat[j, l]
                row_idx += 1
            end
            H[row_idx, col_idx] = abs2(mat[j, l])
            row_idx += 1
        end
        col_idx += 1
    end

    return H
end



# function Δ4!(
#     Δ4::Array{T, 4},
#     Δ3::Array{T, 3},
#     λ::Vector{T},
#     ) where {T <: Real}
#     rteps = sqrt(eps(T))
#     d = length(λ)
#
#     @inbounds for l in 1:d, k in 1:d, j in 1:d, i in 1:d # TODO not over d^4 elements
#         λ_i = λ[i]
#         λ_j = λ[j]
#         λ_k = λ[k]
#         λ_l = λ[l]
#         λ_ij = λ_i - λ_j
#         λ_ik = λ_i - λ_k
#         λ_il = λ_i - λ_l
#         B_ik = (abs(λ_ik) < rteps)
#         B_il = (abs(λ_il) < rteps)
#
#         if (abs(λ_ij) < rteps) && B_ik && B_il
#             t = λ_i^-3 / 3
#         elseif B_ik && B_il
#             t = (Δ3[i, i, i] - Δ3[i, i, j]) / λ_ij
#         elseif B_il
#             t = (Δ3[i, i, j] - Δ3[i, j, k]) / λ_ik
#         else
#             t = (Δ3[i, j, k] - Δ3[j, k, l]) / λ_il
#         end
#
#         Δ4[i, j, k, l] = t
#     end
#
#     return Δ4
# end
#
# function diff_quad!(
#     diff_quad::Matrix{T},
#     diff_tensor::Array{T, 3},
#     V_vals::Vector{T},
#     ) where T
#     rteps = sqrt(eps(T))
#     d = length(V_vals)
#     idx1 = 1
#     @inbounds for j in 1:d, i in 1:d
#         idx2 = 1
#         for l in 1:d, k in 1:d
#             (vi, vj, vk, vl) = (V_vals[i], V_vals[j], V_vals[k], V_vals[l])
#             if (abs(vi - vj) < rteps) && (abs(vi - vk) < rteps) &&
#                 (abs(vi - vl) < rteps)
#                 t = inv(vi^3) / 3 # fourth derivative divided by 3!
#             elseif (abs(vi - vl) < rteps) && (abs(vi - vk) < rteps)
#                 t = (diff_tensor[i, i, i] - diff_tensor[i, i, j]) / (vi - vj)
#             elseif (abs(vi - vl) < rteps)
#                 t = (diff_tensor[i, i, j] - diff_tensor[i, j, k]) / (vi - vk)
#             else
#                 t = (diff_tensor[j, k, l] - diff_tensor[i, j, k]) / (vl - vi)
#             end
#             diff_quad[idx1, idx2] = t
#             idx2 += 1
#         end
#         idx1 += 1
#     end
#     return diff_quad
# end
