# TODO add hess_prod, generalize for complex

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
    dder3_aux_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    rt2::T
    vw_dim::Int
    V_idxs::UnitRange{Int}
    W_idxs::UnitRange{Int}
    V::Matrix{T}
    W::Matrix{T}
    V_fact::Eigen{T}
    W_fact::Eigen{T}
    Vi::Matrix{T}
    Wi::Matrix{T}
    W_sim::Matrix{T}
    V_λ_log::Vector{T}
    W_λ_log::Vector{T}
    V_log::Matrix{T}
    W_log::Matrix{T}
    WV_log::Matrix{T}
    z::T
    Δ2_V::Matrix{T}
    Δ2_W::Matrix{T}
    Δ3_V::Array{T, 3}
    Δ3_W::Array{T, 3}
    dzdV::Vector{T}
    dzdW::Vector{T}
    d2zdV2::Matrix{T}
    d2zdW2::Matrix{T}
    d2zdVW::Matrix{T}
    mat::Matrix{T}
    mat2::Matrix{T}
    mat3::Matrix{T}
    mat4::Matrix{T}
    ten3d::Array{T, 3}
    matd2::Matrix{T}

    function EpiTrRelEntropyTri{T}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real}
        @assert dim > 2
        @assert isodd(dim)
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.vw_dim = div(dim - 1, 2)
        cone.d = svec_side(cone.vw_dim)
        return cone
    end
end

reset_data(cone::EpiTrRelEntropyTri) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated =
    cone.dder3_aux_updated = false)

function setup_extra_data!(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    vw_dim = cone.vw_dim
    d = cone.d
    cone.rt2 = sqrt(T(2))
    cone.V_idxs = 2:(vw_dim + 1)
    cone.W_idxs = (vw_dim + 2):cone.dim
    cone.V = zeros(T, d, d)
    cone.W = zeros(T, d, d)
    cone.Vi = zeros(T, d, d)
    cone.Wi = zeros(T, d, d)
    cone.W_sim = zeros(T, d, d)
    cone.V_λ_log = zeros(T, d)
    cone.W_λ_log = zeros(T, d)
    cone.V_log = zeros(T, d, d)
    cone.W_log = zeros(T, d, d)
    cone.WV_log = zeros(T, d, d)
    cone.Δ2_V = zeros(T, d, d)
    cone.Δ2_W = zeros(T, d, d)
    cone.Δ3_V = zeros(T, d, d, d)
    cone.Δ3_W = zeros(T, d, d, d)
    cone.dzdV = zeros(T, vw_dim)
    cone.dzdW = zeros(T, vw_dim)
    cone.d2zdV2 = zeros(T, vw_dim, vw_dim)
    cone.d2zdW2 = zeros(T, vw_dim, vw_dim)
    cone.d2zdVW = zeros(T, vw_dim, vw_dim)
    cone.mat = zeros(T, d, d)
    cone.mat2 = zeros(T, d, d)
    cone.mat3 = zeros(T, d, d)
    cone.mat4 = zeros(T, d, d)
    cone.ten3d = zeros(T, d, d, d)
    cone.matd2 = zeros(T, d^2, d^2)
    return
end

get_nu(cone::EpiTrRelEntropyTri) = 2 * cone.d + 1

function set_initial_point!(
    arr::AbstractVector{T},
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
                spectral_outer!(X_log, vecs, λ_log, cone.mat)
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
    rt2 = cone.rt2
    zi = inv(cone.z)
    (V_λ, V_vecs) = cone.V_fact
    (W_λ, W_vecs) = cone.W_fact
    Vi = cone.Vi
    Wi = cone.Wi
    W_sim = cone.W_sim
    Δ2_V = cone.Δ2_V
    dzdW = cone.dzdW
    mat = cone.mat
    mat2 = cone.mat2
    mat3 = cone.mat3
    g = cone.grad

    spectral_outer!(Vi, V_vecs, inv.(V_λ), cone.mat)
    spectral_outer!(Wi, W_vecs, inv.(W_λ), cone.mat)

    g[1] = -zi

    @views g_W = g[cone.W_idxs]
    smat_to_svec!(g_W, Wi, rt2)
    copyto!(mat, -zi * I)
    @. mat -= zi * cone.WV_log
    smat_to_svec!(dzdW, mat, rt2)
    axpby!(-1, dzdW, -1, g_W)

    Δ2!(Δ2_V, V_λ, cone.V_λ_log)

    spectral_outer!(W_sim, V_vecs', Symmetric(cone.W, :U), mat)
    @. mat = W_sim * Δ2_V
    spectral_outer!(mat2, V_vecs, Symmetric(mat, :U), mat3)
    @views smat_to_svec!(cone.dzdV, mat2, rt2)

    axpby!(-1, Vi, -zi, mat2)
    @views smat_to_svec!(g[cone.V_idxs], mat2, rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    zi = inv(cone.z)
    (V_λ, V_vecs) = cone.V_fact
    (W_λ, W_vecs) = cone.W_fact
    Δ2_V = cone.Δ2_V
    Δ2_W = cone.Δ2_W
    Δ3_V = cone.Δ3_V
    dzdW = cone.dzdW
    d2zdV2 = cone.d2zdV2
    d2zdW2 = cone.d2zdW2
    d2zdVW = cone.d2zdVW
    mat = cone.mat
    mat2 = cone.mat2
    mat3 = cone.mat3
    mat4 = cone.mat4
    H = cone.hess.data

    # u
    @views dzdVzi = H[V_idxs, 1]
    @. dzdVzi = zi * cone.dzdV

    H[1, 1] = abs2(zi)
    @. H[1, V_idxs] = zi * dzdVzi
    @. H[1, W_idxs] = zi * dzdW

    # vv
    Δ3!(Δ3_V, Δ2_V, V_λ)
    d2zdV2!(d2zdV2, V_vecs, cone.W_sim, Δ3_V, cone.ten3d, cone.matd2,
        mat, mat2, rt2)

    @. d2zdV2 *= -1
    @views Hvv = H[V_idxs, V_idxs]
    symm_kron!(Hvv, cone.Vi, rt2)
    mul!(Hvv, dzdVzi, dzdVzi', true, true)
    @. Hvv += zi * d2zdV2

    # vw
    eig_dot_kron!(d2zdVW, Δ2_V, V_vecs, mat, mat2, mat3, mat4, rt2)
    @views Hvw = H[V_idxs, W_idxs]
    @. Hvw = -zi * d2zdVW
    mul!(Hvw, dzdVzi, dzdW', true, true)

    # ww
    Δ2!(Δ2_W, W_λ, cone.W_λ_log)
    eig_dot_kron!(d2zdW2, Δ2_W, W_vecs, mat, mat2, mat3, mat4, rt2)
    @views Hww = H[W_idxs, W_idxs]
    symm_kron!(Hww, cone.Wi, rt2)
    mul!(Hww, dzdW, dzdW', true, true)
    @. Hww += zi * d2zdW2

    cone.hess_updated = true
    return cone.hess
end

function update_dder3_aux(cone::EpiTrRelEntropyTri)
    @assert !cone.dder3_aux_updated
    @assert cone.hess_updated
    Δ3!(cone.Δ3_W, cone.Δ2_W, cone.W_fact.values)
    cone.dder3_aux_updated = true
    return
end

function dder3(
    cone::EpiTrRelEntropyTri{T},
    dir::AbstractVector{T},
    ) where {T <: Real}
    cone.dder3_aux_updated || update_dder3_aux(cone)
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    zi = inv(cone.z)
    (V_λ, V_vecs) = cone.V_fact
    (W_λ, W_vecs) = cone.W_fact
    Δ3_V = cone.Δ3_V
    Δ3_W = cone.Δ3_W
    dzdV = cone.dzdV
    dzdW = cone.dzdW
    mat = cone.mat
    mat2 = cone.mat2
    dder3 = cone.dder3

    u_dir = dir[1]
    @views v_dir = dir[V_idxs]
    @views w_dir = dir[W_idxs]

    # TODO in-place
    Vvd = Symmetric(cone.d2zdV2, :U) * v_dir
    Wwd = Symmetric(cone.d2zdW2, :U) * w_dir
    d2zdVW = Symmetric(cone.d2zdVW, :U)
    VWwd = d2zdVW * w_dir

    const0 = zi * (u_dir + dot(v_dir, dzdV)) + dot(w_dir, dzdW)
    const1 = abs2(const0) + zi * (-dot(v_dir, VWwd) +
        (dot(v_dir, Vvd) + dot(w_dir, Wwd)) / 2)

    V_part_1a = const0 * (Vvd - VWwd)
    W_part_1a = Wwd - d2zdVW * v_dir

    # u
    ziconst1 = dder3[1] = zi * const1

    # v, w
    # TODO prealloc
    V_dir = Symmetric(zero(mat), :U)
    W_dir = Symmetric(zero(mat), :U)
    V_dir_sim = zero(mat)
    W_dir_sim = zero(mat)
    VW_dir_sim = zero(mat)
    W_part_1 = zero(mat)
    V_part_1 = zero(mat)
    d3WlogVdV = zero(mat)
    diff_dot_V_VV = zero(mat)
    diff_dot_V_VW = zero(mat)
    diff_dot_W_WW = zero(mat)

    svec_to_smat!(V_dir.data, v_dir, rt2)
    svec_to_smat!(W_dir.data, w_dir, rt2)
    spectral_outer!(V_dir_sim, V_vecs', V_dir, mat)
    spectral_outer!(W_dir_sim, W_vecs', W_dir, mat)
    spectral_outer!(VW_dir_sim, V_vecs', W_dir, mat)

    @inbounds @views for j in 1:cone.d
        Vds_j = V_dir_sim[:, j]
        Wds_j = W_dir_sim[:, j]
        for i in 1:j
            Vds_i = V_dir_sim[:, i]
            Wds_i = W_dir_sim[:, i]
            DΔ3_V_ij = Diagonal(Δ3_V[:, i, j])
            DΔ3_W_ij = Diagonal(Δ3_W[:, i, j])
            diff_dot_V_VV[i, j] = dot(Vds_j, DΔ3_V_ij, Vds_i)
            diff_dot_W_WW[i, j] = dot(Wds_j, DΔ3_W_ij, Wds_i)
        end
        for i in 1:cone.d
            VWds_i = VW_dir_sim[:, i]
            DΔ3_V_ij = Diagonal(Δ3_V[:, i, j])
            diff_dot_V_VW[i, j] = dot(Vds_j, DΔ3_V_ij, VWds_i)
        end
    end

    # v
    d3WlogVdV!(d3WlogVdV, Δ3_V, V_λ, V_dir_sim, cone.W_sim, mat)
    svec_to_smat!(V_part_1, V_part_1a, rt2)
    rdiv!(V_dir_sim, Diagonal(sqrt.(V_λ)))
    ldiv!(Diagonal(V_λ), V_dir_sim)
    V_part_2 = d3WlogVdV
    @. V_part_2 += diff_dot_V_VW + diff_dot_V_VW'
    mul!(V_part_2, V_dir_sim, V_dir_sim', true, zi)
    mul!(mat, Symmetric(V_part_2, :U), V_vecs')
    mul!(V_part_1, V_vecs, mat, true, zi)
    @views dder3_V = dder3[V_idxs]
    smat_to_svec!(dder3_V, V_part_1, rt2)
    @. dder3_V += ziconst1 * dzdV

    # w
    svec_to_smat!(W_part_1, W_part_1a, rt2)
    spectral_outer!(mat2, V_vecs, Symmetric(diff_dot_V_VV, :U), mat)
    axpby!(true, mat2, const0, W_part_1)
    rdiv!(W_dir_sim, Diagonal(sqrt.(W_λ)))
    ldiv!(Diagonal(W_λ), W_dir_sim)
    W_part_2 = diff_dot_W_WW
    mul!(W_part_2, W_dir_sim, W_dir_sim', true, -zi)
    mul!(mat, Symmetric(W_part_2, :U), W_vecs')
    mul!(W_part_1, W_vecs, mat, true, zi)
    @views dder3_W = dder3[W_idxs]
    smat_to_svec!(dder3_W, W_part_1, rt2)
    @. dder3_W += const1 * dzdW

    return dder3
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

    # make symmetric
    copytri!(Δ2, 'U')
    return Δ2
end

function Δ3!(
    Δ3::Array{T, 3},
    Δ2::Matrix{T},
    λ::Vector{T},
    ) where {T <: Real}
    @assert issymmetric(Δ2) # must be symmetric (wrapper is less efficient)
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

function d2zdV2!(
    d2zdV2::Matrix{T},
    vecs::Matrix{T},
    inner::Matrix{T},
    Δ3::Array{T, 3},
    ten3d::Array{T, 3}, # temp
    matd2::Matrix{T}, # temp
    mat::Matrix{T}, # temp
    mat2::Matrix{T}, # temp
    rt2::T,
    ) where {T <: Real}
    d = size(vecs, 1)

    @inbounds for i in 1:d
        @views ten3d_i = ten3d[i, :, :]
        @views Δ3_i = Δ3[i, :, :]
        @. mat2 = inner * Δ3_i
        spectral_outer!(ten3d_i, vecs, Symmetric(mat2, :U), mat)
    end
    @inbounds for j in 1:d, i in 1:j
        @views temp_ij = matd2[block_idxs(d, i), block_idxs(d, j)]
        @views D_ij = ten3d[:, i, j]
        spectral_outer!(temp_ij, vecs, D_ij, mat)
    end
    copytri!(matd2, 'U')

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
            d2zdV2[row_idx, col_idx] = rho * (
                matd2[dji, dlk] + matd2[dij, dlk] +
                matd2[dji, dkl] + matd2[dij, dkl])
            col_idx += 1
        end
        row_idx += 1
    end

    return d2zdV2
end

function VdWs_element(
    i::Int,
    j::Int,
    k::Int,
    l::Int,
    Vds::Matrix{T},
    Ws::Matrix{T},
    ) where {T <: Real}
    @inbounds begin
        a = Vds[l, j] * Ws[i, k] + Vds[l, i] * Ws[j, k]
        b = Vds[k, i] * Vds[l, j] * Ws[k, l]
        c = Vds[k, l] * a + b
    end
    return c
end

function d3WlogVdV!(
    d3WlogVdV::Matrix{T},
    Δ3::Array{T, 3},
    λ::Vector{T},
    Vds::Matrix{T},
    Ws::Matrix{T},
    Δ4_ij::Matrix{T}, # temp
    ) where {T <: Real}
    d = length(λ)

    @inbounds for j in 1:d, i in 1:j
        Δ4_ij!(Δ4_ij, i, j, Δ3, λ)
        t = zero(T)
        for l in 1:d
            for k in 1:(l - 1)
                t += Δ4_ij[k, l] * (VdWs_element(i, j, k, l, Vds, Ws) +
                    VdWs_element(i, j, l, k, Vds, Ws))
            end
            t += Δ4_ij[l, l] * VdWs_element(i, j, l, l, Vds, Ws)
        end
        d3WlogVdV[i, j] = t
    end

    copytri!(d3WlogVdV, 'U')
    return d3WlogVdV
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

    @inbounds for l in 1:d, k in 1:l
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
