#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

"""
$(TYPEDEF)

Epigraph of matrix relative entropy cone of dimension `dim` in svec format.

    $(FUNCTIONNAME){T}(dim::Int, use_dual::Bool = false)
"""
mutable struct EpiTrRelEntropyTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    is_complex::Bool

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    hess_aux_updated::Bool
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
    V::Matrix{R}
    W::Matrix{R}
    V_fact::Eigen{R}
    W_fact::Eigen{R}
    Vi::Matrix{R}
    Wi::Matrix{R}
    W_sim::Matrix{R}
    V_λ_log::Vector{T}
    W_λ_log::Vector{T}
    V_log::Matrix{R}
    W_log::Matrix{R}
    WV_log::Matrix{R}
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
    tempvec::Vector{T}
    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}
    ten3d::Array{R, 3}

    function EpiTrRelEntropyTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
    ) where {T <: Real, R <: RealOrComplex{T}}
        @assert dim > 2
        @assert isodd(dim)
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.is_complex = (R <: Complex)
        cone.vw_dim = div(dim - 1, 2)
        cone.d = svec_side(R, cone.vw_dim)
        return cone
    end
end

function reset_data(cone::EpiTrRelEntropyTri)
    return (
        cone.feas_updated =
            cone.grad_updated =
                cone.hess_updated =
                    cone.hess_aux_updated =
                        cone.inv_hess_updated =
                            cone.hess_fact_updated = cone.dder3_aux_updated = false
    )
end

function setup_extra_data!(
    cone::EpiTrRelEntropyTri{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    vw_dim = cone.vw_dim
    d = cone.d
    cone.rt2 = sqrt(T(2))
    cone.V_idxs = 2:(vw_dim + 1)
    cone.W_idxs = (vw_dim + 2):(cone.dim)
    cone.V = zeros(R, d, d)
    cone.W = zeros(R, d, d)
    cone.Vi = zeros(R, d, d)
    cone.Wi = zeros(R, d, d)
    cone.W_sim = zeros(R, d, d)
    cone.V_λ_log = zeros(T, d)
    cone.W_λ_log = zeros(T, d)
    cone.V_log = zeros(R, d, d)
    cone.W_log = zeros(R, d, d)
    cone.WV_log = zeros(R, d, d)
    cone.Δ2_V = zeros(T, d, d)
    cone.Δ2_W = zeros(T, d, d)
    cone.Δ3_V = zeros(T, d, d, d)
    cone.Δ3_W = zeros(T, d, d, d)
    cone.dzdV = zeros(T, vw_dim)
    cone.dzdW = zeros(T, vw_dim)
    cone.d2zdV2 = zeros(T, vw_dim, vw_dim)
    cone.d2zdW2 = zeros(T, vw_dim, vw_dim)
    cone.d2zdVW = zeros(T, vw_dim, vw_dim)
    cone.tempvec = zeros(T, vw_dim)
    cone.mat = zeros(R, d, d)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.mat4 = zeros(R, d, d)
    cone.ten3d = zeros(R, d, d, d)
    return
end

get_nu(cone::EpiTrRelEntropyTri) = 2 * cone.d + 1

function set_initial_point!(
    arr::AbstractVector{T},
    cone::EpiTrRelEntropyTri{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    # at the initial point V and W are diagonal, equivalent to epirelentropy
    (arr[1], v, w) = get_central_ray_epirelentropy(cone.d)
    k = 1
    for i in 1:(cone.d)
        arr[1 + k] = v
        arr[cone.vw_dim + 1 + k] = w
        k += incr * i + 1
    end
    return arr
end

function update_feas(
    cone::EpiTrRelEntropyTri{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
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
            for (fact, λ_log, X_log) in zip(
                (V_fact, W_fact),
                (cone.V_λ_log, cone.W_λ_log),
                (cone.V_log, cone.W_log),
            )
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

function update_grad(
    cone::EpiTrRelEntropyTri{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
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

    spectral_outer!(W_sim, V_vecs', Hermitian(cone.W, :U), mat)
    @. mat = W_sim * Δ2_V
    spectral_outer!(mat2, V_vecs, Hermitian(mat, :U), mat3)
    @. mat2 *= zi
    @views smat_to_svec!(cone.dzdV, mat2, rt2)

    axpby!(-1, Vi, -1, mat2)
    @views smat_to_svec!(g[cone.V_idxs], mat2, rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiTrRelEntropyTri)
    @assert cone.grad_updated
    Δ2!(cone.Δ2_W, cone.W_fact.values, cone.W_λ_log)
    Δ3!(cone.Δ3_V, cone.Δ2_V, cone.V_fact.values)

    cone.hess_aux_updated = true
    return cone.hess_aux_updated
end

function update_hess(cone::EpiTrRelEntropyTri)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    zi = inv(cone.z)
    V_vecs = cone.V_fact.vectors
    W_vecs = cone.W_fact.vectors
    Δ2_V = cone.Δ2_V
    Δ2_W = cone.Δ2_W
    Δ3_V = cone.Δ3_V
    dzdV = cone.dzdV
    dzdW = cone.dzdW
    d2zdV2 = cone.d2zdV2
    d2zdW2 = cone.d2zdW2
    d2zdVW = cone.d2zdVW
    mat = cone.mat
    mat2 = cone.mat2
    mat3 = cone.mat3
    H = cone.hess.data

    # u
    H[1, 1] = abs2(zi)
    @. H[1, V_idxs] = zi * dzdV
    @. H[1, W_idxs] = zi * dzdW

    # vv
    d2zdV2!(d2zdV2, V_vecs, cone.W_sim, Δ3_V, cone.ten3d, mat, mat2, mat3, rt2)

    @. d2zdV2 *= -1
    @views Hvv = H[V_idxs, V_idxs]
    symm_kron!(Hvv, cone.Vi, rt2)
    mul!(Hvv, dzdV, dzdV', true, true)
    @. Hvv += zi * d2zdV2

    # vw
    eig_dot_kron!(d2zdVW, Δ2_V, V_vecs, mat, mat2, mat3, rt2)
    @views Hvw = H[V_idxs, W_idxs]
    @. Hvw = -zi * d2zdVW
    mul!(Hvw, dzdV, dzdW', true, true)

    # ww
    eig_dot_kron!(d2zdW2, Δ2_W, W_vecs, mat, mat2, mat3, rt2)
    @views Hww = H[W_idxs, W_idxs]
    symm_kron!(Hww, cone.Wi, rt2)
    mul!(Hww, dzdW, dzdW', true, true)
    @. Hww += zi * d2zdW2

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiTrRelEntropyTri)
    cone.hess_aux_updated || update_hess_aux(cone)
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    dzdW = cone.dzdW
    dzdV = cone.dzdV
    z = cone.z
    tempvec = cone.tempvec
    temp = cone.mat
    temp2 = cone.mat2
    Varr_simV = cone.mat3
    arr_W_mat = cone.mat4
    (V_λ, V_vecs) = cone.V_fact
    (W_λ, W_vecs) = cone.W_fact
    Δ3 = cone.Δ3_V

    for i in 1:size(arr, 2)
        @views V_arr = arr[V_idxs, i]
        @views W_arr = arr[W_idxs, i]

        const1 = arr[1, i] / z + dot(dzdV, V_arr) + dot(dzdW, W_arr)
        prod[1, i] = const1 / z

        @views @. prod[W_idxs, i] = dzdW * const1
        # Hwv * a_v
        arr_V_mat = svec_to_smat!(temp, V_arr, rt2)
        spectral_outer!(Varr_simV, V_vecs', Hermitian(arr_V_mat, :U), temp2)
        @. temp = Varr_simV * cone.Δ2_V
        spectral_outer!(temp, V_vecs, Hermitian(temp, :U), temp2)
        @. temp /= -z
        @views prod[W_idxs, i] .+= smat_to_svec!(tempvec, temp, rt2)
        # Hww * a_w
        svec_to_smat!(arr_W_mat, W_arr, rt2)
        Warr_simW = spectral_outer!(temp2, W_vecs', Hermitian(arr_W_mat, :U), temp)
        @. temp = Warr_simW * cone.Δ2_W / z
        @. Warr_simW /= W_λ'
        ldiv!(Diagonal(W_λ), Warr_simW)
        @. temp += Warr_simW
        spectral_outer!(temp, W_vecs, Hermitian(temp, :U), temp2)
        @views prod[W_idxs, i] .+= smat_to_svec!(tempvec, temp, rt2)

        @views @. prod[V_idxs, i] = dzdV * const1
        # Hvv * a_v
        for k in 1:(cone.d)
            @views @. temp = cone.W_sim * Δ3[:, :, k]
            @views mul!(temp2[:, k], temp, Varr_simV[:, k])
        end
        @. temp = temp2 + temp2'
        # destroys arr_W_mat
        Warr_simV = spectral_outer!(arr_W_mat, V_vecs', Hermitian(arr_W_mat, :U), temp2)
        @. temp += Warr_simV * cone.Δ2_V
        @. temp /= -z
        @. Varr_simV /= V_λ'
        ldiv!(Diagonal(V_λ), Varr_simV)
        @. temp += Varr_simV
        spectral_outer!(temp, V_vecs, Hermitian(temp, :U), temp2)
        @views prod[V_idxs, i] .+= smat_to_svec!(tempvec, temp, rt2)
    end

    return prod
end

function update_dder3_aux(cone::EpiTrRelEntropyTri)
    @assert !cone.dder3_aux_updated
    @assert cone.hess_updated
    Δ3!(cone.Δ3_W, cone.Δ2_W, cone.W_fact.values)
    cone.dder3_aux_updated = true
    return
end

function dder3(
    cone::EpiTrRelEntropyTri{T, R},
    dir::AbstractVector{T},
) where {T <: Real, R <: RealOrComplex{T}}
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

    const0 = zi * u_dir + dot(v_dir, dzdV) + dot(w_dir, dzdW)
    const1 =
        abs2(const0) + zi * (-dot(v_dir, VWwd) + (dot(v_dir, Vvd) + dot(w_dir, Wwd)) / 2)

    V_part_1a = const0 * (Vvd - VWwd)
    W_part_1a = Wwd - d2zdVW * v_dir

    # u
    dder3[1] = zi * const1

    # v, w
    # TODO prealloc
    V_dir = Hermitian(zero(mat), :U)
    W_dir = Hermitian(zero(mat), :U)
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

    @inbounds @views for j in 1:(cone.d)
        Vds_j = V_dir_sim[:, j]
        Wds_j = W_dir_sim[:, j]
        for i in 1:j
            Vds_i = V_dir_sim[:, i]
            Wds_i = W_dir_sim[:, i]
            DΔ3_V_ij = Diagonal(Δ3_V[:, i, j])
            DΔ3_W_ij = Diagonal(Δ3_W[:, i, j])
            diff_dot_V_VV[i, j] = dot(Vds_i, DΔ3_V_ij, Vds_j)
            diff_dot_W_WW[i, j] = dot(Wds_i, DΔ3_W_ij, Wds_j)
        end
        for i in 1:(cone.d)
            VWds_i = VW_dir_sim[:, i]
            DΔ3_V_ij = Diagonal(Δ3_V[:, i, j])
            diff_dot_V_VW[i, j] = dot(VWds_i, DΔ3_V_ij, Vds_j)
        end
    end

    # v
    d3WlogVdV!(d3WlogVdV, Δ3_V, V_λ, V_dir_sim, cone.W_sim, mat)
    svec_to_smat!(V_part_1, V_part_1a, rt2)
    @. V_dir_sim /= sqrt(V_λ)'
    ldiv!(Diagonal(V_λ), V_dir_sim)
    V_part_2 = d3WlogVdV
    @. V_part_2 += diff_dot_V_VW + diff_dot_V_VW'
    mul!(V_part_2, V_dir_sim, V_dir_sim', true, zi)
    mul!(mat, Hermitian(V_part_2, :U), V_vecs')
    mul!(V_part_1, V_vecs, mat, true, zi)
    @views dder3_V = dder3[V_idxs]
    smat_to_svec!(dder3_V, V_part_1, rt2)
    @. dder3_V += const1 * dzdV

    # w
    svec_to_smat!(W_part_1, W_part_1a, rt2)
    spectral_outer!(mat2, V_vecs, Hermitian(diff_dot_V_VV, :U), mat)
    axpby!(true, mat2, const0, W_part_1)
    @. W_dir_sim /= sqrt(W_λ)'
    ldiv!(Diagonal(W_λ), W_dir_sim)
    W_part_2 = diff_dot_W_WW
    mul!(W_part_2, W_dir_sim, W_dir_sim', true, -zi)
    mul!(mat, Hermitian(W_part_2, :U), W_vecs')
    mul!(W_part_1, W_vecs, mat, true, zi)
    @views dder3_W = dder3[W_idxs]
    smat_to_svec!(dder3_W, W_part_1, rt2)
    @. dder3_W += const1 * dzdW

    return dder3
end

function Δ2!(Δ2::Matrix{T}, λ::Vector{T}, log_λ::Vector{T}) where {T <: Real}
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

function Δ3!(Δ3::Array{T, 3}, Δ2::Matrix{T}, λ::Vector{T}) where {T <: Real}
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

        Δ3[i, j, k] =
            Δ3[i, k, j] = Δ3[j, i, k] = Δ3[j, k, i] = Δ3[k, i, j] = Δ3[k, j, i] = t
    end

    return Δ3
end

# TODO consider refactor with eig_dot_kron!
function d2zdV2!(
    d2zdV2::Matrix{T},
    vecs::Matrix{R},
    inner::Matrix{R},
    Δ3::Array{T, 3},
    ten3d::Array{R, 3}, # temp
    mat::Matrix{R}, # temp
    mat2::Matrix{R}, # temp
    mat3::Matrix{R}, # temp
    rt2::T,
) where {T <: Real, R <: RealOrComplex{T}}
    d = size(vecs, 1)
    V = copyto!(mat, vecs')
    V_views = [view(V, :, i) for i in 1:d]
    rt2i = inv(rt2)
    scals = (R <: Complex{T} ? [rt2i, rt2i * im] : [rt2i])
    @inbounds for k in 1:d
        @. @views ten3d[:, :, k] = inner * Δ3[:, :, k]
    end

    col_idx = 1
    @inbounds for j in 1:d
        for i in 1:(j - 1), scal in scals
            mul!(mat3, V_views[j], V_views[i]', scal, false)
            @. mat2 = mat3 + mat3'
            for k in 1:d
                @views mul!(mat3[:, k], ten3d[:, :, k], mat2[:, k])
            end
            # mat2 = vecs * (mat3 + mat3) * vecs'
            @. mat2 = mat3 + mat3'
            mul!(mat3, Hermitian(mat2, :U), V)
            mul!(mat2, V', mat3)
            @views smat_to_svec!(d2zdV2[:, col_idx], mat2, rt2)
            col_idx += 1
        end

        mul!(mat2, V_views[j], V_views[j]')
        for k in 1:d
            @views mul!(mat3[:, k], ten3d[:, :, k], mat2[:, k])
        end
        @. mat2 = mat3 + mat3'
        mul!(mat3, Hermitian(mat2, :U), V)
        mul!(mat2, V', mat3)
        @views smat_to_svec!(d2zdV2[:, col_idx], mat2, rt2)
        col_idx += 1
    end
    return d2zdV2
end

function VdWs_element(
    i::Int,
    j::Int,
    k::Int,
    l::Int,
    Vds::Matrix{R},
    Ws::Matrix{R},
) where {T <: Real, R <: RealOrComplex{T}}
    @inbounds begin
        a = Ws[i, k] * Vds[k, l] + Vds[i, k] * Ws[k, l]
        b = Vds[i, k] * Vds[k, l] * Ws[l, j]
        c = Vds[l, j] * a + b
    end
    return c
end

function d3WlogVdV!(
    d3WlogVdV::Matrix{R},
    Δ3::Array{T, 3},
    λ::Vector{T},
    Vds::Matrix{R},
    Ws::Matrix{R},
    Δ4_ij::Matrix{R}, # temp TODO could have been a Matrix{T}
) where {T <: Real, R <: RealOrComplex{T}}
    d = length(λ)

    @inbounds for j in 1:d, i in 1:j
        Δ4_ij!(Δ4_ij, i, j, Δ3, λ)
        t = zero(R)
        for l in 1:d
            for k in 1:(l - 1)
                t +=
                    Δ4_ij[k, l] *
                    (VdWs_element(i, j, k, l, Vds, Ws) + VdWs_element(i, j, l, k, Vds, Ws))
            end
            t += Δ4_ij[l, l] * VdWs_element(i, j, l, l, Vds, Ws)
        end
        d3WlogVdV[i, j] = t
    end

    copytri!(d3WlogVdV, 'U', true)
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
