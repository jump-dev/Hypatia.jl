#=
(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle) symmetric positive define matrix with side dimension d
(u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeTriangle definition)

barrier analogous to hypoperlog cone
-log(v*logdet(W/v) - u) - logdet(W) - log(v)

TODO
- describe complex case
=#

mutable struct HypoPerLogdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    is_complex::Bool
    rt2::T

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
    hess_aux_updated::Bool
    inv_hess_aux_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    fact_W
    lwv::T
    z::T
    W::Matrix{R}
    Wi::Matrix{R}
    Wi_vec::Vector{T}
    tempw::Vector{T}

    function HypoPerLogdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 3
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        if R <: Complex
            d = isqrt(dim - 2) # real lower triangle and imaginary under diagonal
            @assert d^2 == dim - 2
            cone.is_complex = true
        else
            d = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
            @assert d * (d + 1) == 2 * (dim - 2)
            cone.is_complex = false
        end
        cone.d = d
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

use_heuristic_neighborhood(cone::HypoPerLogdetTri) = false

reset_data(cone::HypoPerLogdetTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated = cone.inv_hess_aux_updated = cone.hess_fact_updated = false)

function setup_extra_data(cone::HypoPerLogdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    d = cone.d
    cone.mat = zeros(R, d, d)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.W = zeros(R, d, d)
    cone.Wi_vec = zeros(T, dim - 2)
    cone.tempw = zeros(T, dim - 2)
    return cone
end

get_nu(cone::HypoPerLogdetTri) = cone.d + 2

function set_initial_point(arr::AbstractVector{T}, cone::HypoPerLogdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    # central point data are the same as for hypoperlog
    (arr[1], arr[2], w) = get_central_ray_hypoperlog(cone.d)
    incr = (cone.is_complex ? 2 : 1)
    k = 3
    @inbounds for i in 1:cone.d
        arr[k] = w
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoPerLogdetTri{T}) where T
    @assert !cone.feas_updated
    v = cone.point[2]

    if v > eps(T)
        u = cone.point[1]
        @views svec_to_smat!(cone.mat, cone.point[3:end], cone.rt2)
        fact = cone.fact_W = cholesky!(Hermitian(cone.mat, :U), check = false)
        if isposdef(fact)
            cone.lwv = logdet(fact) - cone.d * log(v)
            cone.z = v * cone.lwv - u
            cone.is_feas = (cone.z > eps(T))
        else
            cone.is_feas = false
        end
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoPerLogdetTri{T}) where T
    u = cone.dual_point[1]
    if u < -eps(T)
        v = cone.dual_point[2]
        @views svec_to_smat!(cone.mat2, cone.dual_point[3:end], cone.rt2)
        fact = cholesky!(Hermitian(cone.mat2, :U), check = false)
        return isposdef(fact) && (v - u * (logdet(fact) + cone.d * (1 - log(-u))) > eps(T))
    end
    return false
end

function update_grad(cone::HypoPerLogdetTri)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    g = cone.grad
    z = cone.z

    g[1] = inv(z)
    g[2] = (cone.d - cone.lwv) / z - inv(v)
    # TODO in-place
    # copyto!(cone.Wi, cone.fact_W.factors)
    # LinearAlgebra.inv!(Cholesky(cone.Wi, 'U', 0))
    cone.Wi = inv(cone.fact_W)
    smat_to_svec!(cone.Wi_vec, cone.Wi, cone.rt2)
    zvzi = -(z + v) / z
    @inbounds @. @views g[3:end] = zvzi * cone.Wi_vec

    cone.grad_updated = true
    return cone.grad
end

# update first two rows of the Hessian
function update_hess_aux(cone::HypoPerLogdetTri)
    @assert cone.grad_updated
    @assert !cone.hess_aux_updated
    v = cone.point[2]
    H = cone.hess.data
    z = cone.z
    d = cone.d
    Wi_vec = cone.Wi_vec
    dlzi = (d - cone.lwv) / z

    @inbounds begin
        H[1, 1] = abs2(inv(z))
        H[1, 2] = H[2, 1] = dlzi / z
        H13const = -v / z / z
        @. @views H[1, 3:end] = H13const * Wi_vec

        H[2, 2] = abs2(dlzi) + (d / z + inv(v)) / v
        H23const = ((cone.lwv - d) * v / z - 1) / z
        @. @views H[2, 3:end] = H23const * Wi_vec
    end

    cone.hess_aux_updated = true
    return
end

function update_hess(cone::HypoPerLogdetTri)
    if !cone.hess_aux_updated
        update_hess_aux(cone)
    end
    v = cone.point[2]
    H = cone.hess.data
    z = cone.z
    Wi_vec = cone.Wi_vec
    zvzi = (z + v) / z
    vzi = zvzi - 1
    Wivzi = cone.tempw
    @. Wivzi = vzi * Wi_vec

    @inbounds @views symm_kron(H[3:end, 3:end], cone.Wi, cone.rt2)
    @inbounds for j in eachindex(Wi_vec)
        j2 = 2 + j
        Wivzij = Wivzi[j]
        for i in 1:j
            H[2 + i, j2] = zvzi * H[2 + i, j2] + Wivzi[i] * Wivzij
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLogdetTri)
    if !cone.hess_aux_updated
        update_hess_aux(cone)
    end
    v = cone.point[2]
    H = cone.hess.data
    z = cone.z
    const_diag = v / z * v / (z + v)

    @inbounds @views mul!(prod[1:2, :], H[1:2, :], arr)
    @inbounds for i in 1:size(arr, 2)
        @views svec_to_smat!(cone.mat2, arr[3:end, i], cone.rt2)
        copytri!(cone.mat2, 'U', cone.is_complex)
        rdiv!(cone.mat2, cone.fact_W)
        const_i = tr(cone.mat2) * const_diag
        for j in 1:cone.d
            cone.mat2[j, j] += const_i
        end
        ldiv!(cone.fact_W, cone.mat2)
        @views smat_to_svec!(prod[3:end, i], cone.mat2, cone.rt2)
    end
    @inbounds @views mul!(prod[3:end, :], H[1:2, 3:end]', arr[1:2, :], true, (z + v) / z)

    return prod
end

# update first two rows of the inverse Hessian
function update_inv_hess_aux(cone::HypoPerLogdetTri)
    @assert cone.feas_updated
    @assert !cone.inv_hess_aux_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    svec_to_smat!(cone.W, w, cone.rt2)
    Hi = cone.inv_hess.data
    d = cone.d
    z = cone.z
    zv = z + v
    zuz = 2 * z + u
    den = zv + d * v
    vden = v / den
    zuzvden = zuz * vden
    vvden = v * vden

    @inbounds begin
        Hi[1, 1] = abs2(z + u) + z * (den - v) - d * zuz * zuzvden
        Hi[1, 2] = Hi[2, 1] = vvden * (cone.lwv * (zv - d * v) + d * u)
        @. @views Hi[1, 3:end] = zuzvden * w

        Hi[2, 2] = vvden * zv
        @. @views Hi[2, 3:end] = vvden * w
    end

    cone.inv_hess_aux_updated = true
    return
end

function update_inv_hess(cone::HypoPerLogdetTri)
    if !cone.inv_hess_aux_updated
        update_inv_hess_aux(cone)
    end
    v = cone.point[2]
    @views w = cone.point[3:end]
    W = Hermitian(cone.W, :U)
    Hi = cone.inv_hess.data
    z = cone.z
    zv = z + v
    zzv = z / zv
    zvden = zv * (zv + cone.d * v)
    wv = cone.tempw
    @. wv = w * v

    @inbounds @views symm_kron(Hi[3:end, 3:end], W, cone.rt2)
    @inbounds for j in eachindex(wv)
        j2 = 2 + j
        wvjden = wv[j] / zvden
        for i in 1:j
            Hi[2 + i, j2] = zzv * Hi[2 + i, j2] + wv[i] * wvjden
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLogdetTri)
    if !cone.inv_hess_aux_updated
        update_inv_hess_aux(cone)
    end
    v = cone.point[2]
    @views w = cone.point[3:end]
    W = Hermitian(cone.W, :U)
    Hi = cone.inv_hess.data
    d = cone.d
    z = cone.z
    zv = z + v
    const_w = v / (zv + d * v) * v / z

    @inbounds @views mul!(prod[1:2, :], Hi[1:2, :], arr)
    @inbounds for i in 1:size(arr, 2)
        @views arr_w = arr[3:end, i]
        @views prod_w = prod[3:end, i]
        svec_to_smat!(cone.mat2, arr_w, cone.rt2)
        copytri!(cone.mat2, 'U', cone.is_complex)
        mul!(cone.mat3, cone.mat2, W)
        mul!(cone.mat2, W, cone.mat3)
        smat_to_svec!(prod_w, cone.mat2, cone.rt2)
        const_i = const_w * dot(w, arr_w)
        @. prod_w += const_i * w
    end
    @inbounds @views mul!(prod[3:end, :], Hi[1:2, 3:end]', arr[1:2, :], true, z / zv)

    return prod
end

function correction(cone::HypoPerLogdetTri, primal_dir::AbstractVector)
    @assert cone.grad_updated
    u_dir = primal_dir[1]
    v_dir = primal_dir[2]
    @views w_dir = primal_dir[3:end]
    corr = cone.correction
    @views w_corr = corr[3:end]
    v = cone.point[2]
    z = cone.z
    tempw = cone.tempw
    Wi_vec = cone.Wi_vec
    d = cone.d
    dlzi = (d - cone.lwv) / z
    vz = v / z
    udz = u_dir / z
    vdz = v_dir / z

    dot_Wi_S = dot(Wi_vec, w_dir)
    S = copytri!(svec_to_smat!(cone.mat2, w_dir, cone.rt2), 'U', cone.is_complex)
    ldiv!(cone.fact_W, S)
    dot_skron = real(dot(S, S'))

    const6 = 1 + v * dlzi
    uuw_scal = -2 * udz * vz / z
    vvw_scal = -vdz * (2 * const6 * dlzi + d / z)
    uvw_scal = -2 * (2 * vz * dlzi + inv(z)) / z
    const8 = -2 * (1 + vz)
    const10 = 2 * (vz * udz + vdz * const6)
    const9 = -2 * abs2(vz) * dot_Wi_S + const10
    const7 = uuw_scal + uvw_scal * v_dir
    const11 = -vz * (vz * dot_skron + 2 * (abs2(vz * dot_Wi_S) - dot_Wi_S * const10)) + u_dir * const7 + vvw_scal * v_dir

    @. w_corr = const11 * Wi_vec
    rdiv!(S, cone.fact_W.U)
    mul!(cone.mat3, S, S')
    vec_S2 = smat_to_svec!(tempw, cone.mat3, cone.rt2)
    @. w_corr += const8 * vec_S2
    skron2 = rdiv!(S, cone.fact_W.U')
    vec_skron2 = smat_to_svec!(tempw, skron2, cone.rt2)
    t4awd = (2 * vz * abs2(dot_Wi_S) + dot(vec_skron2, w_dir)) / z
    @. w_corr += const9 * vec_skron2

    const2 = 2 * udz * dlzi / z
    const3 = (2 * abs2(dlzi) + d / z / v) * vdz
    const5 = d / v / z
    const4 = dlzi * (2 * abs2(dlzi) + 3 * const5) - (const5 + 2 * inv(v) / v) / v
    corr[1] = 2 * abs2(udz) / z + (2 * const2 + const3) * v_dir + (uuw_scal + const7) * dot_Wi_S + vz * t4awd
    corr[2] = const4 * abs2(v_dir) + (2 * vvw_scal + uvw_scal * u_dir) * dot_Wi_S + u_dir * (const2 + 2 * const3) + const6 * t4awd

    corr ./= -2

    return corr
end
