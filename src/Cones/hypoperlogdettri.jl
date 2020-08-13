#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle) symmetric positive define matrix
(u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeTriangle definition)

barrier (self-concordance follows from theorem 5.1.4, "Interior-Point Polynomial Algorithms in Convex Programming" by Y. Nesterov and A. Nemirovski):
theta^2 * (-log(v*logdet(W/v) - u) - logdet(W) - (n + 1) log(v))
we use theta = 16

TODO
- describe complex case
- try to tune theta parameter
- investigate numerics in inverse Hessian oracles
=#

mutable struct HypoPerLogdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    side::Int
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}
    rt2::T
    sc_const::T
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    inv_hess_prod_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    mat::Matrix{R}
    dual_mat::Matrix{R}
    mat2::Matrix{R} # TODO named differently in some cones, fix inconsistency
    mat3::Matrix{R}
    mat4::Matrix{R}
    fact_mat
    ldWv::T
    z::T
    Wi::Matrix{R}
    nLz::T
    ldWvuv::T
    vzip1::T
    Wivzi::Matrix{R}
    tmpw::Vector{T}

    function HypoPerLogdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        # sc_const::Real = 256, # TODO reduce this
        sc_const::Real = 25 / T(9), # NOTE not SC but works well (same as rootdet)
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 3
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        if R <: Complex
            side = isqrt(dim - 2) # real lower triangle and imaginary under diagonal
            @assert side^2 == dim - 2
            cone.is_complex = true
        else
            side = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
            @assert side * (side + 1) == 2 * (dim - 2)
            cone.is_complex = false
        end
        cone.side = side
        cone.sc_const = sc_const
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

reset_data(cone::HypoPerLogdetTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.inv_hess_prod_updated = cone.hess_fact_updated = false)

function setup_data(cone::HypoPerLogdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    side = cone.side
    cone.mat = zeros(R, side, side)
    cone.dual_mat = zeros(R, side, side)
    cone.mat2 = zeros(R, side, side)
    cone.mat3 = zeros(R, side, side)
    cone.mat4 = zeros(R, side, side)
    # cone.Wi = zeros(R, side, side)
    cone.Wivzi = zeros(R, side, side)
    cone.tmpw = zeros(T, dim - 2)
    return
end

get_nu(cone::HypoPerLogdetTri) = cone.side + 2

function set_initial_point(arr::AbstractVector{T}, cone::HypoPerLogdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    # NOTE if not using theta = 16, rescaling the ray yields central ray
    (arr[1], arr[2], w) = (sqrt(cone.sc_const) / T(16)) * get_central_ray_hypoperlogdettri(cone.side)
    incr = (cone.is_complex ? 2 : 1)
    k = 3
    @inbounds for i in 1:cone.side
        arr[k] = w
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoPerLogdetTri{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]

    if v > eps(T)
        svec_to_smat!(cone.mat, view(cone.point, 3:cone.dim), cone.rt2)
        cone.fact_mat = cholesky!(Hermitian(cone.mat, :U), check = false)
        if isposdef(cone.fact_mat)
            cone.ldWv = logdet(cone.fact_mat) - cone.side * log(v)
            cone.z = v * cone.ldWv - u
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

function is_dual_feas(cone::HypoPerLogdetTri{T}) where {T}
    u = cone.dual_point[1]
    v = cone.dual_point[2]
    if u < -eps(T)
        svec_to_smat!(cone.dual_mat, view(cone.dual_point, 3:cone.dim), cone.rt2)
        dual_fact = cholesky!(Hermitian(cone.dual_mat, :U), check = false)
        return isposdef(dual_fact) && (v - u * (logdet(dual_fact) - cone.side * log(-u) + cone.side) > eps(T))
    end
    return false
end

function update_grad(cone::HypoPerLogdetTri)
    @assert cone.is_feas
    z = cone.z
    u = cone.point[1]
    v = cone.point[2]

    # copyto!(cone.Wi, cone.fact_mat.factors)
    # LinearAlgebra.inv!(Cholesky(cone.Wi, 'U', 0))
    cone.Wi = inv(cone.fact_mat)
    cone.nLz = (cone.side - cone.ldWv) / z
    cone.ldWvuv = cone.ldWv - u / v
    cone.vzip1 = 1 + v / z
    cone.grad[1] = inv(z)
    cone.grad[2] = cone.nLz - inv(v)
    gend = view(cone.grad, 3:cone.dim)
    smat_to_svec!(gend, cone.Wi, cone.rt2)
    gend .*= -cone.vzip1

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLogdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    Wi = cone.Wi
    Wivzi = cone.Wivzi
    H = cone.hess.data
    side = cone.side
    @views Hww = cone.hess.data[3:cone.dim, 3:cone.dim]
    symm_kron(Hww, Wi, cone.rt2)
    Hww .*= cone.vzip1
    Wivzi_vec = smat_to_svec!(cone.tmpw, Wivzi, cone.rt2)
    mul!(Hww, Wivzi_vec, Wivzi_vec', true, true)

    cone.hess_updated = true
    return cone.hess
end

# function update_inv_hess(cone::HypoPerLogdetTri)
#     if !cone.inv_hess_prod_updated
#         update_inv_hess_prod(cone)
#     end
#     H = cone.inv_hess.data
#     side = cone.side
#     z = cone.z
#     v = cone.point[2]
#     @views w = cone.point[3:end]
#     W = Hermitian(svec_to_smat!(cone.mat2, w, cone.rt2), :U)
#     update_inv_hess_prod(cone)
#     denom = z + (side + 1) * v
#
#     @views Hww = H[3:cone.dim, 3:cone.dim]
#     symm_kron(Hww, W, cone.rt2)
#     Hww .*= z
#     mul!(Hww, w, w', abs2(v) / denom, true)
#     Hww ./= (z + v)
#
#     cone.inv_hess_updated = true
#     return cone.inv_hess
# end

# update first two rows of the Hessian
function update_hess_prod(cone::HypoPerLogdetTri)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    z = cone.z
    H = cone.hess.data

    @. cone.Wivzi = cone.Wi / cone.ldWvuv
    H[1, 1] = inv(z) / z
    H[1, 2] = cone.nLz / z
    h1end = view(H, 1, 3:cone.dim)
    smat_to_svec!(h1end, cone.Wivzi, cone.rt2)
    h1end ./= -z
    H[2, 2] = abs2(cone.nLz) + (cone.side / z + inv(v)) / v
    h2end = view(H, 2, 3:cone.dim)
    smat_to_svec!(h2end, cone.Wi, cone.rt2)
    h2end .*= ((cone.ldWv - cone.side) / cone.ldWvuv - 1) / z

    cone.hess_prod_updated = true
    return
end

# function update_inv_hess_prod(cone::HypoPerLogdetTri)
#     # TODO remove with explicit expression for (1,1) element
#     if !cone.hess_prod_updated
#         update_hess_prod(cone)
#     end
#     H = cone.inv_hess.data
#     side = cone.side
#     ldWv = cone.ldWv
#     z = cone.z
#     u = cone.point[1]
#     v = cone.point[2]
#     w = cone.point[3:end]
#     denom = z + (side + 1) * v
#
#     H[1, 2] = v * (v * abs2(ldWv) - (u + (side - 1) * v) * ldWv + side * u) * v
#     @. @views H[1, 3:end] = w * v * (v * ldWv + z) # = (2 * v * ldWv - u)
#     @views H[1, 1] = (denom - dot(H[1, 2:end], cone.hess[1, 2:end])) / cone.hess[1, 1] # TODO complicated but doable
#     H[2, 2] = v * (z + v) * v
#     @views H[2, 3:end] .= v * w * v
#     H ./= denom
#
#     cone.inv_hess_prod_updated = true
#     return
# end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLogdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end

    const_diag = cone.ldWvuv * cone.vzip1 * cone.ldWvuv
    @views mul!(prod[1:2, :], cone.hess[1:2, :], arr)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat2, view(arr, 3:cone.dim, i), cone.rt2)
        copytri!(cone.mat2, 'U', cone.is_complex)
        rdiv!(cone.mat2, cone.fact_mat)
        const_i = tr(cone.mat2) / const_diag
        for j in 1:cone.side
            @inbounds cone.mat2[j, j] += const_i
        end
        ldiv!(cone.fact_mat, cone.mat2)
        smat_to_svec!(view(prod, 3:cone.dim, i), cone.mat2, cone.rt2)
    end
    @views mul!(prod[3:cone.dim, :], cone.hess[3:cone.dim, 1:2], arr[1:2, :], true, cone.vzip1)

    return prod
end

# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLogdetTri)
#     if !cone.inv_hess_prod_updated
#         update_inv_hess_prod(cone)
#     end
#     side = cone.side
#     z = cone.z
#     v = cone.point[2]
#     @views w = cone.point[3:end]
#     W = Hermitian(svec_to_smat!(cone.mat4, w, cone.rt2), :U)
#
#     denom = z + (side + 1) * v
#     @views mul!(prod[1:2, :], cone.inv_hess[1:2, :], arr)
#     @inbounds for i in 1:size(arr, 2)
#         @views arr_w = arr[3:end, i]
#         @views prod_w = prod[3:end, i]
#         svec_to_smat!(cone.mat2, arr_w, cone.rt2)
#         copytri!(cone.mat2, 'U', cone.is_complex)
#         mul!(cone.mat3, cone.mat2, W)
#         mul!(cone.mat2, W, cone.mat3, z, false)
#         smat_to_svec!(prod_w, cone.mat2, cone.rt2)
#         prod_w .+= dot(w, arr_w) * abs2(v) .* w / denom
#         prod_w ./= (z + v)
#     end
#     @views mul!(prod[3:cone.dim, :], cone.inv_hess[3:cone.dim, 1:2], arr[1:2, :], true, true)
#
#     return prod
# end

function correction(cone::HypoPerLogdetTri{T}, primal_dir::AbstractVector{T}) where T
    @assert cone.grad_updated
    u_dir = primal_dir[1]
    v_dir = primal_dir[2]
    @views w_dir = primal_dir[3:end]
    corr = cone.correction
    @views w_corr = corr[3:end]
    v = cone.point[2]
    z = cone.z
    tmpw = cone.tmpw
    Wi = Hermitian(cone.Wi, :U)
    pi = cone.ldWv # TODO rename
    nLz = cone.nLz
    side = cone.side
    w_dim = cone.dim - 2
    vz = v / z
    udz = u_dir / z
    vdz = v_dir / z

    vec_Wi = smat_to_svec!(tmpw, Wi, cone.rt2)
    dot_Wi_S = dot(vec_Wi, w_dir)
    S = copytri!(svec_to_smat!(cone.mat2, w_dir, cone.rt2), 'U', cone.is_complex)
    ldiv!(cone.fact_mat, S)
    dot_skron = real(dot(S, S'))

    const6 = 1 + v * nLz
    uuw_scal = -2 * udz * vz / z
    vvw_scal = -vdz * (2 * const6 * nLz + side / z)
    uvw_scal =-2 * (2 * vz * nLz + inv(z)) / z
    const8 = -2 * (1 + vz)
    const10 = 2 * (vz * udz + vdz * const6)
    const9 = -2 * abs2(vz) * dot_Wi_S + const10
    const7 = uuw_scal + uvw_scal * v_dir
    const11 = -vz * (vz * dot_skron + 2 * (abs2(vz * dot_Wi_S) - dot_Wi_S * const10)) + u_dir * const7 + vvw_scal * v_dir

    @. w_corr = const11 * vec_Wi
    rdiv!(S, cone.fact_mat.U)
    mul!(cone.mat3, S, S')
    vec_S2 = smat_to_svec!(tmpw, cone.mat3, cone.rt2)
    @. w_corr += const8 * vec_S2
    skron2 = rdiv!(S, cone.fact_mat.U')
    vec_skron2 = smat_to_svec!(tmpw, skron2, cone.rt2)
    t4awd = (2 * vz * abs2(dot_Wi_S) + dot(vec_skron2, w_dir)) / z
    @. w_corr += const9 * vec_skron2

    const2 = -2 * udz * (pi - side) / z / z
    const3 = (2 * abs2(nLz) + side / z / v) * vdz
    const5 = side / v / z
    const4 = nLz * (2 * abs2(nLz) + 3 * const5) - (const5 + 2 * inv(v) / v) / v
    corr[1] = 2 * abs2(udz) / z + (2 * const2 + const3) * v_dir + (uuw_scal + const7) * dot_Wi_S + vz * t4awd
    corr[2] = const4 * abs2(v_dir) + (2 * vvw_scal + uvw_scal * u_dir) * dot_Wi_S + u_dir * (const2 + 2 * const3) + const6 * t4awd

    corr /= -2

    return corr
end

# see analysis in https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_hypoperlogdettri(Wside::Int)
    if Wside <= 10
        # lookup points where x = f'(x)
        return central_rays_hypoperlogdettri[Wside, :]
    end
    # use nonlinear fit for higher dimensions
    x = log10(Wside)
    u = -0.102485 * x ^ 4 + 0.908632 * x ^ 3 - 3.029054 * x ^ 2 + 4.528779 * x - 13.901470
    v = 0.358933 * x ^ 3 - 2.592002 * x ^ 2 + 6.354740 * x + 17.280377
    w = 0.027883 * x ^ 3 - 0.231444 * x ^ 2 + 0.652673 * x + 21.997811
    return [u, v, w]
end

const central_rays_hypoperlogdettri = [
    -14.06325335  17.86151855  22.52090275
    -13.08878205  18.91121795  22.4393585
    -12.54888342  19.60639116  22.40621157
    -12.22471372  20.09640151  22.39805249
    -12.01656536  20.45698931  22.40140061
    -11.87537532  20.73162694  22.4097267
    -11.77522327  20.9468238  22.41993593
    -11.70152722  21.11948341  22.4305642
    -11.64562635  21.26079849  22.44092946
    -11.6021318  21.37842775  22.45073131
    ]
