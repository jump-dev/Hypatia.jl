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
- try to reduce theta parameter but maintain self-concordance
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
    scal_hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    old_hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    mat::Matrix{R}
    dual_mat::Matrix{R}
    mat2::Matrix{R} # TODO named differently in some cones, fix inconsistency
    mat3::Matrix{R}
    fact_mat
    ldWv::T
    z::T
    Wi::Matrix{R}
    nLz::T
    ldWvuv::T
    vzip1::T
    Wivzi::Matrix{R}

    correction::Vector{T}

    function HypoPerLogdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        sc_const::Real = 256, # TODO reduce this
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert !use_dual # TODO delete later
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

reset_data(cone::HypoPerLogdetTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.hess_fact_updated = false)

use_scaling(::HypoPerLogdetTri) = false

use_correction(::HypoPerLogdetTri) = true

function setup_data(cone::HypoPerLogdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.old_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.mat = Matrix{R}(undef, cone.side, cone.side)
    cone.dual_mat = Matrix{R}(undef, cone.side, cone.side)
    cone.mat2 = similar(cone.mat)
    cone.mat3 = similar(cone.mat)
    cone.Wi = similar(cone.mat)
    cone.Wivzi = similar(cone.mat)
    cone.correction = zeros(T, dim)
    return
end

get_nu(cone::HypoPerLogdetTri) = 2 * cone.sc_const * (cone.side + 1)

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

function update_feas(cone::HypoPerLogdetTri)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]

    if v > 0
        svec_to_smat!(cone.mat, view(cone.point, 3:cone.dim), cone.rt2)
        cone.fact_mat = cholesky!(Hermitian(cone.mat, :U), check = false)
        if isposdef(cone.fact_mat)
            cone.ldWv = logdet(cone.fact_mat) - cone.side * log(v)
            cone.z = v * cone.ldWv - u
            cone.is_feas = (cone.z > 0)
        else
            cone.is_feas = false
        end
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_dual_feas(cone::HypoPerLogdetTri)
    @assert !cone.feas_updated
    u = cone.dual_point[1]
    v = cone.dual_point[2]
    side = cone.side

    if u < 0
        svec_to_smat!(cone.dual_mat, view(cone.point, 3:cone.dim), cone.rt2)
        dual_fact = cholesky!(Hermitian(cone.dual_mat, :U), check = false)
        if isposdef(dual_fact)
            return v > u * (logdet(Symmetric(W)) - side * log(-u) + side)
        else
            return false
        end
    else
        return false
    end
end

function update_grad(cone::HypoPerLogdetTri)
    @assert cone.is_feas
    z = cone.z
    u = cone.point[1]
    v = cone.point[2]

    copyto!(cone.Wi, cone.fact_mat.factors)
    LinearAlgebra.inv!(Cholesky(cone.Wi, 'U', 0))
    cone.nLz = (cone.side - cone.ldWv) / z
    cone.ldWvuv = cone.ldWv - u / v
    cone.vzip1 = 1 + v / z
    cone.grad[1] = cone.sc_const / z
    cone.grad[2] = cone.sc_const * (cone.nLz - (cone.side + 1) / v)
    gend = view(cone.grad, 3:cone.dim)
    smat_to_svec!(gend, cone.Wi, cone.rt2)
    gend .*= -cone.vzip1 * cone.sc_const

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLogdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    Wi = cone.Wi
    Wivzi = cone.Wivzi
    rt2 = cone.rt2
    H = cone.hess.data
    side = cone.side

    idx_incr = (cone.is_complex ? 2 : 1)
    for i in 1:side
        for j in 1:(i - 1)
            row_idx = (cone.is_complex ? 1 + (i - 1)^2 + 2j : 2 + div((i - 1) * i, 2) + j)
            col_idx = row_idx
            @inbounds for k in i:side
                @inbounds for l in (i == k ? j : 1):(k - 1)
                    terma = Wi[k, i] * Wi[j, l]
                    termb = Wi[l, i] * Wi[j, k]
                    Wiji = Wivzi[j, i]
                    Wilk = Wivzi[l, k]
                    term1 = (terma + termb) * cone.vzip1 + Wiji * 2 * real(Wilk)
                    H[row_idx, col_idx] = real(term1)
                    @inbounds if cone.is_complex
                        H[row_idx + 1, col_idx] = -imag(term1)
                        term2 = (terma - termb) * cone.vzip1 - Wiji * 2im * imag(Wilk)
                        H[row_idx, col_idx + 1] = imag(term2)
                        H[row_idx + 1, col_idx + 1] = real(term2)
                    end
                    col_idx += idx_incr
                end

                l = k
                term = cone.rt2 * (Wi[i, k] * Wi[k, j] * cone.vzip1 + Wivzi[i, j] * Wivzi[k, k])
                H[row_idx, col_idx] = real(term)
                @inbounds if cone.is_complex
                    H[row_idx + 1, col_idx] = imag(term)
                end
                col_idx += 1
            end
        end

        j = i
        row_idx = (cone.is_complex ? 1 + (i - 1)^2 + 2j : 2 + div((i - 1) * i, 2) + j)
        col_idx = row_idx
        @inbounds for k in i:side
            @inbounds for l in (i == k ? j : 1):(k - 1)
                term = cone.rt2 * (Wi[k, i] * Wi[j, l] * cone.vzip1 + Wivzi[i, j] * Wivzi[k, l])
                H[row_idx, col_idx] = real(term)
                @inbounds if cone.is_complex
                    H[row_idx, col_idx + 1] = imag(term)
                end
                col_idx += idx_incr
            end

            l = k
            H[row_idx, col_idx] = abs2(Wi[k, i]) * cone.vzip1 + real(Wivzi[i, i] * Wivzi[k, k])
            col_idx += 1
        end
    end

    @. @views cone.hess.data[3:cone.dim, :] *= cone.sc_const
    copyto!(cone.old_hess.data, H)

    cone.hess_updated = true
    return cone.hess
end

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
    H[2, 2] = abs2(cone.nLz) + (cone.side / z + inv(v) * (cone.side + 1)) / v
    h2end = view(H, 2, 3:cone.dim)
    smat_to_svec!(h2end, cone.Wi, cone.rt2)
    h2end .*= ((cone.ldWv - cone.side) / cone.ldWvuv - 1) / z
    @. @views cone.hess.data[1:2, :] *= cone.sc_const

    cone.hess_prod_updated = true
    return
end

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
    @views mul!(prod[3:cone.dim, :], cone.hess[3:cone.dim, 1:2], arr[1:2, :], true, cone.vzip1 * cone.sc_const)

    return prod
end

# TODO allocs and simplifications
# TODO try to reuse fields already calculated for g and H
function correction2(cone::HypoPerLogdetTri, primal_dir::AbstractVector)
    @assert cone.grad_updated
    v = cone.point[2]
    z = cone.z
    Wi = Symmetric(cone.Wi)
    pi = cone.ldWv # TODO rename
    vzip1 = cone.vzip1 # 1 + v / z
    nLz = cone.nLz # (side - pi) / z # TODO use where pi - d appears
    d = cone.side # TODO rename
    w_dim = cone.dim - 2

    T = typeof(v) # TODO delete

    u_dir = primal_dir[1]
    v_dir = primal_dir[2]
    @views w_dir = primal_dir[3:end]
    vec_Wi = smat_to_svec!(similar(primal_dir, w_dim), Wi, cone.rt2) # TODO allocates
    S = copytri!(svec_to_smat!(cone.mat2, w_dir, cone.rt2), 'U', cone.is_complex)

    dot_Wi_S = dot(vec_Wi, w_dir)
    ldiv!(cone.fact_mat, S)
    dot_skron = real(dot(S, S'))

    rdiv!(S, cone.fact_mat.U)
    mul!(cone.mat3, S, S') # TODO use outer prod function
    term1 = smat_to_svec!(zeros(T, w_dim), cone.mat3, cone.rt2) # TODO allocates
    term1 .*= -2 * vzip1

    skron2 = rdiv!(S, cone.fact_mat.U')
    vec_skron2 = smat_to_svec!(zeros(T, w_dim), skron2, cone.rt2) # TODO allocates
    scal1 = -abs2(v / z)
    term2 = (dot_skron * scal1) * vec_Wi + (2 * dot_Wi_S * scal1) * vec_skron2

    scal2 = 2 * scal1 * v / z * abs2(dot_Wi_S)
    term3 = scal2 * vec_Wi

    # TODO factor like rootdet
    # Tuww contribution to w part of correction
    term4a = 2 * abs2(v) / z ^ 3 * dot_Wi_S * vec_Wi + v / abs2(z) * vec_skron2
    term4 = 2 * u_dir * term4a

    # Tvww contribution to w part of correction
    term5a = (inv(z) - v * (pi - d) / abs2(z)) * (2 * v / z * dot_Wi_S * vec_Wi + vec_skron2)
    term5 = 2 * v_dir * term5a

    uuw_scal = -2 * u_dir * v / z ^ 3
    vvw_scal = v_dir / abs2(z) * (2 * (pi - d) * (1 - v * (pi - d) / z) - d)
    uvw_scal = 2 * (2 * v * (pi - d) / abs2(z) - inv(z)) / z
    corrw =
        term1 + term2 + term3 + term4 + term5 +
        uuw_scal * u_dir * vec_Wi + # uuw
        vvw_scal * v_dir * vec_Wi + # vvw
        uvw_scal * u_dir * v_dir * vec_Wi # uvw

    Tuuu = 2 / z ^ 3
    Tuuv = -2 / z ^ 3 * (pi - d)
    Tuvv = 2 * abs2(pi - d) / z ^ 3 + d / abs2(z) / v
    corru = Tuuu * u_dir ^ 2 + Tuuv * u_dir * v_dir * 2 + # uuu + uuv
        uuw_scal * dot(vec_Wi, w_dir) * 2 + # uuw
        uvw_scal * dot(vec_Wi, w_dir) * v_dir + # uvw
        dot(term4a, w_dir) + # uww
        Tuvv * abs2(v_dir) # uvv
    Tvvv = -2 * (pi - d) / abs2(z) * (abs2(pi - d) / z + d / v) - d / z / v * (1 / v + (pi - d) / z) - 2 * (d + 1) / v ^ 3
    corrv = Tvvv * abs2(v_dir) + Tuvv * u_dir * v_dir * 2 + vvw_scal * dot(vec_Wi, w_dir) * 2 + # vvv + uvv + wvv
        Tuuv * abs2(u_dir) + dot(term5a, w_dir) + uvw_scal * dot(vec_Wi, w_dir) * u_dir # vuu + vww + vwu

    corr = vcat(corru, corrv, corrw)

    corr *= cone.sc_const / -2

    return corr
end


# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
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
