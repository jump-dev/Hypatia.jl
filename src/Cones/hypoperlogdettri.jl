#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle) symmetric positive define matrix
(u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeConeTriangle definition)

barrier (self-concordance follows from theorem 5.1.4, "Interior-Point Polynomial Algorithms in Convex Programming" by Y. Nesterov and A. Nemirovski):
theta^2 * (-log(v*logdet(W/v) - u) - logdet(W) - (n + 1) log(v))
we use theta = 16

TODO
- describe complex case
- try to reduce theta parameter but maintain self-concordance
=#

mutable struct HypoPerLogdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    max_neighborhood::T
    dim::Int
    side::Int
    is_complex::Bool
    point::Vector{T}
    rt2::T
    sc_const::T
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    mat::Matrix{R}
    mat2::Matrix{R}
    fact_mat
    ldWv::T
    z::T
    Wi::Matrix{R}
    nLz::T
    ldWvuv::T
    vzip1::T
    Wivzi::Matrix{R}

    function HypoPerLogdetTri{T, R}(
        dim::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 3
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.max_neighborhood = default_max_neighborhood() / 2 # TODO depends on selected sc_const
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
        cone.sc_const = T(1) #T(256)
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoPerLogdetTri{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = HypoPerLogdetTri{T, R}(dim, false)

reset_data(cone::HypoPerLogdetTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.hess_fact_updated = false)

function setup_data(cone::HypoPerLogdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.mat = Matrix{R}(undef, cone.side, cone.side)
    cone.mat2 = similar(cone.mat)
    cone.Wivzi = similar(cone.mat)
    return
end

get_nu(cone::HypoPerLogdetTri) = 2 * cone.sc_const * (cone.side + 1)

function set_initial_point(arr::AbstractVector{T}, cone::HypoPerLogdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    (arr[1], arr[2], w) = get_central_ray_hypoperlogdettri(cone.side)
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

function update_grad(cone::HypoPerLogdetTri)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]

    cone.Wi = inv(cone.fact_mat)
    cone.nLz = (cone.side - cone.ldWv) / cone.z
    cone.ldWvuv = cone.ldWv - u / v
    cone.vzip1 = 1 + v / (cone.z)
    cone.grad[1] = inv(cone.z)
    cone.grad[2] = cone.nLz - inv(v) * (cone.side + 1)
    gend = view(cone.grad, 3:cone.dim)
    smat_to_svec!(gend, cone.Wi, cone.rt2)
    gend .*= -cone.vzip1
    @. cone.grad *= cone.sc_const

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

    @views mul!(prod[1:2, :], cone.hess[1:2, :], arr)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat2, view(arr, 3:cone.dim, i), cone.rt2)
        dot_prod = dot(Hermitian(cone.mat2, :U), Hermitian(cone.Wivzi, :U))
        copytri!(cone.mat2, 'U', cone.is_complex)
        rdiv!(cone.mat2, cone.fact_mat)
        ldiv!(cone.fact_mat, cone.mat2)
        axpby!(dot_prod, cone.Wivzi, cone.vzip1, cone.mat2)
        smat_to_svec!(view(prod, 3:cone.dim, i), cone.mat2, cone.rt2)
    end
    @. @views prod[3:cone.dim, :] *= cone.sc_const
    @views mul!(prod[3:cone.dim, :], cone.hess[3:cone.dim, 1:2], arr[1:2, :], true, true)

    return prod
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
