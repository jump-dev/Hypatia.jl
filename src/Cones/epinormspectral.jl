"""
$(TYPEDEF)

Epigraph of real or complex matrix spectral norm (i.e. maximum singular value)
for a matrix (stacked column-wise) of `nrows` rows and `ncols` columns with
`nrows ≤ ncols`.

    $(FUNCTIONNAME){T, R}(nrows::Int, ncols::Int, use_dual::Bool = false)
"""
mutable struct EpiNormSpectral{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d1::Int
    d2::Int
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
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    hess_aux_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    W_svd
    s::Vector{T}
    U::Matrix{R}
    Vt::Matrix{R}
    zh::Vector{T}
    uzi::Vector{T}
    szi::Vector{T}
    cu::T
    # tdd::Matrix{T}
    # uzti1::T
    # usti::Vector{T}
    # z2tidd::Matrix{T}

    z::Vector{T}
    zi::Vector{T}

    huu::T
    huw::Vector{T}
    hiuui::T
    hiuw::Vector{T}
    hiuw2::Vector{T}
    hiww::Matrix{T}

    w1::Matrix{R}

    function EpiNormSpectral{T, R}(
        d1::Int,
        d2::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert 1 <= d1 <= d2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.is_complex = (R <: Complex)
        cone.dim = 1 + vec_length(R, d1 * d2)
        cone.d1 = d1
        cone.d2 = d2
        return cone
    end
end

reset_data(cone::EpiNormSpectral) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
    cone.hess_fact_updated = false)

function setup_extra_data!(
    cone::EpiNormSpectral{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    (d1, d2) = (cone.d1, cone.d2)
    # cone.V = zeros(R, d, d)
    # cone.V2 = zeros(R, d, d)
    cone.zh = zeros(T, d1)
    cone.uzi = zeros(T, d1)
    cone.szi = zeros(T, d1)
    # cone.usti = zeros(T, d)
    # cone.tdd = zeros(T, d, d)
    # cone.z2tidd = zeros(T, d, d)

    cone.z = zeros(T, d1)
    cone.zi = zeros(T, d1)
    cone.huw = zeros(T, d1)
    cone.hiuw = zeros(T, d1)
    cone.hiuw2 = zeros(T, d1)
    cone.hiww = zeros(T, d1, d1)

    cone.w1 = zeros(R, d1, d2)
    return cone
end

get_nu(cone::EpiNormSpectral) = 1 + cone.d1

function set_initial_point!(
    arr::AbstractVector{T},
    cone::EpiNormSpectral{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

# TODO speed up using cholesky?
# TODO also use norm bound
function update_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views vec_copyto!(cone.w1, cone.point[2:end])
        cone.W_svd = svd(cone.w1, full = false) # TODO in place
        cone.is_feas = (u - maximum(cone.W_svd.S) > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

# TODO speed up using cholesky?
# TODO also use norm bound
function is_dual_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    u = cone.dual_point[1]
    if u > eps(T)
        W = @views vec_copyto!(cone.w1, cone.dual_point[2:end])
        return (u - sum(svdvals!(W)) > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormSpectral{T}) where T
    @assert cone.is_feas
    u = cone.point[1]
    U = cone.U = cone.W_svd.U
    Vt = cone.Vt = cone.W_svd.Vt
    s = cone.s = cone.W_svd.S
    zh = cone.zh
    uzi = cone.uzi
    szi = cone.szi
    g = cone.grad

    cone.cu = (cone.d1 - 1) / u
    u2 = abs2(u)
    @. zh = T(0.5) * (u2 - abs2(s))
    @. uzi = u / zh
    @. szi = s / zh

    g[1] = cone.cu - sum(uzi)
    gW = U * Diagonal(szi) * Vt
    @views vec_copyto!(g[2:end], gW)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiNormSpectral{T}) where T
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    d1 = cone.d1
    d2 = cone.d2
    u = cone.point[1]
    s = cone.s
    U = cone.U
    Vt = cone.Vt
    zh = cone.zh
    # TODO delete some
    z = cone.z
    zi = cone.zi
    uzi = cone.uzi
    szi = cone.szi
    hiww = cone.hiww # TODO rename

    u2 = abs2(u)
    @. z = u2 - abs2(s)
    @. zi = inv(z)
    cu = (cone.d1 - 1) / u2
    t = u2 .+ abs2.(s) # TODO
    cone.huu = 2 * sum(t[i] / z[i] * zi[i] for i in 1:d1) - cu
    @. cone.huw = -uzi * szi
    cone.hiuui = 2 * sum(inv, t) - cu
    @. cone.hiuw = 2 * u / cone.hiuui * s ./ t
    @. cone.hiuw2 = 2 * uzi / t

    for j in 1:d1
        s_j = s[j]
        for i in 1:d1
            sij = s[i] * s_j
            hiww[i, j] = z[i] * s_j / ((u2 - sij) * (u2 + sij))
        end
        hiww[j, j] = s_j / t[j]
    end

    cone.hess_aux_updated = true
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectral,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    zh = cone.zh
    cu = cone.cu
    r = cone.w1
    uzi = cone.uzi
# TODO
    s = cone.s
    U = cone.U
    Vt = cone.Vt
    u2 = abs2(u)
    tdd = 0.5 * (u2 .+ s * s')

    rtzh = sqrt.(zh)
    U2 = U * Diagonal(inv.(rtzh))
    Vt2 = Diagonal(s ./ rtzh) * Vt

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        simU = U2' * r
        sims = simU * Vt2'

        prod[1, j] = sum((p * tdd[i, i] / zh[i] - u * real(sims[i, i])) /
            zh[i] for i in 1:d1) - cu * p / u

        w3 = Diagonal(-p * uzi) + 0.5 * (sims + sims')
        w2 = U2 * (simU + w3 * Vt2)
        @views vec_copyto!(prod[2:end, j], w2)
    end

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectral,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    U = cone.U
    Vt = cone.Vt
    u = cone.point[1]
    zh = cone.zh
    r = cone.w1

# TODO
    s = cone.s
    u2 = abs2(u)
    th = u2 .- zh
    uzti1 = u / (1 + sum(zh[i] / th[i] for i in 1:d1))
    tdd = 0.5 * (u2 .+ s * s')
    zdd = u2 .- s * s'

    U3 = U * Diagonal(zh)
    Vt3 = Diagonal(s) * Vt

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        simU = U' * r
        sims = (simU * Vt3') ./ tdd

        c1 = u * (p + sum(u * real(sims[i, i]) for i in 1:d1)) * uzti1
        prod[1, j] = c1

        lmul!(Diagonal(zh), sims)
        w3 = (Diagonal(2 * u * c1 ./ diag(tdd)) - (sims + sims')) ./ zdd
        w2 = U3 * (simU + w3 * Vt3)

        @views vec_copyto!(prod[2:end, j], w2)
    end

    return prod
end

function dder3(cone::EpiNormSpectral, dir::AbstractVector)
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    U = cone.U
    Vt = cone.Vt
    s = cone.s
    z = cone.z
    zi = cone.zi
    uzi = cone.uzi
    dder3 = cone.dder3

    p = dir[1]
    @views r = vec_copyto!(cone.w1, dir[2:end])

    Ds = Diagonal(s)
    Dzi = Diagonal(zi)

    simU = U' * r
    sims = simU * Vt' * Ds #
    M3 = sims + sims'
    M4 = Dzi * M3
    M5 = M4 + M4'
    simUsqr = simU * simU'

    M7 = M3 * Dzi * M3 + simUsqr +
        -2 * p * u * M5 +
        p * Diagonal(p * (2 * u * uzi .- 1))

    M8 = M3 - 2 * u * p * I

    Wcorr = -2 * U * Dzi * (M8 * Dzi * simU + M7 * Dzi * Ds * Vt)

    @views vec_copyto!(dder3[2:end], Wcorr)

    dder3[1] = u * real(dot(M3, Dzi * M5 * Dzi)) + 2 * sum(
        (-p * real(M3[i, i]) / z[i] * (2 * u * uzi[i] - 1) +
        u * (real(simUsqr[i, i]) / z[i] + p * (2 * u * uzi[i] - 3) * p / z[i])
        ) / z[i] for i in 1:d1) -
        cone.cu * abs2(p / u)

    return dder3
end



function update_hess(cone::EpiNormSpectral)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    U = cone.U
    Vt = cone.Vt
    s = cone.s
    zi = cone.zi
    szi = cone.szi
    huw = cone.huw
    w1 = cone.w1
    H = cone.hess.data

    # u, u
    H[1, 1] = cone.huu

    # u, w
    Huw = U * Diagonal(huw) * Vt
    @views vec_copyto!(H[1, 2:end], Huw)

    # w, w
    ziUt = Diagonal(zi) * U'
    sziVt = Diagonal(szi) * Vt
    c_idx = 2
    reim1s = (cone.is_complex ? [1, im] : [1,])
    for j in 1:d2, i in 1:d1, reim1 in reim1s
        ziUt_i = reim1 * ziUt[:, i]
        simszi2 = ziUt_i * sziVt[:, j]'
        HW1 = (simszi2 + simszi2') * Diagonal(s)
        HW2 = HW1 * Vt
        HW2[:, j] += 2 * ziUt_i
        HW = U * HW2
        @views vec_copyto!(H[2:end, c_idx], HW)
        c_idx += 1
    end

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiNormSpectral{T}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    U = cone.U
    Vt = cone.Vt
    u = cone.point[1]
    s = cone.s
    z = cone.z
    hiuui = cone.hiuui
    hiuw = cone.hiuw
    hiww = cone.hiww
    w1 = cone.w1
    Hi = cone.inv_hess.data

    # u, u
    Hi[1, 1] = inv(hiuui)

    # u, w
    HiuW = U * Diagonal(hiuw) * Vt
    @views Hiuwvec = Hi[1, 2:end]
    @views vec_copyto!(Hiuwvec, HiuW)

    # w, w
    zUt = (T(0.5) * Diagonal(z)) * U'
    sVt = Diagonal(s) * Vt
    c_idx = 2
    reim1s = (cone.is_complex ? [1, im] : [1,])
    for j in 1:d2, i in 1:d1, reim1 in reim1s
        zUt_i = reim1 * zUt[:, i]
        sim = zUt_i * sVt[:, j]'
        w3 = hiww .* (sim + sim')
        w4 = -w3 * Vt
        w4[:, j] += zUt_i
        w2 = U * w4
        @views vec_copyto!(Hi[2:end, c_idx], w2)
        c_idx += 1
    end
    @views mul!(Hi[2:end, 2:end], Hiuwvec, Hiuwvec', hiuui, true)

    cone.inv_hess_updated = true
    return cone.inv_hess
end
