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
    z::Vector{T}
    zi::Vector{T}
    szi::Vector{T}

    huu::T
    huw::Vector{T}
    hiuui::T
    hiuw::Vector{T}
    hiuw2::Vector{T}
    hiww::Matrix{T}

    temp12::Matrix{R}

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
    dim = cone.dim
    (d1, d2) = (cone.d1, cone.d2)
    cone.z = zeros(T, d1)
    cone.zi = zeros(T, d1)
    cone.szi = zeros(T, d1)
    cone.huw = zeros(T, d1)
    cone.hiuw = zeros(T, d1)
    cone.hiuw2 = zeros(T, d1)
    cone.hiww = zeros(T, d1, d1)
    cone.temp12 = zeros(R, d1, d2)
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
        @views vec_copyto!(cone.temp12, cone.point[2:end])
        cone.W_svd = svd(cone.temp12, full = false) # TODO in place
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
        W = @views vec_copyto!(cone.temp12, cone.dual_point[2:end])
        return (u - sum(svdvals!(W)) > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormSpectral)
    @assert cone.is_feas
    u = cone.point[1]
    s = cone.s = cone.W_svd.S
    U = cone.U = cone.W_svd.U
    Vt = cone.Vt = cone.W_svd.Vt
    z = cone.z
    zi = cone.zi
    szi = cone.szi
    g = cone.grad

    u2 = abs2(u)
    @. z = u2 - abs2(s)
    @. zi = inv(z)
    @. szi = 2 * s / z

    g[1] = -2 * u * sum(zi) + (cone.d1 - 1) / u
    gW = U * Diagonal(szi) * Vt
    @views vec_copyto!(g[2:end], gW)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiNormSpectral)
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    u = cone.point[1]
    s = cone.s
    U = cone.U
    Vt = cone.Vt
    z = cone.z
    zi = cone.zi
    szi = cone.szi
    d1 = cone.d1
    d2 = cone.d2
    hiww = cone.hiww

    u2 = abs2(u)
    cu = (cone.d1 - 1) / u2
    t = u2 .+ abs2.(s) # TODO
    cone.huu = 2 * sum(t[i] / z[i] * zi[i] for i in 1:d1) - cu
    @. cone.huw = -2 * u / z * szi
    cone.hiuui = 2 * sum(inv, t) - cu
    @. cone.hiuw = 2 * u / cone.hiuui * s ./ t
    @. cone.hiuw2 = 4 * u / t * zi

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
    temp12 = cone.temp12
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

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectral,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    U = cone.U
    Vt = cone.Vt
    u = cone.point[1]
    s = cone.s
    zi = cone.zi
    szi = cone.szi
    huw = cone.huw
    huu = cone.huu
    W_dir = cone.temp12

    @inbounds for j in 1:size(prod, 2)
        u_dir = arr[1, j]
        @views vec_copyto!(W_dir, arr[2:end, j])
        simUzi = Diagonal(zi) * U' * W_dir
        simszi2 = (simUzi * Vt') * Diagonal(szi)

        prod[1, j] = huu * u_dir - 2 * u * tr(Hermitian(simszi2))

        HW1 = Diagonal(huw * u_dir) + (simszi2 + simszi2') * Diagonal(s)
        HW = U * (2 * simUzi + HW1 * Vt)
        @views vec_copyto!(prod[2:end, j], HW)
    end

    return prod
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
    temp12 = cone.temp12
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
        HiW1 = hiww .* (sim + sim')
        HiW2 = -HiW1 * Vt
        HiW2[:, j] += zUt_i
        HiW = U * HiW2
        @views vec_copyto!(Hi[2:end, c_idx], HiW)
        c_idx += 1
    end
    @views mul!(Hi[2:end, 2:end], Hiuwvec, Hiuwvec', hiuui, true)

    cone.inv_hess_updated = true
    return cone.inv_hess
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
    s = cone.s
    z = cone.z
    hiuui = cone.hiuui
    hiuw2 = cone.hiuw2
    hiww = cone.hiww
    W_dir = cone.temp12

    @inbounds for j in 1:size(prod, 2)
        u_dir = arr[1, j]
        @views vec_copyto!(W_dir, arr[2:end, j])
        simUz = Diagonal(z / 2) * U' * W_dir
        simsz = (simUz * Vt') * Diagonal(s)

        c1 = (u_dir + sum(hiuw2[i] * real(simsz[i, i]) for i in 1:d1)) / hiuui
        prod[1, j] = c1

        dconst = 2 * u * c1
        HiW1 = hiww .* (dconst * I - (simsz + simsz'))
        HiW = U * (simUz + HiW1 * Vt)
        @views vec_copyto!(prod[2:end, j], HiW)
    end

    return prod
end

function dder3(cone::EpiNormSpectral, dir::AbstractVector)
    @assert cone.grad_updated
    u = cone.point[1]
    U = cone.U
    Vt = cone.Vt
    s = cone.s
    z = cone.z
    zi = cone.zi
    dder3 = cone.dder3

    u_dir = dir[1]
    @views W_dir = vec_copyto!(cone.temp12, dir[2:end])

    # TODO
    c1 = abs2.(2 * u ./ z) - zi
    udzi = u_dir ./ z
    szi = s ./ z
    Ds = Diagonal(s)
    Dzi = Diagonal(zi)

    simU = U' * W_dir
    sims = (simU * Vt') * Ds

    M3 = sims + sims'
    M2 = Hermitian(Dzi * M3 * Dzi)

    M4a = Dzi * M2
    D1 = Diagonal(udzi .* c1)
    M6 = -2 * u * Hermitian(M4a + M4a') + D1
    tr2 = real(dot(sims, M6 + 3 * D1))

    simUa = Dzi * simU
    M5 = Hermitian(simUa * simUa', :U)
    tr1 = tr(M5)

    M5 += u_dir * M6
    M7 = (M2 * M3 * Dzi + M5) * Ds

    M2 += Diagonal(-2 * u * udzi .* zi)
    M8 = M2 * simU

    Wcorr = -2 * U * (M8 + M7 * Vt)
    @views vec_copyto!(dder3[2:end], Wcorr)

    c1 .-= 2 * zi
    dder3[1] = 2 * u * (tr1 + u_dir * sum(udzi .* c1)) -
        tr2 - (cone.d1 - 1) / u * abs2(u_dir / u)

    return dder3
end
