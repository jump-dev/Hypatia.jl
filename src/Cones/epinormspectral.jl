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
    uzi::Vector{T}
    U2::Matrix{R}
    Vt2::Matrix{R}
    cu::T
    zti1::T
    tzi::Vector{T}
    usti::Vector{T}
    hiww::Matrix{T}
    hiww2::Matrix{T}

    w1::Matrix{R}
    sd1::Vector{T}
    sd2::Vector{T}

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
    cone.z = zeros(T, d1)
    cone.uzi = zeros(T, d1)
    cone.U2 = zeros(R, d1, d1)
    cone.Vt2 = zeros(R, d1, d2)
    cone.tzi = zeros(T, d1)
    cone.usti = zeros(T, d1)
    cone.hiww = zeros(T, d1, d1)
    cone.hiww2 = zeros(T, d1, d1)
    cone.w1 = zeros(R, d1, d2)
    cone.sd1 = zeros(T, d1)
    cone.sd2 = zeros(T, d1)
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
    z = cone.z
    uzi = cone.uzi
    sd1 = cone.sd1
    rtz = cone.sd2
    w1 = cone.w1
    g = cone.grad

    cone.cu = (cone.d1 - 1) / u
    @. z = (u - s) * (u + s)
    @. uzi = 2 * u / z
    @. rtz = sqrt(z)
    @. sd1 = inv(rtz)
    mul!(cone.U2, U, Diagonal(sd1))
    @. sd1 = s / rtz
    mul!(cone.Vt2, Diagonal(sd1), Vt)

    g[1] = cone.cu - sum(uzi)
    mul!(w1, cone.U2, cone.Vt2, 2, false)
    @views vec_copyto!(g[2:end], w1)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiNormSpectral{T}) where T
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    d1 = cone.d1
    u = cone.point[1]
    s = cone.s
    z = cone.z
    tzi = cone.tzi
    usti = cone.usti
    hiww = cone.hiww
    hiww2 = cone.hiww2

    u2 = abs2(u)
    zti1 = one(u)
    @inbounds for j in 1:d1
        s_j = s[j]
        for i in 1:d1
            z_i = z[i]
            s_ij = s[i] * s_j
            t_ij = u2 + s_ij
            hiww[i, j] = z_i / (u2 - s_ij) * s_j
            hiww2[i, j] = z_i / t_ij * s_j
        end
        z_j = z[j]
        tj = u2 + abs2(s_j)
        zt_ij = z_j / tj
        zti1 += zt_ij
        tzi[j] = tj / z_j
        usti[j] = 2 * u * s_j / tj
        hiww[j, j] = s_j
        hiww2[j, j] = zt_ij * s_j
    end
    cone.zti1 = zti1

    cone.hess_aux_updated = true
end

function update_hess(cone::EpiNormSpectral)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    u = cone.point[1]
    z = cone.z
    uzi = cone.uzi
    U2 = cone.U2
    Vt2 = cone.Vt2
    tzi = cone.tzi
    w1 = cone.w1
    H = cone.hess.data

    # u, u
    H[1, 1] = -cone.cu / u + 2 * sum(tzi ./ z) #

    # u, w
    Huw = U2 * Diagonal(-2 * uzi) * Vt2 #
    @views vec_copyto!(H[1, 2:end], Huw)

    # w, w
    U2t = Matrix(U2') # TODO
    c_idx = 2
    reim1s = (cone.is_complex ? [1, im] : [1,])
    @inbounds for j in 1:d2, i in 1:d1, reim1 in reim1s
        U2_i = reim1 * U2t[:, i]
        sims_ij = U2_i * Vt2[:, j]'
        sims_ij += sims_ij'
        w2 = Hermitian(sims_ij, :U) * Vt2
        w2[:, j] += U2_i
        w3 = 2 * U2 * w2
        @views vec_copyto!(H[2:end, c_idx], w3)
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
    d1 = cone.d1
    u = cone.point[1]
    z = cone.z
    uzi = cone.uzi
    U2 = cone.U2
    Vt2 = cone.Vt2
    cu = cone.cu
    tzi = cone.tzi
    r = cone.w1

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])
        simU = U2' * r
        sims = simU * Vt2'
        sims += sims'

        prod[1, j] = -cu * p / u + 2 * sum((p * tzi[i] -
            u * real(sims[i, i])) / z[i] for i in 1:d1)

        w3 = Diagonal(-p * uzi) + sims
        w2 = 2 * U2 * (simU + Hermitian(w3, :U) * Vt2)
        @views vec_copyto!(prod[2:end, j], w2)
    end

    return prod
end

function update_inv_hess(cone::EpiNormSpectral{T}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    u = cone.point[1]
    U = cone.U
    Vt = cone.Vt
    z = cone.z
    zti1 = cone.zti1
    usti = cone.usti
    hiww = cone.hiww
    hiww2 = cone.hiww2
    Hi = cone.inv_hess.data

    # u, u
    huu = u / zti1 * u
    Hi[1, 1] = huu

    # u, w
    HiuW = U * Diagonal(huu * usti) * Vt #
    @views vec_copyto!(Hi[1, 2:end], HiuW)

    # w, w
    Ut = Matrix(U') # TODO
    c_idx = 2
    reim1s = (cone.is_complex ? [1, im] : [1,])
    @inbounds for j in 1:d2, i in 1:d1, reim1 in reim1s
        U_i = reim1 * Ut[:, i]
        sim_ij = U_i * Vt[:, j]'
        sim_ij .*= hiww
        sim_ij += sim_ij'
        w2 = sim_ij .* hiww2
        w3 = -w2 * Vt
        w3[:, j] += z .* U_i
        w4 = T(0.5) * U * w3
        @views vec_copyto!(Hi[2:end, c_idx], w4)
        c_idx += 1
    end

    HiuW2 = U * Diagonal(sqrt(huu) * usti) * Vt #
    @views Hiuw2vec = Hi[2:end, 1]
    vec_copyto!(Hiuw2vec, HiuW2)
    @views mul!(Hi[2:end, 2:end], Hiuw2vec, Hiuw2vec', true, true)

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormSpectral{T},
    ) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    U = cone.U
    Vt = cone.Vt
    z = cone.z
    zti1 = cone.zti1
    usti = cone.usti
    hiww = cone.hiww
    hiww2 = cone.hiww2
    r = cone.w1

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])
        simU = U' * r
        sim = simU * Vt'

        c1 = u * (p + sum(usti[i] * real(sim[i, i]) for i in 1:d1)) / zti1 * u
        prod[1, j] = c1

        sim2 = sim .* hiww
        sim2 += sim2'
        w3 = Diagonal(2 * c1 * usti) - sim2 .* hiww2

        lmul!(Diagonal(z), simU)
        w2 = T(0.5) * U * (simU + w3 * Vt)
        @views vec_copyto!(prod[2:end, j], w2)
    end

    return prod
end

function dder3(cone::EpiNormSpectral{T}, dir::AbstractVector{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    z = cone.z
    uzi = cone.uzi
    U2 = cone.U2
    Vt2 = cone.Vt2
    dder3 = cone.dder3

    p = dir[1]
    @views r = vec_copyto!(cone.w1, dir[2:end])

    simU = U2' * r
    sims = simU * Vt2'
    sims += sims'

    puzi = p * uzi
    v1 = 2 * u * puzi .- p

    M4 = Diagonal(uzi) * sims
    M5 = M4 + M4'
    simUsqr = simU * simU'

    M7 = Hermitian(sims) * sims + simUsqr - p * M5 + Diagonal(p * v1 ./ z)
    M8 = sims - Diagonal(puzi)

    Wcorr = -2 * U2 * (Hermitian(M8) * simU + Hermitian(M7) * Vt2)
    @views vec_copyto!(dder3[2:end], Wcorr)

    @inbounds dder3[1] = -cone.cu * abs2(p / u) +
        T(0.5) * real(dot(Hermitian(sims), Hermitian(M5))) +
        2 * sum((puzi[i] * (T(0.5) * v1[i] - p) + u * real(simUsqr[i, i]) -
        real(sims[i, i]) * v1[i]) / z[i] for i in 1:d1)

    return dder3
end
