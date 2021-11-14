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
    hess_aux_updated::Bool
    hess_fact_updated::Bool
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
    Urzi::Matrix{R}
    Vtsrzi::Matrix{R}
    cu::T
    zti1::T
    tzi::Vector{T}
    usti::Vector{T}
    zszidd::Matrix{T}
    zstidd::Matrix{T}

    w1::Matrix{R}
    w2::Matrix{R}
    w3::Matrix{R}
    s1::Vector{T}
    s2::Vector{T}
    U1::Matrix{R}
    U2::Matrix{R}
    U3::Matrix{R}

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


# TODO may fix some numerical failures that could be caused by ugly hessian
use_sqrt_hess_oracles(::Int, cone::EpiNormSpectral) = false



reset_data(cone::EpiNormSpectral) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
    cone.hess_fact_updated = false)

function setup_extra_data!(
    cone::EpiNormSpectral{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    (d1, d2) = (cone.d1, cone.d2)
    cone.z = zeros(T, d1)
    cone.uzi = zeros(T, d1)
    cone.Urzi = zeros(R, d1, d1)
    cone.Vtsrzi = zeros(R, d1, d2)
    cone.tzi = zeros(T, d1)
    cone.usti = zeros(T, d1)
    cone.zszidd = zeros(T, d1, d1)
    cone.zstidd = zeros(T, d1, d1)
    cone.w1 = zeros(R, d1, d2)
    cone.w2 = zeros(R, d1, d2)
    cone.w3 = zeros(R, d1, d2)
    cone.s1 = zeros(T, d1)
    cone.s2 = zeros(T, d1)
    cone.U1 = zeros(R, d1, d1)
    cone.U2 = zeros(R, d1, d1)
    cone.U3 = zeros(R, d1, d1)
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

function update_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    # TODO speed up using norm bounds, cholesky of Z?
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

function is_dual_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    # TODO speed up using norm bound, sum sqrt eigvals of WW'?
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
    s1 = cone.s1
    w1 = cone.w1
    g = cone.grad

mu = zero(s1)
zeta = zero(s1)
    @. mu = s / u
    @. zeta = T(0.5) * (u - mu * s)
    cone.cu = (cone.d1 - 1) / u

    g[1] = cone.cu - sum(inv, zeta)

    w1 .= (U * Diagonal(mu ./ zeta)) * Vt

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
    zszidd = cone.zszidd
    zstidd = cone.zstidd
    U = cone.U
    Vt = cone.Vt
    s = cone.s
    z = cone.z
    uzi = cone.uzi
    s1 = cone.s1
    rz = cone.s2
    w1 = cone.w1

    @. z = (u - s) * (u + s)
    @. uzi = 2 * u / z
    @. rz = sqrt(z)
    @. s1 = inv(rz)
    mul!(cone.Urzi, U, Diagonal(s1))
    @. s1 = s / rz
    mul!(cone.Vtsrzi, Diagonal(s1), Vt)


    u2 = abs2(u)
    zti1 = one(u)
    @inbounds for j in 1:d1
        s_j = s[j]
        z_j = z[j]
        for i in 1:(j - 1)
            s_i = s[i]
            z_i = z[i]
            s_ij = s_i * s_j
            z_ij = u2 - s_ij
            t_ij = u2 + s_ij
            # zszidd and zstidd are nonsymmetric
            zszidd[i, j] = z_i / z_ij * s_j
            zszidd[j, i] = z_j / z_ij * s_i
            zstidd[i, j] = z_i / t_ij * s_j
            zstidd[j, i] = z_j / t_ij * s_i
        end
        t_j = u2 + abs2(s_j)
        zt_ij = z_j / t_j
        zti1 += zt_ij
        tzi[j] = t_j / z_j
        usti[j] = 2 * u * s_j / t_j
        zszidd[j, j] = s_j
        zstidd[j, j] = zt_ij * s_j
    end
    cone.zti1 = zti1

    cone.hess_aux_updated = true
end

# function update_hess(cone::EpiNormSpectral)
#     cone.hess_aux_updated || update_hess_aux(cone)
#     isdefined(cone, :hess) || alloc_hess!(cone)
#     d1 = cone.d1
#     d2 = cone.d2
#     u = cone.point[1]
#     z = cone.z
#     uzi = cone.uzi
#     Urzi = cone.Urzi
#     Vtsrzi = cone.Vtsrzi
#     tzi = cone.tzi
#     w1 = cone.w1
#     w2 = cone.w2
#     w3 = cone.w3
#     U1 = cone.U1
#     U2 = cone.U2
#     H = cone.hess.data
#
#     # u, u
#     @inbounds H[1, 1] = -cone.cu / u + 2 * sum(tzi[i] / z[i] for i in 1:d1)
#
#     # u, w
#     mul!(U1, Urzi, Diagonal(uzi), -2, false)
#     mul!(w1, U1, Vtsrzi)
#     @views vec_copyto!(H[1, 2:end], w1)
#
#     # w, w
#     Urzit = copyto!(w1, Urzi') # accessing columns below
#     c_idx = 2
#     reim1s = (cone.is_complex ? [1, im] : [1,])
#     @inbounds for j in 1:d2, i in 1:d1, reim1 in reim1s
#         @views Urzi_i = Urzit[:, i]
#         @views mul!(U1, Urzi_i, Vtsrzi[:, j]', reim1, false)
#         @. U2 = U1 + U1'
#         mul!(w2, Hermitian(U2, :U), Vtsrzi)
#         @. @views w2[:, j] += reim1 * Urzi_i
#         mul!(w3, Urzi, w2, 2, false)
#         @views vec_copyto!(H[2:end, c_idx], w3)
#         c_idx += 1
#     end
#
#     cone.hess_updated = true
#     return cone.hess
# end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectral,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    U = cone.U
    Vt = cone.Vt
    s = cone.s
    s1 = cone.s1

mu = zero(s1)
zeta = zero(s1)
    @. mu = s / u
    @. zeta = 0.5 * (u - mu * s)

rtzeta = sqrt.(zeta)
scVt = Diagonal(mu ./ rtzeta) * Vt
scU = U / Diagonal(rtzeta)

    cu = cone.cu
    r = cone.w1
    w2 = cone.w2
    U1 = cone.U1
    U2 = cone.U2

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        pui = p / u
        simU = scU' * r
        sim = simU * scVt'
        s1 = Hermitian(Diagonal(p ./ zeta) - 0.5 * (sim + sim'), :U)

        prod[1, j] = -sum((pui - real(s1[i, i])) / zeta[i] for i in 1:d1) -
            cone.cu * pui

        w2 = scU * (simU / u - s1 * scVt)
        @views vec_copyto!(prod[2:end, j], w2)
    end

    # z = cone.z
    # Urzi = cone.Urzi
    # Vtsrzi = cone.Vtsrzi
    # cu = cone.cu
    # tzi = cone.tzi
    # w1 = cone.w1
    # w2 = cone.w2
    # U1 = cone.U1
    # U2 = cone.U2
    # Duzi = Diagonal(cone.uzi)
    #
    # @inbounds for j in 1:size(prod, 2)
    #     p = arr[1, j]
    #     @views vec_copyto!(w1, arr[2:end, j])
    #     mul!(w2, Urzi', w1)
    #     mul!(U1, w2, Vtsrzi')
    #     @. U2 = U1 + U1'
    #
    #     prod[1, j] = -cu * p / u + 2 * sum((p * tzi[i] -
    #         u * real(U2[i, i])) / z[i] for i in 1:d1)
    #
    #     @. U2 -= p * Duzi
    #     mul!(w2, Hermitian(U2, :U), Vtsrzi, true, true)
    #     mul!(w1, Urzi, w2, 2, false)
    #     @views vec_copyto!(prod[2:end, j], w1)
    # end

    return prod
end

# function update_inv_hess(cone::EpiNormSpectral{T}) where {T <: Real}
#     cone.hess_aux_updated || update_hess_aux(cone)
#     isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
#     d1 = cone.d1
#     d2 = cone.d2
#     u = cone.point[1]
#     U = cone.U
#     Vt = cone.Vt
#     z = cone.z
#     zti1 = cone.zti1
#     usti = cone.usti
#     zszidd = cone.zszidd
#     zstidd = cone.zstidd
#     s1 = cone.s1
#     w1 = cone.w1
#     w2 = cone.w2
#     w3 = cone.w3
#     U1 = cone.U1
#     U2 = cone.U2
#     Hi = cone.inv_hess.data
#
#     # u, u
#     huu = u / zti1 * u
#     Hi[1, 1] = huu
#
#     # u, w
#     @. s1 = huu * usti
#     mul!(U1, U, Diagonal(s1))
#     mul!(w1, U1, Vt)
#     @views vec_copyto!(Hi[1, 2:end], w1)
#
#     # w, w
#     Ut = copyto!(w1, U') # accessing columns below
#     c_idx = 2
#     reim1s = (cone.is_complex ? [1, im] : [1,])
#     @inbounds for j in 1:d2, i in 1:d1, reim1 in reim1s
#         @views U_i = Ut[:, i]
#         @views mul!(U1, U_i, Vt[:, j]', reim1, false)
#         U1 .*= zszidd
#         @. U2 = (U1 + U1') * zstidd
#         mul!(w2, U2, Vt, -1, false)
#         @. @views w2[:, j] += reim1 * z * U_i
#         mul!(w3, U, w2, T(0.5), false)
#         @views vec_copyto!(Hi[2:end, c_idx], w3)
#         c_idx += 1
#     end
#
#     rthuu = sqrt(huu)
#     @. s1 = rthuu * usti
#     mul!(U1, U, Diagonal(s1))
#     mul!(w1, U1, Vt)
#     @views Hiuw2vec = Hi[2:end, 1]
#     vec_copyto!(Hiuw2vec, w1)
#     @views mul!(Hi[2:end, 2:end], Hiuw2vec, Hiuw2vec', true, true)
#
#     cone.inv_hess_updated = true
#     return cone.inv_hess
# end

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
    s = cone.s
    s1 = cone.s1

mu = zero(s1)
zeta = zero(s1)
    @. mu = s / u
    @. zeta = 0.5 * (u - mu * s)
umz = zero(s1)
    @. umz = u - zeta
    @inbounds Zu = -cone.cu + sum(inv, umz)

r = cone.w1

w1 = cone.w1
w2 = cone.w2
U1 = cone.U1
U2 = cone.U2

sVt = Diagonal(s) * Vt


    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        simU = U' * r
        sim = simU * sVt'

        c1 = u * (p + sum(real(sim[i, i]) / umz[i] for i in 1:d1)) / Zu
        prod[1, j] = c1

        sim2 = Diagonal(zeta) * (sim ./ (u .- mu * s'))
        U2 = (Diagonal(c1 ./ zeta) - sim2 - sim2') ./ (0.5 * (u .+ mu * s'))
        w1 = U * Diagonal(zeta) * (simU * u + Hermitian(U2, :U) * sVt)

        @views vec_copyto!(prod[2:end, j], w1)
    end


    return prod
end

function dder3(cone::EpiNormSpectral{T}, dir::AbstractVector{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    s = cone.s
    U = cone.U
    Vt = cone.Vt
    r = w1 = cone.w1
    w2 = cone.w2
    s1 = cone.s1
    s2 = cone.s2
    U1 = cone.U1
    U2 = cone.U2
    U3 = cone.U3
    dder3 = cone.dder3

mu = zero(s1)
zeta = zero(s1)
    @. mu = s / u
    @. zeta = 0.5 * (u - mu * s)

rtzeta = sqrt.(zeta)
scVt = Diagonal(mu ./ rtzeta) * Vt
scU = U / Diagonal(rtzeta)

    p = dir[1]
    @views vec_copyto!(r, dir[2:end])

pui = p / u
simU = scU' * r
sim = simU * scVt'

    s1 = Hermitian(Diagonal(p ./ zeta) - 0.5 * (sim + sim'), :U)
    s2 = Hermitian(0.5 * (Diagonal(p * pui ./ zeta) - simU * simU' / u) - s1 * s1')

    @inbounds c1 = sum((real(s1[i, i]) * pui + real(s2[i, i])) / zeta[i] for i in 1:d1)
    dder3[1] = -c1 - cone.cu * abs2(pui)

    w2 = scU * (s1 * simU / u + s2 * scVt)
    @views vec_copyto!(dder3[2:end], w2)

    return dder3
end
