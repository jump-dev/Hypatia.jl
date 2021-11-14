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
    mu::Vector{T}
    zeta::Vector{T}
    scU::Matrix{R}
    scVt::Matrix{R}
    cu::T
    Zu::T
    umzdd::Matrix{T}
    simdot::Matrix{T}
    sVt::Matrix{R}
    Uz::Matrix{R}

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
    cone.mu = zeros(T, d1)
    cone.zeta = zeros(T, d1)
    cone.scU = zeros(R, d1, d1)
    cone.scVt = zeros(R, d1, d2)
    cone.umzdd = zeros(T, d1, d1)
    cone.simdot = zeros(T, d1, d1)
    cone.Uz = zeros(R, d1, d1)
    cone.sVt = zeros(R, d1, d2)
    cone.w1 = zeros(R, d1, d2)
    cone.w2 = zeros(R, d1, d2)
    cone.w3 = zeros(R, d1, d2) # TODO check if using - only alloc in hess/invhess
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
    mu = cone.mu
    zeta = cone.zeta
    s1 = cone.s1
    w1 = cone.w1
    U1 = cone.U1
    g = cone.grad

    @. mu = s / u
    @. zeta = T(0.5) * (u - mu * s)
    cone.cu = (cone.d1 - 1) / u

    g[1] = cone.cu - sum(inv, zeta)

    @. s1 = mu / zeta
    mul!(U1, U, Diagonal(s1))
    mul!(w1, U1, Vt)
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
    U = cone.U
    Vt = cone.Vt
    mu = cone.mu
    zeta = cone.zeta
    umzdd = cone.umzdd
    simdot = cone.simdot
    s1 = cone.s1
    s2 = cone.s2

    # for hess
    @. s1 = sqrt(zeta)
    @. s2 = mu / s1
    mul!(cone.scVt, Diagonal(s2), Vt)
    @. s2 = inv(s1)
    mul!(cone.scU, U, Diagonal(s2))


# TODO separate inv hess aux function


    # for inv hess
    mul!(cone.sVt, Diagonal(s), Vt)
    mul!(cone.Uz, U, Diagonal(zeta))

    cone.Zu = -cone.cu + sum(inv, u - z_i for z_i in zeta)

    # umzdd = 0.5 * (u .+ mu * s')
    # simdot = zeta ./ (u .- mu * s')
    @inbounds for j in 1:d1
        mu_j = mu[j]
        z_j = zeta[j]
        for i in 1:(j - 1)
            mus_ij = mu_j * s[i]
            umzdd[i, j] = umzdd[j, i] = T(0.5) * (u + mus_ij)
            umus_ij = u - mus_ij
            simdot[i, j] = zeta[i] / umus_ij
            simdot[j, i] = z_j / umus_ij
        end
        umzdd[j, j] = u - z_j
        simdot[j, j] = T(0.5)
    end

    cone.hess_aux_updated = true
end

function update_hess(cone::EpiNormSpectral{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    u = cone.point[1]
    zeta = cone.zeta
    scU = cone.scU
    scVt = cone.scVt

    s1 = cone.s1
    w1 = cone.w1
    w2 = cone.w2
    w3 = cone.w3 # TODO only alloc here if needed
    U1 = cone.U1
    U2 = cone.U2
    H = cone.hess.data

    # u, u
    ui = inv(u)
    @. s1 = (inv(zeta) - ui) / zeta # TODO can put in sum if not using later
    @inbounds H[1, 1] = sum(s1) - cone.cu / u

    # u, w
    @. s1 = -inv(zeta)
    mul!(U1, scU, Diagonal(s1))
    mul!(w1, U1, scVt)
    @views vec_copyto!(H[1, 2:end], w1)

    # w, w
    # TODO alloc only WtauI if used
    # TODO bring kron functions into utils
    Zi = scU * scU'
    tau = scU * scVt / sqrt(T(2)) # TODO just gradient part
    WtauI = T(0.5) * scVt' * scVt + ui * I

    idx_incr = (cone.is_complex ? 2 : 1)
    r_idx = 2
    for i in 1:d2, j in 1:d1
        c_idx = r_idx
        @inbounds for k in i:d2
            taujk = tau[j, k]
            WtauIik = WtauI[i, k]
            lstart = (i == k ? j : 1)
            @inbounds for l in lstart:d1
                term1 = Zi[l, j] * WtauIik
                term2 = tau[l, i] * taujk
                spectral_kron_element!(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
        r_idx += idx_incr
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
    zeta = cone.zeta
    scU = cone.scU
    scVt = cone.scVt
    r = w1 = cone.w1
    simU = w2 = cone.w2
    sim = cone.U1
    S1 = cone.U2
    @views S1diag = S1[diagind(S1)]

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        pui = p / u
        mul!(simU, scU', r)
        mul!(sim, simU, scVt')
        @. S1 = 0.5 * (sim + sim')
        @. S1diag -= p / zeta

        prod[1, j] = -sum((pui + real(S1[i, i])) / zeta[i] for i in 1:d1) -
            cone.cu * pui

        mul!(w2, Hermitian(S1, :U), scVt, true, inv(u))
        mul!(w1, scU, w2)
        @views vec_copyto!(prod[2:end, j], w1)
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
    u = cone.point[1]
    zeta = cone.zeta
    umzdd = cone.umzdd
    sVt = cone.sVt
    r = w1 = cone.w1
    simU = w2 = cone.w2
    sim = cone.U1
    S1 = cone.U2
    @views S1diag = S1[diagind(S1)]

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        mul!(simU, cone.U', r)
        mul!(sim, simU, sVt')

        c1 = u * (p + sum(real(sim[i, i]) / umzdd[i, i] for i in 1:d1)) / cone.Zu
        prod[1, j] = c1

        sim .*= cone.simdot
        @. S1 = sim + sim'
        @. S1diag -= c1 / zeta
        S1 ./= umzdd

        mul!(w2, Hermitian(S1, :U), sVt, -1, u)
        mul!(w1, cone.Uz, w2)
        @views vec_copyto!(prod[2:end, j], w1)
    end

    return prod
end

function dder3(cone::EpiNormSpectral{T}, dir::AbstractVector{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    u = cone.point[1]
    zeta = cone.zeta
    scU = cone.scU
    scVt = cone.scVt
    r = w1 = cone.w1
    simU = w2 = cone.w2
    sim = cone.U1
    S1 = cone.U2
    S2 = cone.U3
    @views S1diag = S1[diagind(S1)]
    @views S2diag = S2[diagind(S2)]
    dder3 = cone.dder3

    p = dir[1]
    @views vec_copyto!(r, dir[2:end])

    pui = p / u
    mul!(simU, scU', r)
    mul!(sim, simU, scVt')
    @. S1 = T(-0.5) * (sim + sim')
    @. S1diag += p / zeta

    mul!(S2, simU, simU', T(-0.5) / u, false)
    @. S2diag += T(0.5) * p / zeta * pui
    mul!(S2, Hermitian(S1, :U), S1, -1, true)

    @inbounds dder3[1] = -sum((real(S1[i, i]) * pui + real(S2[i, i])) / zeta[i]
        for i in 1:cone.d1) - cone.cu * abs2(pui)

    mul!(w1, Hermitian(S2, :U), scVt)
    mul!(w1, Hermitian(S1, :U), simU, inv(u), true)
    mul!(w2, scU, w1)
    @views vec_copyto!(dder3[2:end], w2)

    return dder3
end
