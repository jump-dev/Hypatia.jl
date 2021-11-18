"""
$(TYPEDEF)

Epigraph of real symmetric or complex Hermitian matrix spectral norm (i.e.
maximum absolute value of eigenvalues) cone of dimension `dim` in svec format.

    $(FUNCTIONNAME){T, R}(dim::Int, use_dual::Bool = false)
"""
mutable struct EpiNormSpectralTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    is_complex::Bool
    rt2::T

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
    inv_hess_aux_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    s::Vector{T}
    V::Matrix{R}
    mu::Vector{T}
    zeta::Vector{T}
    Vrzi::Matrix{R}
    Vmrzi::Matrix{R}
    cu::T
    Zu::T
    sumzi::Vector{T}
    umzzzi::Matrix{T}
    zumziz::Matrix{T}

    w1::Matrix{R}
    w2::Matrix{R}
    w3::Matrix{R}
    w4::Matrix{R}
    s1::Vector{T}
    s2::Vector{T}

    function EpiNormSpectralTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        cone.is_complex = (R <: Complex)
        cone.d = svec_side(R, dim - 1)
        return cone
    end
end

reset_data(cone::EpiNormSpectralTri) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
    cone.inv_hess_aux_updated = cone.hess_fact_updated = false)

function setup_extra_data!(
    cone::EpiNormSpectralTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    d = cone.d
    cone.V = zeros(R, d, d)
    cone.zeta = zeros(T, d)
    cone.mu = zeros(T, d)
    cone.Vrzi = zeros(R, d, d)
    cone.Vmrzi = zeros(R, d, d)
    cone.sumzi = zeros(T, d)
    cone.zumziz = zeros(T, d, d)
    cone.w1 = zeros(R, d, d)
    cone.w2 = zeros(R, d, d)
    cone.w3 = zeros(R, d, d)
    cone.w4 = zeros(R, d, d)
    cone.s1 = zeros(T, d)
    cone.s2 = zeros(T, d)
    return cone
end

get_nu(cone::EpiNormSpectralTri) = 1 + cone.d

function set_initial_point!(
    arr::AbstractVector{T},
    cone::EpiNormSpectralTri{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormSpectralTri{T}) where {T <: Real}
    @assert !cone.feas_updated
    u = cone.point[1]
    cone.feas_updated = true
    cone.is_feas = false
    (u > eps(T)) || return false
    @views W = svec_to_smat!(cone.V, cone.point[2:end], cone.rt2)
    copytri!(W, 'U', true)

    # fast bounds (note opinf = op1):
    # spec <= frob <= spec * rtd
    # op1 / rtd <= spec <= op1
    frob = norm(W, 2)
    op1 = opnorm(W, 1)
    rtd = sqrt(T(cone.d))

    # lower bounds
    lb = max(frob, op1) / rtd
    (u - lb > eps(T)) || return false

    # upper bounds
    ub = min(frob, op1)
    if u - ub < eps(T)
        # use fast Cholesky-based feasibility check, rescale W*W' by inv(u)
        Z = mul!(cone.w1, W, W', -inv(u), false)
        @inbounds for i in 1:cone.d
            Z[i, i] += u
        end
        isposdef(cholesky!(Hermitian(Z, :U), check = false)) || return false
    end

    # compute eigendecomposition and final feasibility check
    cone.s = update_eigen!(cone.V)
    cone.is_feas = (u - maximum(abs, cone.s) > eps(T))

    return cone.is_feas
end

function is_dual_feas(cone::EpiNormSpectralTri{T}) where {T <: Real}
    u = cone.dual_point[1]
    (u > eps(T)) || return false
    @views W = svec_to_smat!(cone.w1, cone.dual_point[2:end], cone.rt2)
    copytri!(W, 'U', true)

    # fast bounds: frob <= nuc <= frob * rtd
    frob = norm(W, 2)
    (u - sqrt(T(cone.d)) * frob > eps(T)) && return true
    (u - frob > eps(T)) || return false

    # final feasibility check
    return (u - sum(abs, eigvals!(Hermitian(W, :U))) > eps(T))
end

function update_grad(cone::EpiNormSpectralTri{T}) where T
    @assert cone.is_feas
    u = cone.point[1]
    V = cone.V
    s = cone.s
    mu = cone.mu
    zeta = cone.zeta
    s1 = cone.s1
    w1 = cone.w1
    w2 = cone.w2
    g = cone.grad

    @. mu = s / u
    @. zeta = T(0.5) * (u - mu * s)
    cone.cu = (cone.d - 1) / u

    g[1] = cone.cu - sum(inv, zeta)

    @. s1 = mu / zeta
    mul!(w1, V, Diagonal(s1))
    mul!(w2, w1, V')
    @views smat_to_svec!(g[2:end], w2, cone.rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiNormSpectralTri{T}) where T
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    s1 = cone.s1
    s2 = cone.s2

    @. s1 = sqrt(cone.zeta)
    @. s2 = cone.mu / s1
    @. cone.Vmrzi = s2 * cone.V'
    @. s2 = inv(s1)
    @. cone.Vrzi = s2 * cone.V'

    cone.hess_aux_updated = true
end

function update_hess(cone::EpiNormSpectralTri{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d = cone.d
    isdefined(cone, :umzzzi) || (cone.umzzzi = zeros(T, d, d))
    u = cone.point[1]
    mu = cone.mu
    zeta = cone.zeta
    umzzzi = cone.umzzzi
    s1 = cone.s1
    w1 = cone.w1
    w2 = cone.w2
    H = cone.hess.data
    ui = inv(u)

    # u, w
    @. s1 = -inv(zeta)
    mul!(w2, Diagonal(s1), cone.Vmrzi)
    mul!(w1, cone.Vrzi', w2)
    @views smat_to_svec!(H[1, 2:end], w1, cone.rt2)

    # w, w
    # TODO or write faster symmetric spectral kron
    @inbounds for j in 1:cone.d
        z_j = zeta[j]
        zi_j = inv(z_j)
        mzi_j = mu[j] / z_j
        for i in 1:(j - 1)
            umzzzi[i, j] = umzzzi[j, i] = T(0.5) *
                (zi_j + mzi_j * mu[i]) / zeta[i]
        end
        umzzzi[j, j] = (zi_j - ui) / z_j
    end
    @views Hww = H[2:end, 2:end]
    eig_dot_kron!(Hww, umzzzi, cone.V, w1, w2, cone.w3, cone.rt2)

    # u, u
    H[1, 1] = tr(umzzzi) - cone.cu / u

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormSpectralTri{T},
    ) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    d = cone.d
    u = cone.point[1]
    zeta = cone.zeta
    Vrzi = cone.Vrzi
    Vmrzi = cone.Vmrzi
    sim = r = w1 = cone.w1
    w2 = cone.w2
    S1 = cone.w3
    @views S1diag = S1[diagind(S1)]

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views svec_to_smat!(r, arr[2:end, j], cone.rt2)

        pui = p / u
        mul!(w2, Vrzi, Hermitian(r, :U))
        mul!(sim, w2, Vmrzi')
        @. S1 = T(0.5) * (sim + sim')
        @. S1diag -= p / zeta

        prod[1, j] = -sum((pui + real(S1[i, i])) / zeta[i] for i in 1:d) -
            cone.cu * pui

        mul!(w2, Hermitian(S1, :U), Vmrzi, true, inv(u))
        mul!(w1, Vrzi', w2)
        @views smat_to_svec!(prod[2:end, j], w1, cone.rt2)
    end

    return prod
end

function update_inv_hess_aux(cone::EpiNormSpectralTri{T}) where T
    @assert !cone.inv_hess_aux_updated
    @assert cone.grad_updated
    u = cone.point[1]
    s = cone.s
    zeta = cone.zeta
    zumziz = cone.zumziz

    cone.Zu = -cone.cu + sum(inv, u - z_i for z_i in zeta)

    @. cone.sumzi = s / (u - zeta)

    @inbounds for j in 1:cone.d
        mu_j = cone.mu[j]
        z_j = zeta[j]
        for i in 1:(j - 1)
            zumziz[i, j] = zumziz[j, i] = 2 * zeta[i] / (u + mu_j * s[i]) * z_j
        end
        zumziz[j, j] = z_j / (u - z_j) * z_j
    end

    cone.inv_hess_aux_updated = true
end

function update_inv_hess(cone::EpiNormSpectralTri)
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    u = cone.point[1]
    V = cone.V
    s1 = cone.s1
    w1 = cone.w1
    w2 = cone.w2
    Hi = cone.inv_hess.data

    # u, u
    hiuu = Hi[1, 1] = u / cone.Zu

    # w, w
    @views Hiww = Hi[2:end, 2:end]
    eig_dot_kron!(Hiww, cone.zumziz, V, w1, w2, cone.w3, cone.rt2)

    # u, w and w, w
    rthiuu = sqrt(hiuu)
    @. s1 = rthiuu * cone.sumzi
    mul!(w2, V, Diagonal(s1))
    mul!(w1, w2, V')
    @views Hiuwvec = Hi[1, 2:end]
    smat_to_svec!(Hiuwvec, w1, cone.rt2)
    mul!(Hiww, Hiuwvec, Hiuwvec', true, u)
    Hiuwvec .*= rthiuu

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectralTri,
    )
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    d = cone.d
    u = cone.point[1]
    V = cone.V
    sumzi = cone.sumzi
    sim = r = w1 = cone.w1
    w2 = cone.w2
    @views simdiag = sim[diagind(sim)]

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views svec_to_smat!(r, arr[2:end, j], cone.rt2)

        mul!(w2, Hermitian(r, :U), V)
        mul!(sim, V', w2)

        c1 = (p + sum(sumzi[i] * real(sim[i, i]) for i in 1:d)) / cone.Zu
        prod[1, j] = u * c1

        sim .*= cone.zumziz
        @. simdiag += c1 * sumzi
        mul!(w2, V, Hermitian(sim, :U))
        mul!(w1, w2, V', u, false)
        @views smat_to_svec!(prod[2:end, j], w1, cone.rt2)
    end

    return prod
end

function dder3(cone::EpiNormSpectralTri{T}, dir::AbstractVector{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    u = cone.point[1]
    zeta = cone.zeta
    Vrzi = cone.Vrzi
    Vmrzi = cone.Vmrzi
    r = w1 = cone.w1
    simU = w2 = cone.w2
    sim = S2 = cone.w3
    S1 = cone.w4
    @views S1diag = S1[diagind(S1)]
    @views S2diag = S2[diagind(S2)]
    dder3 = cone.dder3

    p = dir[1]
    @views svec_to_smat!(r, dir[2:end], cone.rt2)

    pui = p / u
    mul!(simU, Vrzi, Hermitian(r, :U))
    mul!(sim, simU, Vmrzi')
    @. S1 = T(-0.5) * (sim + sim')
    @. S1diag += p / zeta

    mul!(S2, simU, simU', T(-0.5) / u, false)
    @. S2diag += T(0.5) * p / zeta * pui
    mul!(S2, Hermitian(S1, :U), S1, -1, true)

    @inbounds dder3[1] = -sum((real(S1[i, i]) * pui + real(S2[i, i])) / zeta[i]
        for i in 1:cone.d) - cone.cu * abs2(pui)

    mul!(w1, Hermitian(S2, :U), Vmrzi)
    mul!(w1, Hermitian(S1, :U), simU, inv(u), true)
    mul!(w2, Vrzi', w1)
    @views smat_to_svec!(dder3[2:end], w2, cone.rt2)

    return dder3
end
