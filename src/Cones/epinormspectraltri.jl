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
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    s::Vector{T}
    V::Matrix{R}
    z::Vector{T} #
    zh::Vector{T}
    V2::Matrix{R}
    cu::T
    tdd::Matrix{T}
    uzti1::T
    usti::Vector{T}

    uzi::Vector{T}
    szi::Vector{T}

    z2tidd::Matrix{T}

    w1::Matrix{R}

    function EpiNormSpectralTri{T, R}(
        dim::Int,
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
    cone.hess_fact_updated = false)

function setup_extra_data!(
    cone::EpiNormSpectralTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    d = cone.d
    cone.z = zeros(T, d)
    cone.zh = zeros(T, d)

    cone.uzi = zeros(T, d)
    cone.szi = zeros(T, d)
    cone.V = zeros(R, d, d)
    cone.V2 = zeros(R, d, d)
    cone.usti = zeros(T, d)
    cone.tdd = zeros(T, d, d)
    cone.z2tidd = zeros(T, d, d)

    cone.w1 = zeros(R, d, d)
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

# TODO speed up using cholesky?
# TODO also use norm bound
function update_feas(cone::EpiNormSpectralTri{T}) where {T <: Real}
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views svec_to_smat!(cone.V, cone.point[2:end], cone.rt2)
        cone.s = update_eigen!(cone.V)
        cone.is_feas = (u - maximum(abs, cone.s) > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

# TODO speed up using cholesky?
# TODO also use norm bound
function is_dual_feas(cone::EpiNormSpectralTri{T}) where {T <: Real}
    u = cone.dual_point[1]
    if u > eps(T)
        W = @views svec_to_smat!(cone.w1, cone.dual_point[2:end], cone.rt2)
        return (u - sum(abs, eigvals!(Hermitian(W, :U))) > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormSpectralTri{T}) where T
    @assert cone.is_feas
    u = cone.point[1]
    V = cone.V
    s = cone.s
    zh = cone.zh
    uzi = cone.uzi
    szi = cone.szi
    g = cone.grad

    cone.cu = (cone.d - 1) / u
    u2 = abs2(u)
    @. zh = T(0.5) * (u2 - abs2(s))
    @. uzi = u / zh
    @. szi = s / zh

    g[1] = cone.cu - sum(uzi)
    gW = V * Diagonal(szi) * V'
    @views smat_to_svec!(g[2:end], gW, cone.rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiNormSpectralTri{T}) where T
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    d = cone.d
    u = cone.point[1]
    s = cone.s
    z = cone.z
    zh = cone.zh
    uzi = cone.uzi
    szi = cone.szi
    V2 = cone.V2
    tdd = cone.tdd
    z2tidd = cone.z2tidd

    irtzh = inv.(sqrt.(zh))
    mul!(V2, cone.V, Diagonal(irtzh))

    u2 = abs2(u)
    th = u2 .- zh
    @. cone.usti = u * s / th
    cone.uzti1 = u / (1 + sum(zh[i] / th[i] for i in 1:d))

    u2h = T(0.5) * abs2(u)
    for j in 1:d
        sh_j = T(0.5) * s[j]
        zh_j = zh[j]
        for i in 1:j
            tij = u2h + s[i] * sh_j
            tdd[i, j] = tij
            z2tidd[i, j] = zh[i] / tij * zh_j
        end
    end

    cone.hess_aux_updated = true
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectralTri,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    d = cone.d
    u = cone.point[1]
    s = cone.s
    V2 = cone.V2
    zh = cone.zh
    szi = cone.szi
    cu = cone.cu
    tdd = cone.tdd
    r = cone.w1

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views svec_to_smat!(r, arr[2:end, j], cone.rt2)

        sim = V2' * Hermitian(r, :U) * V2

        c1 = sum((p * tdd[i, i] / zh[i] - u * real(sim[i, i]) * s[i])
            / zh[i] for i in 1:d)
        prod[1, j] = c1 - cu * p / u

        w3 = Diagonal(-p * u * szi) + sim .* tdd
        w2 = V2 * Hermitian(w3, :U) * V2'
        @views smat_to_svec!(prod[2:end, j], w2, cone.rt2)
    end

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectralTri,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    d = cone.d
    u = cone.point[1]
    V = cone.V
    uzti1 = cone.uzti1
    usti = cone.usti
    z2tidd = cone.z2tidd
    r = cone.w1

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views svec_to_smat!(r, arr[2:end, j], cone.rt2)

        sim = V' * Hermitian(r, :U) * V

        c1 = u * (p + sum(usti[i] * real(sim[i, i]) for i in 1:d)) * uzti1
        prod[1, j] = c1

        w3 = Diagonal(c1 * usti) + sim .* z2tidd
        w2 = V * Hermitian(w3, :U) * V'
        @views smat_to_svec!(prod[2:end, j], w2, cone.rt2)
    end

    return prod
end

function dder3(cone::EpiNormSpectralTri, dir::AbstractVector)
    cone.hess_aux_updated || update_hess_aux(cone)
    d = cone.d
    u = cone.point[1]
    V = cone.V
    V2 = cone.V2
    s = cone.s
    zh = cone.zh
    uzi = cone.uzi
    dder3 = cone.dder3

    p = dir[1]
    @views svec_to_smat!(cone.w1, dir[2:end], cone.rt2)
    r = Hermitian(cone.w1, :U)

    # TODO
    zi = inv.(2 * zh)
    c1 = (2 * u * uzi .- 1) .* zi # TODO s^2/z?
    Ds = Diagonal(s)
    Dzi = Diagonal(zi)

    # TODO change to V2, but careful of rt2 factor
    simV = V' * r
    sim = V' * r * V
    sims = sim * Ds
    M1 = Hermitian(sims + sims', :U)

    M4a = Dzi * M1
    D1 = p * Diagonal(2 * u * uzi .- 1)

    simVa = Dzi * simV

    Wcorr = -2 * V * (
        Diagonal(-p * uzi) * Dzi * simV + (
        Dzi * M1 * Dzi * Hermitian(2 * cone.tdd .* sim, :U) * Dzi +
        Dzi * (simV * simV' + p * (-2 * u * (M4a + M4a') + D1)) * Ds * Dzi
        ) * V')
    @views smat_to_svec!(dder3[2:end], Wcorr, cone.rt2)

    M9 = Dzi * Hermitian(
    2 * u * (Diagonal(p^2 * (4 * u^2 * zi .- 3)) + simV * simV')
    , :U) * Dzi

    dder3[1] = tr(M9) - dot(Dzi * M1 * Dzi, -u * (M4a + M4a') + 2 * D1) -
        cone.cu * abs2(p / u)

    return dder3
end



# function dder3(cone::EpiNormSpectralTri, dir::AbstractVector)
#     cone.hess_aux_updated || update_hess_aux(cone)
#     u = cone.point[1]
#     V = cone.V
#     V2 = cone.V2
#     s = cone.s
#     zh = cone.zh
#     uzi = cone.uzi
#     dder3 = cone.dder3
#
#     z = 2 * zh #
#     zi = inv.(z)
#
#     p = dir[1]
#     @views svec_to_smat!(cone.w1, dir[2:end], cone.rt2)
#     r = Hermitian(cone.w1, :U)
#
#     # TODO
#     c1 = abs2.(uzi) - zi # TODO s^2/z?
#     Ds = Diagonal(s)
#     Dzi = Diagonal(zi)
#
#     simV = V' * r
#     sim = simV * V
#     sims = sim * Ds
#
#     M2 = Hermitian(Dzi * (sims + sims') * Dzi)
#
#     M4a = Dzi * M2
#     D1 = Diagonal(p ./ z .* c1)
#     M6 = -2 * u * Hermitian(M4a + M4a') + D1
#     tr2 = real(dot(sims, M6 + 3 * D1))
#
#     simVa = Dzi * simV
#     M5 = Hermitian(simVa * simVa', :U)
#     tr1 = tr(M5)
#     M5 += p * M6
#
#     Wcorr = -2 * V * (
#         Diagonal(-p * uzi) * Dzi * simV + (
#         M2 * (abs2(u) * sim + Ds * sim * Ds) * Dzi +
#         M5 * Ds
#         ) * V')
#     @views smat_to_svec!(dder3[2:end], Wcorr, cone.rt2)
#
#     c1 .-= 2 * zi
#     dder3[1] = 2 * u * (tr1 + p * sum(p ./ z .* c1)) -
#         cone.cu * abs2(p / u) - tr2
#
#     return dder3
# end



function update_hess(cone::EpiNormSpectralTri)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d = cone.d
    V = cone.V
    huw = cone.huw
    z2tidd = cone.z2tidd
    w1 = cone.w1
    H = cone.hess.data

    # u, u
    H[1, 1] = cone.huu

    # u, w
    Huw = V * Diagonal(huw) * V'
    @views smat_to_svec!(H[1, 2:end], Huw, cone.rt2)

    # w, w
    w1 = zero(w1)
    w2 = zero(w1)
    w3 = zero(w1)
    w4 = zero(w1)
    hwwscal = Matrix(Symmetric(inv.(z2tidd), :U)) # TODO maybe form hww directly
    @views Hww = H[2:end, 2:end]
    eig_dot_kron!(Hww, hwwscal, V, w1, w2, w3, w4, cone.rt2)

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiNormSpectralTri)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    d = cone.d
    V = cone.V
    hiuui = cone.hiuui
    usti = cone.usti
    z2tidd = cone.z2tidd
    w1 = cone.w1
    Hi = cone.inv_hess.data

    # u, u
    Hi[1, 1] = inv(hiuui)

    # u, w
    HiuW = V * Diagonal(usti ./ hiuui) * V'
    @views Hiuwvec = Hi[1, 2:end]
    smat_to_svec!(Hiuwvec, HiuW, cone.rt2)

    # w, w
    w1 = zero(w1)
    w2 = zero(w1)
    w3 = zero(w1)
    w4 = zero(w1)
    z2tiddscal = Matrix(Symmetric(z2tidd, :U)) # TODO
    @views Hiww = Hi[2:end, 2:end]
    eig_dot_kron!(Hiww, z2tiddscal, V, w1, w2, w3, w4, cone.rt2)

    mul!(Hiww, Hiuwvec, Hiuwvec', hiuui, true)

    cone.inv_hess_updated = true
    return cone.inv_hess
end
