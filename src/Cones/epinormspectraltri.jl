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
    mu::Vector{T}
    zeta::Vector{T}
    Vrzi::Matrix{R}
    Vmrzi::Matrix{R}


    zh::Vector{T}
    uzi::Vector{T}
    szi::Vector{T}
    cu::T
    tdd::Matrix{T}
    uzti1::T
    usti::Vector{T}
    z2tidd::Matrix{T}

    w1::Matrix{R}
    w2::Matrix{R}
    w3::Matrix{R}
    s1::Vector{T}
    s2::Vector{T}

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

    cone.V = zeros(R, d, d)
    cone.zeta = zeros(T, d)
    cone.mu = zeros(T, d)
    cone.Vrzi = zeros(R, d, d)
    cone.Vmrzi = zeros(R, d, d)

    cone.zh = zeros(T, d)
    cone.uzi = zeros(T, d)
    cone.szi = zeros(T, d)
    cone.usti = zeros(T, d)
    cone.tdd = zeros(T, d, d)
    cone.z2tidd = zeros(T, d, d)

    cone.w1 = zeros(R, d, d)
    cone.w2 = zeros(R, d, d)
    cone.w3 = zeros(R, d, d)
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
    d = cone.d
    s1 = cone.s1
    s2 = cone.s2

    @. s1 = sqrt(cone.zeta)
    @. s2 = cone.mu / s1
    mul!(cone.Vmrzi, cone.V, Diagonal(s2))
    @. s2 = inv(s1)
    mul!(cone.Vrzi, cone.V, Diagonal(s2))

    # @. s1 = inv(sqrt(zeta))
    # mul!(cone.Vrzi, cone.V, Diagonal(s1))
    # @. s1 =
    #
    #
    # u = cone.point[1]
    # s = cone.s
    # zh = cone.zh
    # tdd = cone.tdd
    # z2tidd = cone.z2tidd
    #
    # # irtzh = inv.(sqrt.(zh))
    # # mul!(cone.Vrzi, cone.V, Diagonal(irtzh))
    #
    # zh = cone.zh
    # uzi = cone.uzi
    # szi = cone.szi
    #
    # u2 = abs2(u)
    # @. zh = T(0.5) * (u2 - abs2(s))
    # @. uzi = u / zh
    # @. szi = s / zh
    #
    #
    #
    # u2 = abs2(u)
    # th = u2 .- zh
    # @. cone.usti = u * s / th
    # cone.uzti1 = u / (1 + sum(zh[i] / th[i] for i in 1:d))
    #
    # u2h = T(0.5) * abs2(u)
    # for j in 1:d
    #     sh_j = T(0.5) * s[j]
    #     zh_j = zh[j]
    #     for i in 1:j
    #         tij = u2h + s[i] * sh_j
    #         tdd[i, j] = tij
    #         z2tidd[i, j] = zh[i] / tij * zh_j
    #     end
    # end

    cone.hess_aux_updated = true
end

function update_hess(cone::EpiNormSpectralTri{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d = cone.d
    u = cone.point[1]
    zeta = cone.zeta
    Vrzi = cone.Vrzi
    Vmrzi = cone.Vmrzi
    s1 = cone.s1
    w1 = cone.w1
    w2 = cone.w2
    w3 = cone.w3
    H = cone.hess.data

    # u, u
    ui = inv(u)
    H[1, 1] = sum((inv(z_i) - ui) / z_i for z_i in zeta) - cone.cu / u

    # u, w
    @. s1 = -inv(zeta)
    mul!(w2, Vrzi, Diagonal(s1))
    mul!(w1, w2, Vmrzi')
    @views smat_to_svec!(H[1, 2:end], w1, cone.rt2)

    # w, w
    # TODO or write faster symmetric spectral kron
    s = cone.s
    mu = cone.mu
    Dzi = Diagonal(inv.(zeta))
    hwwscal = (Dzi * (u .+ mu * s') * Dzi) ./ (2 * u)
    hwwscal = Matrix(Symmetric(hwwscal))
    @views Hww = H[2:end, 2:end]
    eig_dot_kron!(Hww, hwwscal, cone.V, w1, w2, w3, cone.rt2)

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
    simU = w2 = cone.w2
    S1 = cone.w3
    @views S1diag = S1[diagind(S1)]

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views svec_to_smat!(r, arr[2:end, j], cone.rt2)

        pui = p / u
        mul!(simU, Vrzi', Hermitian(r, :U))
        mul!(sim, simU, Vmrzi)
        @. S1 = T(0.5) * (sim + sim')
        @. S1diag -= p / zeta

        prod[1, j] = -sum((pui + real(S1[i, i])) / zeta[i] for i in 1:d) -
            cone.cu * pui

        mul!(w2, Hermitian(S1, :U), Vmrzi', true, inv(u))
        mul!(w1, Vrzi, w2)
        @views smat_to_svec!(prod[2:end, j], w1, cone.rt2)
    end

    return prod
end

# function update_inv_hess(cone::EpiNormSpectralTri)
#     cone.hess_aux_updated || update_hess_aux(cone)
#     isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
#     d = cone.d
#     V = cone.V
#     hiuui = cone.hiuui
#     usti = cone.usti
#     z2tidd = cone.z2tidd
#     w1 = cone.w1
#     Hi = cone.inv_hess.data
#
#     # u, u
#     Hi[1, 1] = inv(hiuui)
#
#     # u, w
#     HiuW = V * Diagonal(usti ./ hiuui) * V'
#     @views Hiuwvec = Hi[1, 2:end]
#     smat_to_svec!(Hiuwvec, HiuW, cone.rt2)
#
#     # w, w
#     w1 = zero(w1)
#     w2 = zero(w1)
#     w3 = zero(w1)
#     z2tiddscal = Matrix(Symmetric(z2tidd, :U)) # TODO
#     @views Hiww = Hi[2:end, 2:end]
#     eig_dot_kron!(Hiww, z2tiddscal, V, w1, w2, w3, cone.rt2)
#
#     mul!(Hiww, Hiuwvec, Hiuwvec', hiuui, true) # TODO sqrt
#
#     cone.inv_hess_updated = true
#     return cone.inv_hess
# end

# function inv_hess_prod!(
#     prod::AbstractVecOrMat,
#     arr::AbstractVecOrMat,
#     cone::EpiNormSpectralTri,
#     )
#
# # TODO
#
#
#     cone.hess_aux_updated || update_hess_aux(cone)
#     d = cone.d
#     u = cone.point[1]
#     V = cone.V
#     uzti1 = cone.uzti1
#     usti = cone.usti
#     z2tidd = cone.z2tidd
#     r = cone.w1
#
#     @inbounds for j in 1:size(prod, 2)
#         p = arr[1, j]
#         @views svec_to_smat!(r, arr[2:end, j], cone.rt2)
#
#         sim = V' * Hermitian(r, :U) * V
#
#         c1 = u * (p + sum(usti[i] * real(sim[i, i]) for i in 1:d)) * uzti1
#         prod[1, j] = c1
#
#         w3 = Diagonal(c1 * usti) + sim .* z2tidd
#         w2 = V * Hermitian(w3, :U) * V'
#         @views smat_to_svec!(prod[2:end, j], w2, cone.rt2)
#     end
#
#     return prod
# end

function dder3(cone::EpiNormSpectralTri{T}, dir::AbstractVector{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    u = cone.point[1]
    zeta = cone.zeta
    Vrzi = cone.Vrzi
    Vmrzi = cone.Vmrzi
    # TODO maybe need fewer fields
    r = w1 = cone.w1
    simU = w2 = cone.w2
    sim = zero(w1) # TODO
    S1 = cone.w3
    S2 = zero(w1) # TODO
    @views S1diag = S1[diagind(S1)]
    @views S2diag = S2[diagind(S2)]
    dder3 = cone.dder3

    p = dir[1]
    @views svec_to_smat!(r, dir[2:end], cone.rt2)

    pui = p / u
    mul!(simU, Vrzi', Hermitian(r, :U))
    mul!(sim, simU, Vmrzi)
    @. S1 = T(-0.5) * (sim + sim')
    @. S1diag += p / zeta

    mul!(S2, simU, simU', T(-0.5) / u, false)
    @. S2diag += T(0.5) * p / zeta * pui
    mul!(S2, Hermitian(S1, :U), S1, -1, true)

    @inbounds dder3[1] = -sum((real(S1[i, i]) * pui + real(S2[i, i])) / zeta[i]
        for i in 1:cone.d) - cone.cu * abs2(pui)

    mul!(w1, Hermitian(S2, :U), Vmrzi')
    mul!(w1, Hermitian(S1, :U), simU, inv(u), true)
    mul!(w2, Vrzi, w1)
    @views smat_to_svec!(dder3[2:end], w2, cone.rt2)

    return dder3
end
