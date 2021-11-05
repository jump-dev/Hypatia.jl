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
    z::Vector{T}
    szi::Vector{T}
    trZi::T
    # trZi2::T
    # trZi3::T
    cu::T

    huu::T
    huw::Vector{T}
    hiuui::T
    hiuw::Vector{T}
    hiww::Matrix{T}

    tempdd::Matrix{R}

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
        @show cone.d
        return cone
    end
end

reset_data(cone::EpiNormSpectralTri) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
    cone.hess_fact_updated = false)

function setup_extra_data!(
    cone::EpiNormSpectralTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    dim = cone.dim
    d = cone.d
    cone.z = zeros(T, d)
    cone.szi = zeros(T, d)
    cone.V = zeros(R, d, d)

    cone.huw = zeros(T, d)
    cone.hiuw = zeros(T, d)
    cone.hiww = zeros(T, d, d)
    cone.tempdd = zeros(R, d, d)

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
        W = @views svec_to_smat!(cone.tempdd, cone.dual_point[2:end], cone.rt2)
        return (u - sum(abs, eigvals!(Hermitian(W, :U))) > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormSpectralTri)
    @assert cone.is_feas
    u = cone.point[1]
    s = cone.s
    V = cone.V
    z = cone.z
    szi = cone.szi
    g = cone.grad

    @. z = abs2(u) - abs2(s)
    @. szi = 2 * s / z
    cone.trZi = sum(inv, z)
    cone.cu = (cone.d - 1) / u

    g[1] = -2 * u * cone.trZi + cone.cu
    gW = V * Diagonal(szi) * V'
    @views smat_to_svec!(g[2:end], gW, cone.rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiNormSpectralTri)
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    u = cone.point[1]
    s = cone.s
    z = cone.z
    szi = cone.szi
    d = cone.d
    hiww = cone.hiww

    u2 = abs2(u)
    zi = inv.(z) # TODO
    # trZi2 = sum(abs2, zi)
    uzi = u ./ z # TODO

    cone.huu = 4 * sum(abs2, uzi) - 2 * cone.trZi - cone.cu / u
    cone.huw = -2 * uzi .* szi # TODO

    t = u2 .+ abs2.(s) # TODO
    cone.hiuui = -cone.cu / u + 2 * sum(inv, t)
    cone.hiuw = 2 * u * s ./ t # TODO

    for j in 1:d
        s_j = s[j]
        z_j = z[j]
        for i in 1:(j - 1)
            dij = u2 + s[i] * s_j
            hiww[i, j] = z[i] / (2 * dij) * z_j
        end
        hiww[j, j] = z_j / (2 * t[j]) * z_j
    end

    cone.hess_aux_updated = true
end

function update_hess(cone::EpiNormSpectralTri)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d = cone.d
    V = cone.V
    huw = cone.huw
    hiww = cone.hiww
    tempdd = cone.tempdd
    H = cone.hess.data

    # u, u
    H[1, 1] = cone.huu

    # u, w
    Huw = V * Diagonal(huw) * V'
    @views smat_to_svec!(H[1, 2:end], Huw, cone.rt2)

    # w, w
    w1 = zero(tempdd)
    w2 = zero(tempdd)
    w3 = zero(tempdd)
    w4 = zero(tempdd)
    hwwscal = Matrix(Symmetric(inv.(hiww), :U)) # TODO
    @views Hww = H[2:end, 2:end] # TODO maybe pass below a starting index 2
    eig_dot_kron!(Hww, hwwscal, V, w1, w2, w3, w4, cone.rt2)

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectralTri,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    d = cone.d
    V = cone.V
    huw = cone.huw
    hiww = cone.hiww
    huu = cone.huu
    tempdd = cone.tempdd

    @inbounds for j in 1:size(prod, 2)
        u_dir = arr[1, j]
        @views svec_to_smat!(tempdd, arr[2:end, j], cone.rt2)
        sim = V' * Hermitian(tempdd, :U) * V

        prod[1, j] = huu * u_dir + sum(huw[i] * real(sim[i, i]) for i in 1:d)

        HW = V * Hermitian(Diagonal(u_dir * huw) + sim ./ hiww, :U) * V'
        @views smat_to_svec!(prod[2:end, j], HW, cone.rt2)
    end

    return prod
end

function update_inv_hess(cone::EpiNormSpectralTri)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    d = cone.d
    V = cone.V
    hiuui = cone.hiuui
    hiuw = cone.hiuw
    hiww = cone.hiww
    tempdd = cone.tempdd
    Hi = cone.inv_hess.data

    # u, u
    Hi[1, 1] = inv(hiuui)

    # u, w
    HiuW = V * Diagonal(hiuw ./ hiuui) * V'
    @views Hiuwvec = Hi[1, 2:end]
    smat_to_svec!(Hiuwvec, HiuW, cone.rt2)

    # w, w
    w1 = zero(tempdd)
    w2 = zero(tempdd)
    w3 = zero(tempdd)
    w4 = zero(tempdd)
    hiwwscal = Matrix(Symmetric(hiww, :U)) # TODO
    @views Hiww = Hi[2:end, 2:end]
    eig_dot_kron!(Hiww, hiwwscal, V, w1, w2, w3, w4, cone.rt2)

    mul!(Hiww, Hiuwvec, Hiuwvec', hiuui, true)

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectralTri,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    d = cone.d
    V = cone.V
    hiuui = cone.hiuui
    hiuw = cone.hiuw
    hiww = cone.hiww
    tempdd = cone.tempdd

    @inbounds for j in 1:size(prod, 2)
        u_dir = arr[1, j]
        @views svec_to_smat!(tempdd, arr[2:end, j], cone.rt2)
        sim = V' * Hermitian(tempdd, :U) * V

        c1 = (u_dir + sum(hiuw[i] * real(sim[i, i]) for i in 1:d)) / hiuui
        prod[1, j] = c1

        HiW = V * Hermitian(Diagonal(c1 * hiuw) + sim .* hiww, :U) * V'
        @views smat_to_svec!(prod[2:end, j], HiW, cone.rt2)
    end

    return prod
end

function dder3(cone::EpiNormSpectralTri, dir::AbstractVector)
    @assert cone.grad_updated
    u = cone.point[1]
    s = cone.s
    V = cone.V
    z = cone.z
    dder3 = cone.dder3

    u_dir = dir[1]
    @views svec_to_smat!(cone.tempdd, dir[2:end], cone.rt2)
    W_dir = Hermitian(cone.tempdd, :U)

    u2 = abs2(u)
    zi = inv.(z)
    c1 = abs2.(2 * u ./ z) - zi
    udzi = u_dir ./ z
    Ds = Diagonal(s)
    Dzi = Diagonal(zi)

    simV = V' * W_dir
    sim = simV * V
    sims = sim * Ds

    M2 = Hermitian(Dzi * (sims + sims') * Dzi)

    M4a = Dzi * M2
    D1 = Diagonal(udzi .* c1)
    M6 = -2 * u * Hermitian(M4a + M4a') + D1
    tr2 = real(dot(sims, M6 + 3 * D1))

    simVa = Dzi * simV
    M5 = Hermitian(simVa * simVa', :U)
    tr1 = tr(M5)
    M5 += u_dir * M6

    Wcorr = -2 * V * (
        Diagonal(-2 * u * udzi ./ z) * simV + (
        M2 * (u2 * sim + Ds * sim * Ds) * Dzi +
        M5 * Ds
        ) * V')
    @views smat_to_svec!(dder3[2:end], Wcorr, cone.rt2)

    c1 .-= 2 * zi
    dder3[1] = 2 * u * (tr1 + u_dir * sum(udzi .* c1)) -
        cone.cu * abs2(u_dir / u) - tr2

    return dder3
end
