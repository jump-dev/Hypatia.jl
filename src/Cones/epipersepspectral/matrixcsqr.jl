#=
matrix cone of squares, i.e. 𝕊₊ᵈ for d ≥ 1, with rank d
=#

struct MatrixCSqr{T <: Real, R <: RealOrComplex{T}} <: ConeOfSquares{T} end

vector_dim(::Type{<:MatrixCSqr{T, T} where {T <: Real}}, d::Int) = svec_length(d)
vector_dim(::Type{<:MatrixCSqr{T, Complex{T}} where {T <: Real}}, d::Int) = d^2

mutable struct MatrixCSqrCache{T <: Real, R <: RealOrComplex{T}} <: CSqrCache{T}
    is_complex::Bool
    rt2::T
    # TODO check if we need both w and viw
    w::Matrix{R}
    viw::Matrix{R}
    viw_eigen
    w_λ::Vector{T}
    w_λi::Vector{T}
    ϕ::T
    ζ::T
    ζi::T
    σ::T
    ∇h::Vector{T}
    ∇2h::Vector{T}
    ∇3h::Vector{T}
    Δh::Matrix{T}
    Δ2h::Matrix{T}
    θ::Matrix{T}

    # TODO reduce number of aux fields
    wd::Vector{T}
    wT::Matrix{T}
    w1::Matrix{R}
    w2::Matrix{R}
    w3::Matrix{R}
    w4::Matrix{R}
    α::Vector{T}
    γ::Vector{T}
    c0::T
    c4::T
    c5::T

    MatrixCSqrCache{T, R}() where {T <: Real, R <: RealOrComplex{T}} = new{T, R}()
end

function setup_csqr_cache(cone::EpiPerSepSpectral{MatrixCSqr{T, R}}) where {T, R}
    cone.cache = cache = MatrixCSqrCache{T, R}()
    cache.is_complex = (R <: Complex{T})
    cache.rt2 = sqrt(T(2))
    d = cone.d
    cache.w = zeros(R, d, d)
    cache.viw = zeros(R, d, d)
    cache.w_λ = zeros(T, d)
    cache.w_λi = zeros(T, d)
    cache.∇h = zeros(T, d)
    cache.∇2h = zeros(T, d)
    cache.∇3h = zeros(T, d)
    cache.Δh = zeros(T, d, d)
    cache.Δ2h = zeros(T, d, svec_length(d))
    cache.θ = zeros(T, d, d)
    cache.wd = zeros(T, d)
    cache.wT = zeros(T, d, d)
    cache.w1 = zeros(R, d, d)
    cache.w2 = zeros(R, d, d)
    cache.w3 = zeros(R, d, d)
    cache.w4 = zeros(R, d, d)
    cache.α = zeros(T, d)
    cache.γ = zeros(T, d)
    return
end

function set_initial_point(arr::AbstractVector, cone::EpiPerSepSpectral{<:MatrixCSqr})
    (arr[1], arr[2], w0) = get_initial_point(cone.d, cone.h)
    @views fill!(arr[3:end], 0)
    incr = (cone.cache.is_complex ? 2 : 1)
    idx = 3
    @inbounds for i in 1:cone.d
        arr[idx] = 1
        idx += incr * i + 1
    end
    return arr
end

# TODO check whether it is faster to do chol before eigdecomp
function update_feas(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.feas_updated
    cache = cone.cache
    v = cone.point[2]

    cone.is_feas = false
    if v > eps(T)
        w = cache.w
        svec_to_smat!(w, cone.w_view, cache.rt2)
        w_chol = cholesky!(Hermitian(w, :U), check = false)
        if isposdef(w_chol)
            svec_to_smat!(w, cone.w_view, cache.rt2)
            viw = cache.viw
            @. viw = w / v
            # TODO other options? eigen(A; permute::Bool=true, scale::Bool=true, sortby)
            # TODO in-place and dispatch to GLA or LAPACK.geevx! directly for efficiency
            viw_eigen = cache.viw_eigen = eigen(Hermitian(viw, :U))
            viw_λ = viw_eigen.values
            if all(>(eps(T)), viw_λ)
                cache.ϕ = h_val(viw_λ, cone.h)
                cache.ζ = cone.point[1] - v * cache.ϕ
                cone.is_feas = (cache.ζ > eps(T))
            end
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

# TODO check whether it is faster to do chol before eigdecomp
# TODO check if this is faster or slower than only using nbhd check
function is_dual_feas(cone::EpiPerSepSpectral{MatrixCSqr{T, R}}) where {T, R}
    u = cone.dual_point[1]
    (u < eps(T)) && return false
    @views w = cone.dual_point[3:end]

    uiw = cone.cache.w1
    if h_conj_dom_pos(cone.h)
        # use cholesky to check conjugate domain feasibility
        svec_to_smat!(uiw, w, cone.cache.rt2)
        w_chol = cholesky!(Hermitian(uiw, :U), check = false)
        isposdef(w_chol) || return false
    end

    svec_to_smat!(uiw, w, cone.cache.rt2)
    uiw ./= u
    # TODO in-place and dispatch to GLA or LAPACK.geevx! directly for efficiency
    uiw_eigen = eigen(Hermitian(uiw, :U))
    return (cone.dual_point[2] - u * h_conj(uiw_eigen.values, cone.h) > eps(T))
end

function update_grad(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.grad_updated && cone.is_feas
    v = cone.point[2]
    grad = cone.grad
    cache = cone.cache
    ζi = cache.ζi = inv(cache.ζ)
    viw_λ = cache.viw_eigen.values
    viw_X = cache.viw_eigen.vectors
    ∇h = cache.∇h
    h_der1(∇h, viw_λ, cone.h)
    cache.σ = cache.ϕ - dot(viw_λ, ∇h)
    @. cache.w_λ = v * viw_λ
    @. cache.w_λi = inv(cache.w_λ)

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @. cache.wd = ζi * ∇h - cache.w_λi
    mul!(cache.w1, Diagonal(cache.wd), viw_X') # TODO check efficient
    gw = mul!(cache.w2, viw_X, cache.w1)
    @views smat_to_svec!(cone.grad[3:end], gw, cache.rt2)

    cone.grad_updated = true
    return grad
end

function update_hess_aux(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    cache = cone.cache
    viw_λ = cache.viw_eigen.values
    w_λi = cache.w_λi
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    Δh = cache.Δh

    h_der2(∇2h, viw_λ, cone.h)

    rteps = sqrt(eps(T))
    @inbounds for j in 1:cone.d
        viw_λ_j = viw_λ[j]
        ∇h_j = ∇h[j]
        ∇2h_j = ∇2h[j]
        for i in 1:(j - 1)
            denom = viw_λ[i] - viw_λ_j
            if abs(denom) < rteps
                Δh[i, j] = (∇2h[i] + ∇2h_j) / 2
            else
                Δh[i, j] = (∇h[i] - ∇h_j) / denom
            end
        end
        Δh[j, j] = ∇2h_j
    end

    ζivi = cache.ζi / cone.point[2]
    @. cache.θ = ζivi * Δh + w_λi * w_λi'

    cone.hess_aux_updated = true
end

function update_hess(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    d = cone.d
    v = cone.point[2]
    H = cone.hess.data
    cache = cone.cache
    rt2 = cache.rt2
    ζi = cache.ζi
    σ = cache.σ
    viw_X = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    wd = cache.wd
    w1 = cache.w1
    w2 = cache.w2
    ζi2 = abs2(ζi)
    ζivi = ζi / v

    # Huu
    H[1, 1] = ζi2

    # Huv
    H[1, 2] = -ζi2 * σ

    # Hvv
    @inbounds sum1 = sum(abs2(viw_λ[j]) * ∇2h[j] for j in 1:d)
    H[2, 2] = v^-2 + abs2(ζi * σ) + ζivi * sum1

    # Huw
    @. wd = -ζi * ∇h
    mul!(w1, Diagonal(wd), viw_X')
    mul!(w2, viw_X, w1)
    @views Hwu = H[3:end, 1] # use later for Hww
    @views smat_to_svec!(Hwu, w2, rt2)
    @. H[1, 3:end] = ζi * Hwu

    # Hvw
    wd .*= -ζi * σ
    @. wd -= ζivi * ∇2h * viw_λ
    mul!(w1, Diagonal(wd), viw_X')
    mul!(w2, viw_X, w1)
    @views smat_to_svec!(H[2, 3:end], w2, rt2)

    # Hww
    @views Hww = H[3:end, 3:end]
    eig_kron!(Hww, cache.θ, cone)
    mul!(Hww, Hwu, Hwu', true, true)

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSepSpectral{<:MatrixCSqr{T}},
    ) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    v = cone.point[2]
    cache = cone.cache
    ζi = cache.ζi
    σ = cache.σ
    viw_X = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    w_λi = cache.w_λi
    ∇h = cache.∇h
    Δh = cache.Δh
    r_X = cache.w1
    w_aux = cache.w2
    w3 = cache.w3
    D_λi = Diagonal(w_λi)
    D_viw_λ = Diagonal(viw_λ)
    D_∇h = Diagonal(∇h)
    ζivi = ζi / v

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views svec_to_smat!(r_X, arr[3:end, j], cache.rt2)
        mul!(w_aux, Hermitian(r_X, :U), viw_X)
        mul!(r_X, viw_X', w_aux)

        @inbounds sum1 = sum(∇h[i] * real(r_X[i, i]) for i in 1:cone.d)
        c1 = -ζi * (p - σ * q - sum1) * ζi
        @. w_aux = ζivi * Δh * (r_X - q * D_viw_λ)
        @inbounds c2 = sum(viw_λ[i] * real(w_aux[i, i]) for i in 1:cone.d)

        rmul!(r_X, D_λi)
        @. w_aux += w_λi * r_X + c1 * D_∇h
        mul!(w3, Hermitian(w_aux, :U), viw_X')
        mul!(w_aux, viw_X, w3)

        prod[1, j] = -c1
        prod[2, j] = c1 * σ - c2 + q / v / v
        @views smat_to_svec!(prod[3:end, j], w_aux, cache.rt2)
    end

    return prod
end

function update_inv_hess_aux(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.inv_hess_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)
    v = cone.point[2]
    cache = cone.cache
    σ = cache.σ
    viw_X = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    ∇h = cache.∇h
    wd = cache.wd
    α = cache.α
    γ = cache.γ
    ζivi = cache.ζi / v

    @views diag_θ = cache.θ[1:(1 + cone.d):end]
    @. wd = ζivi * cache.∇2h
    @. α = ∇h / diag_θ
    wd .*= viw_λ
    @. γ = wd / diag_θ

    ζ2β = abs2(cache.ζ) + dot(∇h, α)
    c0 = σ + dot(∇h, γ)
    c1 = c0 / ζ2β
    @inbounds sum1 = sum((viw_λ[i] + c1 * α[i] - γ[i]) * wd[i] for i in 1:cone.d)
    c3 = v^-2 + σ * c1 + sum1
    c4 = inv(c3 - c0 * c1)
    c5 = ζ2β * c3
    cache.c0 = c0
    cache.c4 = c4
    cache.c5 = c5

    cone.inv_hess_aux_updated = true
end

function update_inv_hess(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    Hi = cone.inv_hess.data
    cache = cone.cache
    viw_X = cache.viw_eigen.vectors
    c4 = cache.c4
    wT = cache.wT
    w1 = cache.w1
    w2 = cache.w2

    # Hiuu, Hiuv, Hivv
    Hi[1, 1] = c4 * cache.c5
    Hiuv = Hi[1, 2] = c4 * cache.c0
    Hi[2, 2] = c4

    # Hiuw, Hivw
    @views HiuW = Hi[1, 3:end]
    @views γ_vec = Hi[3:end, 2]
    mul!(w2, Diagonal(cache.γ), viw_X')
    mul!(w1, viw_X, w2)
    smat_to_svec!(γ_vec, w1, cache.rt2)
    @. Hi[2, 3:end] = c4 * γ_vec
    mul!(w2, Diagonal(cache.α), viw_X')
    mul!(w1, viw_X, w2)
    smat_to_svec!(HiuW, w1, cache.rt2)
    @. HiuW += Hiuv * γ_vec

    # Hiww
    @views Hiww = Hi[3:end, 3:end]
    @. wT = inv(cache.θ)
    eig_kron!(Hiww, wT, cone)
    mul!(Hiww, γ_vec, γ_vec', c4, true)

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSepSpectral{<:MatrixCSqr{T}},
    ) where T
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    d = cone.d
    cache = cone.cache
    viw_X = cache.viw_eigen.vectors
    α = cache.α
    γ = cache.γ
    c0 = cache.c0
    c4 = cache.c4
    c5 = cache.c5
    r_X = Hermitian(cache.w1, :U)
    w2 = cache.w2

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views svec_to_smat!(r_X.data, arr[3:end, j], cache.rt2)
        mul!(w2, r_X, viw_X)
        mul!(r_X.data, viw_X', w2)

        qγr = q + sum(γ[i] * r_X[i, i] for i in 1:d)
        cu = c4 * (c5 * p + c0 * qγr)
        cv = c4 * (c0 * p + qγr)

        prod[1, j] = cu + sum(α[i] * r_X[i, i] for i in 1:d)
        prod[2, j] = cv

        w_prod = r_X
        w_prod.data ./= cache.θ
        @inbounds for i in 1:d
            w_prod.data[i, i] += p * α[i] + cv * γ[i]
        end
        mul!(w2, w_prod, viw_X')
        mul!(w_prod.data, viw_X, w2)
        @views smat_to_svec!(prod[3:end, j], w_prod.data, cache.rt2)
    end

    return prod
end

function update_correction_aux(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.correction_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)
    d = cone.d
    cache = cone.cache
    viw_λ = cache.viw_eigen.values
    ∇3h = cache.∇3h
    Δh = cache.Δh
    Δ2h = cache.Δ2h

    h_der3(∇3h, viw_λ, cone.h)

    rteps = sqrt(eps(T))
    @inbounds for k in 1:d, j in 1:k, i in 1:j
        (viw_λ_i, viw_λ_j, viw_λ_k) = (viw_λ[i], viw_λ[j], viw_λ[k])
        (∇3h_i, ∇3h_j, ∇3h_k) = (∇3h[i], ∇3h[j], ∇3h[k])
        denom_ij = viw_λ_i - viw_λ_j
        denom_ik = viw_λ_i - viw_λ_k

        if abs(denom_ij) < rteps
            if abs(denom_ik) < rteps
                t = (∇3h_i + ∇3h_j + ∇3h_k) / 6
            else
                t = (Δh[i, j] - Δh[j, k]) / denom_ik
            end
        else
            t = (Δh[i, k] - Δh[j, k]) / denom_ij
        end

        Δ2h[i, svec_idx(k, j)] = Δ2h[j, svec_idx(k, i)] = Δ2h[k, svec_idx(j, i)] = t
    end

    cone.correction_aux_updated = true
end

function correction(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}, dir::AbstractVector{T}) where T
    cone.correction_aux_updated || update_correction_aux(cone)
    d = cone.d
    v = cone.point[2]
    corr = cone.correction
    cache = cone.cache
    ζi = cache.ζi
    viw_X = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    w_λi = cache.w_λi
    σ = cache.σ
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    ∇3h = cache.∇3h
    Δh = cache.Δh
    Δ2h = cache.Δ2h
    vi = inv(v)

    r_X = cache.w1
    ξ_X = cache.w2
    ξb = cache.w3
    wd = cache.wd

    p = dir[1]
    q = dir[2]
    @views svec_to_smat!(r_X, dir[3:end], cache.rt2)
    mul!(ξ_X, Hermitian(r_X, :U), viw_X)
    mul!(r_X, viw_X', ξ_X)
    LinearAlgebra.copytri!(r_X, 'U', cache.is_complex)

    viq = vi * q
    D = Diagonal(viw_λ)
    @. ξ_X = vi * r_X - viq * D
    @. ξb = ζi * Δh * ξ_X
    @inbounds sum1 = sum(∇h[i] * real(r_X[i, i]) for i in 1:d)
    ζiχ = ζi * (p - σ * q - sum1)
    ξbξ = dot(Hermitian(ξb, :U), Hermitian(ξ_X, :U)) / 2
    c1 = -ζi * (ζiχ^2 + v * ξbξ)

    w_aux = ξb
    w_aux .*= ζiχ + viq
    col = 1
    @inbounds for j in 1:d, i in 1:j
        w_aux[i, j] -= ζi * sum(ξ_X[k, i]' * ξ_X[k, j] * Δ2h[k, col] for k in 1:d)
        col += 1
    end
    @inbounds c2 = sum(viw_λ[i] * real(w_aux[i, i]) for i in 1:d)

    @. wd = sqrt(w_λi)
    lmul!(Diagonal(w_λi), r_X)
    rmul!(r_X, Diagonal(wd))
    mul!(w_aux, r_X, r_X', true, true)
    D_∇h = Diagonal(∇h)
    @. w_aux += c1 * D_∇h
    mul!(ξ_X, Hermitian(w_aux, :U), viw_X')
    mul!(w_aux, viw_X, ξ_X)

    corr[1] = -c1
    @inbounds corr[2] = c1 * σ - c2 + ξbξ + viq^2 / v
    @views smat_to_svec!(corr[3:end], w_aux, cache.rt2)

    return corr
end

function eig_kron!(
    Hww::AbstractMatrix{T},
    dot_mat::Matrix{T},
    cone::EpiPerSepSpectral{<:MatrixCSqr{T}},
    ) where T
    rt2 = sqrt(T(2))
    rt2i = inv(rt2)
    d = cone.d
    cache = cone.cache
    w1 = cache.w1
    w2 = cache.w2
    w3 = cache.w3
    V = cache.w4
    copyto!(V, cache.viw_eigen.vectors') # allows column slices

    col_idx = 1
    @inbounds for j in 1:d
        @views V_j = V[:, j]
        for i in 1:(j - 1)
            @views V_i = V[:, i]
            mul!(w2, V_j, V_i', rt2i, zero(T))

            @. w3 = w2 + w2'
            w3 .*= dot_mat
            mul!(w1, Hermitian(w3, :U), V)
            mul!(w3, V', w1)
            @views smat_to_svec!(Hww[:, col_idx], w3, rt2)
            col_idx += 1

            if cache.is_complex
                w2 *= im
                @. w3 = w2 + w2'
                w3 .*= dot_mat
                mul!(w1, Hermitian(w3, :U), V)
                mul!(w3, V', w1)
                @views smat_to_svec!(Hww[:, col_idx], w3, rt2)
                col_idx += 1
            end
        end

        mul!(w3, V_j, V_j')
        w3 .*= dot_mat
        mul!(w1, Hermitian(w3, :U), V)
        mul!(w3, V', w1)
        @views smat_to_svec!(Hww[:, col_idx], w3, rt2)
        col_idx += 1
    end

    return Hww
end
