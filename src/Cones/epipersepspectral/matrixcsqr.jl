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
    diff_mat::Matrix{T} # first difference matrix # TODO maybe rename to Δh
    diff_ten::Matrix{T} # some elements of second difference tensor # TODO maybe rename to Δ2h

    w1::Matrix{R}
    w2::Matrix{R}
    w3::Matrix{R}
    vec_d::Vector{T}
    # inv hess aux
    # TODO check T or R below
    m::Matrix{R}
    α::Matrix{R}
    γ::Matrix{R}
    # TODO or move to cone if common
    # TODO rename constants?
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
    cache.diff_mat = zeros(T, d, d)
    cache.diff_ten = zeros(T, d, svec_length(d))
    cache.w1 = zeros(R, d, d)
    cache.w2 = zeros(R, d, d)
    cache.w3 = zeros(R, d, d)
    cache.vec_d = zeros(T, d)
    cache.m = zeros(R, d, d) # TODO check T or R
    cache.α = zeros(R, d, d) # TODO check T or R
    cache.γ = zeros(R, d, d) # TODO check T or R
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
    viw_vecs = cache.viw_eigen.vectors
    ∇h = cache.∇h
    h_der1(∇h, viw_λ, cone.h)
    cache.σ = cache.ϕ - dot(viw_λ, ∇h)
    @. cache.w_λ = v * viw_λ
    @. cache.w_λi = inv(cache.w_λ)

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @. cache.vec_d = ζi * ∇h - cache.w_λi
    mul!(cache.w1, Diagonal(cache.vec_d), viw_vecs') # TODO check efficient
    gw = mul!(cache.w2, viw_vecs, cache.w1)
    @views smat_to_svec!(cone.grad[3:end], gw, cache.rt2)

    cone.grad_updated = true
    return grad
end

function update_hess_aux(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    cache = cone.cache
    viw_λ = cache.viw_eigen.values
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    diff_mat = cache.diff_mat

    h_der2(∇2h, viw_λ, cone.h)

    rteps = sqrt(eps(T))
    for j in 1:cone.d
        viw_λ_j = viw_λ[j]
        ∇h_j = ∇h[j]
        ∇2h_j = ∇2h[j]
        for i in 1:(j - 1)
            denom = viw_λ[i] - viw_λ_j
            if abs(denom) < rteps
                # NOTE or take ∇2h at the average (viw[i] + viw[j]) / 2
                diff_mat[i, j] = (∇2h[i] + ∇2h_j) / 2
            else
                diff_mat[i, j] = (∇h[i] - ∇h_j) / denom
            end
        end
        diff_mat[j, j] = ∇2h_j
    end

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
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    w_λi = cache.w_λi
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    diff_mat = Symmetric(cache.diff_mat, :U)
    vec_d = cache.vec_d
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
    @. vec_d = -ζi * ∇h
    mul!(w1, Diagonal(vec_d), viw_vecs')
    mul!(w2, viw_vecs, w1)
    @views Hwu = H[3:end, 1] # use later for Hww
    @views smat_to_svec!(Hwu, w2, rt2)
    @. H[1, 3:end] = ζi * Hwu

    # Hvw
    vec_d .*= -ζi * σ
    @. vec_d -= ζivi * ∇2h * viw_λ
    mul!(w1, Diagonal(vec_d), viw_vecs')
    mul!(w2, viw_vecs, w1)
    @views smat_to_svec!(H[2, 3:end], w2, rt2)

    # Hww
    @views Hww = H[3:end, 3:end]
    copyto!(w1, diff_mat)
    mul!(w1, w_λi, w_λi', true, ζivi)
    eig_kron!(Hww, w1, cone)
    mul!(Hww, Hwu, Hwu', true, true)

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    v = cone.point[2]
    cache = cone.cache
    w = Hermitian(cache.w)
    ζi = cache.ζi
    σ = cache.σ
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    w_λi = cache.w_λi
    ∇h = cache.∇h
    diff_mat = Symmetric(cache.diff_mat, :U)
    r = Hermitian(cache.w1, :U)
    ζivi = ζi / v

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views svec_to_smat!(r.data, arr[3:end, j], cache.rt2)

        # TODO in-place
        r_vecs = Hermitian(viw_vecs' * r * viw_vecs)
        c1 = -ζi * (p - σ * q - dot(∇h, diag(r_vecs))) * ζi
        diag_λi = Diagonal(w_λi)
        w_aux = ζivi * Hermitian(diff_mat .* (r_vecs - Diagonal(q * viw_λ)))
        w_aux2 = c1 * Diagonal(∇h) + w_aux + diag_λi * r_vecs * diag_λi
        prod_w = viw_vecs * w_aux2 * viw_vecs'

        prod[1, j] = -c1
        prod[2, j] = c1 * σ - dot(viw_λ, diag(w_aux)) + q / v / v
        @views smat_to_svec!(prod[3:end, j], prod_w, cache.rt2)
    end

    return prod
end

function update_inv_hess_aux(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.inv_hess_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)
    v = cone.point[2]
    cache = cone.cache
    σ = cache.σ
    viw = cache.viw
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    wi = cache.wi
    ζivi = cache.ζi / v
    w1 = cache.w1
    m = cache.m
    α = cache.α
    γ = cache.γ

    # TODO prealloc
    @. w1 = ζivi * ∇2h
    @. m = inv(w1 + abs2(wi))
    @. α = m * ∇h
    w1 .*= viw
    @. γ = m * w1

    ζ2β = abs2(cache.ζ) + dot(∇h, α)
    c0 = σ + dot(∇h, γ)
    c1 = c0 / ζ2β
    @inbounds c3 = v^-2 + σ * c1 + sum((viw[i] + c1 * α[i] - γ[i]) * w1[i] for i in 1:cone.d)
    c4 = inv(c3 - c0 * c1)
    c5 = ζ2β * c3
    cache.c0 = c0
    cache.c4 = c4
    cache.c5 = c5

    cone.inv_hess_aux_updated = true
end

function update_inv_hess(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    hess(cone) # TODO
    @assert cone.hess_updated # TODO
    d = cone.d
    v = cone.point[2]
    cache = cone.cache
    Hi = cone.inv_hess.data
    ζi = cache.ζi
    ζ = cache.ζ
    ζi2 = abs2(ζi)
    σ = cache.σ
    # viw = cache.viw
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    ζivi = ζi / v
    rt2 = cache.rt2
    diff_mat = Symmetric(cache.diff_mat, :U)

    ζivi = ζi / v
    ζivi2 = ζivi / v
    w_λ = cache.w_λ
    w_λi = cache.w_λi

    m = inv.(ζivi * ∇2h + abs2.(w_λi))
    α1 = m .* ∇h
    α = viw_vecs * Diagonal(α1) * viw_vecs' # TODO can use sqrt
    β = dot(∇h, α1)
    ζ2β = ζ^2 + β

    w∇2h = ζivi2 * w_λ .* ∇2h
    γ1 = m .* w∇2h
    γ = viw_vecs * Diagonal(γ1) * viw_vecs' # TODO maybe can use sqrt
    c1 = (σ + dot(∇h, γ1)) / ζ2β

    c3 = ζi2 * σ
    c4 = ζi2 * β
    Zuu = ζi2 - c4 / ζ2β
    Zvu = -c3 + c1 * c4 - ζi2 * dot(γ1, ∇h)
    Zvv = (inv(v) + dot(w_λ, w∇2h)) / v + abs2(ζi * σ) + dot(w∇2h - c3 * ∇h, c1 * α1 - γ1)

    # Hiuu, Hiuv, Hivv
    DZi = inv(Zuu * Zvv - Zvu^2)
    Hiuu = Hi[1, 1] = DZi * Zvv
    Hiuv = Hi[1, 2] = -DZi * Zvu
    Hivv = Hi[2, 2] = DZi * Zuu

    # Hiuw, Hivw
    @views HiuW = Hi[1, 3:end]
    @views HivW = Hi[2, 3:end]
    αvec = similar(HiuW)
    γvec = similar(HiuW)
    smat_to_svec!(αvec, α, rt2)
    smat_to_svec!(γvec, γ, rt2)
    c5 = -inv(ζ2β)
    yu = c5 * αvec
    yv = c1 * αvec - γvec
    @. HiuW = -Hiuu * yu - Hiuv * yv
    @. HivW = -Hiuv * yu - Hivv * yv

    # Hiww
    @views Hiww = Hi[3:end, 3:end]

    # Hiww kron parts
    tempa = inv.(ζivi * diff_mat + w_λi * w_λi')
    # copyto!(w1, diff_mat)
    # mul!(w1, w_λi, w_λi', true, ζivi)
    # @. w2 = inv(w1) # TODO or map!
    eig_kron!(Hiww, tempa, cone)
    # mul!(Hww, Hwu, Hwu', true, true)
    Hiww .+= yu * (αvec - HiuW)' - yv * HivW'

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    # cone.hess_aux_updated || update_hess_aux(cone) # TODO
    hess(cone) # TODO
    @assert cone.hess_updated
    v = cone.point[2]
    cache = cone.cache



    Hi = inv_hess(cone)
    mul!(prod, Hi, arr)

    # # TODO @inbounds
    # for j in 1:size(arr, 2)
    #
    #
    # end

    return prod
end

function update_correction_aux(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.correction_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)
    d = cone.d
    cache = cone.cache
    viw_λ = cache.viw_eigen.values
    ∇3h = cache.∇3h
    diff_mat = Symmetric(cache.diff_mat, :U)
    diff_ten = cache.diff_ten

    h_der3(∇3h, viw_λ, cone.h)

    rteps = sqrt(eps(T))
    for k in 1:d, j in 1:k, i in 1:j
        (viw_λ_i, viw_λ_j, viw_λ_k) = (viw_λ[i], viw_λ[j], viw_λ[k])
        (∇3h_i, ∇3h_j, ∇3h_k) = (∇3h[i], ∇3h[j], ∇3h[k])
        denom_ij = viw_λ_i - viw_λ_j
        denom_ik = viw_λ_i - viw_λ_k

        if abs(denom_ij) < rteps
            if abs(denom_ik) < rteps
                t = (∇3h_i + ∇3h_j + ∇3h_k) / 6
            else
                t = (diff_mat[i, j] - diff_mat[j, k]) / denom_ik
            end
        else
            t = (diff_mat[i, k] - diff_mat[j, k]) / denom_ij
        end

        diff_ten[i, svec_idx(k, j)] = diff_ten[j, svec_idx(k, i)] = diff_ten[k, svec_idx(j, i)] = t
    end

    cone.correction_aux_updated = true
end

# TODO check all is efficient
function correction(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}, dir::AbstractVector{T}) where T
    cone.correction_aux_updated || update_correction_aux(cone)
    d = cone.d
    v = cone.point[2]
    corr = cone.correction
    cache = cone.cache
    ζi = cache.ζi
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    w_λi = cache.w_λi
    σ = cache.σ
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    ∇3h = cache.∇3h
    diff_mat = Symmetric(cache.diff_mat, :U)
    diff_ten = cache.diff_ten
    vi = inv(v)

    r_vecs = cache.w1
    ξ_vecs = cache.w2
    ξb = cache.w3
    vec_d = cache.vec_d

    p = dir[1]
    q = dir[2]
    @views svec_to_smat!(r_vecs, dir[3:end], cache.rt2)
    mul!(ξ_vecs, Hermitian(r_vecs, :U), viw_vecs)
    mul!(r_vecs, viw_vecs', ξ_vecs)
    LinearAlgebra.copytri!(r_vecs, 'U', cache.is_complex)

    viq = vi * q
    D = Diagonal(viw_λ)
    @. ξ_vecs = vi * r_vecs - viq * D
    @. ξb = ζi * diff_mat * ξ_vecs
    @inbounds ζiχ = ζi * (p - σ * q - sum(∇h[i] * real(r_vecs[i, i]) for i in 1:d))
    ξbξ = dot(Hermitian(ξb, :U), Hermitian(ξ_vecs, :U)) / 2
    c1 = -ζi * (ζiχ^2 + v * ξbξ)

    w_aux = ξb
    w_aux .*= ζiχ + viq
    col = 1
    @inbounds for j in 1:d, i in 1:j
        w_aux[i, j] -= ζi * sum(ξ_vecs[k, i]' * ξ_vecs[k, j] * diff_ten[k, col] for k in 1:d)
        col += 1
    end
    c2 = sum(viw_λ[i] * real(w_aux[i, i]) for i in 1:d)

    @. vec_d = sqrt(w_λi)
    lmul!(Diagonal(w_λi), r_vecs)
    rmul!(r_vecs, Diagonal(vec_d))
    mul!(w_aux, r_vecs, r_vecs', true, true)
    D_∇h = Diagonal(∇h)
    @. w_aux += c1 * D_∇h
    mul!(ξ_vecs, Hermitian(w_aux, :U), viw_vecs')
    mul!(w_aux, viw_vecs, ξ_vecs)

    corr[1] = -c1
    @inbounds corr[2] = c1 * σ - c2 + ξbξ + viq^2 / v
    @views smat_to_svec!(corr[3:end], w_aux, cache.rt2)

    return corr
end




# TODO refac, in-place, simplify, precompute parts
function eig_kron!(
    Hww::AbstractMatrix{T},
    dot_mat::AbstractMatrix,
    cone::EpiPerSepSpectral{<:MatrixCSqr{T}},
    ) where T
    rt2 = sqrt(T(2))
    rt2i = inv(rt2)
    d = cone.d
    V = Matrix(cone.cache.viw_eigen.vectors') # TODO in-place; allows column access

    col_idx = 1
    for j in 1:d, i in 1:j
        V_i = V[:, i]
        if i == j
            mat = V_i * V_i'
        else
            V_j = V[:, j]
            mat = V_j * V_i'
            mat = mat + mat'
            mat .*= rt2i
        end

        mat .*= dot_mat
        mat = V' * mat * V
        @views smat_to_svec!(Hww[:, col_idx], mat, rt2)
        col_idx += 1

        # TODO refac below part

        if cone.cache.is_complex && (i != j)
            V_j = V[:, j]
            mat = V_j * V_i'
            mat .*= rt2i * im
            mat = mat + mat'

            mat .*= dot_mat
            mat = V' * mat * V
            @views smat_to_svec!(Hww[:, col_idx], mat, rt2)
            col_idx += 1
        end
    end

    return Hww
end
