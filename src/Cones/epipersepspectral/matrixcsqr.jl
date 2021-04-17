#=
TODO

matrix cone of squares, i.e. 𝕊₊ᵈ for d ≥ 1, with rank d
=#


struct MatrixCSqr{T <: Real, R <: RealOrComplex{T}} <: ConeOfSquares{T} end

vector_dim(::Type{<:MatrixCSqr{T, T} where {T <: Real}}, d::Int) = svec_length(d)
vector_dim(::Type{<:MatrixCSqr{T, Complex{T}} where {T <: Real}}, d::Int) = d^2

mutable struct MatrixCSqrCache{T <: Real, R <: RealOrComplex{T}} <: CSqrCache{T}
    is_complex::Bool
    rt2::T
    w::Matrix{R}
    w_chol
    viw::Matrix{R} # TODO is it needed?
    viw_eigen
    wi::Matrix{R}
    ϕ::T
    ζ::T
    ζi::T
    σ::T
    ∇h_viw::Vector{T}
    ∇2h_viw::Vector{T}
    ∇3h_viw::Vector{T}
    # ζi∇h_viw::Vector{T}
    MatrixCSqrCache{T, R}() where {T <: Real, R <: RealOrComplex{T}} = new{T, R}()
end

function setup_csqr_cache(cone::EpiPerSepSpectral{MatrixCSqr{T, R}}) where {T, R}
    cone.cache = cache = MatrixCSqrCache{T, R}()
    cache.is_complex = (R <: Complex{T})
    cache.rt2 = sqrt(T(2))
    d = cone.d
    cache.w = zeros(R, d, d)
    cache.viw = zeros(R, d, d)
    cache.wi = zeros(R, d, d)
    cache.∇h_viw = zeros(T, d)
    cache.∇2h_viw = zeros(T, d)
    cache.∇3h_viw = zeros(T, d)
    # cache.ζi∇h_viw = zeros(T, d)
    return
end

function set_initial_point(arr::AbstractVector, cone::EpiPerSepSpectral{<:MatrixCSqr, F}) where F
    (arr[1], arr[2], w0) = get_initial_point(F, cone.d)
    @views fill!(arr[3:end], 0)
    incr = (cone.cache.is_complex ? 2 : 1)
    idx = 3
    @inbounds for i in 1:cone.d
        arr[idx] = 1
        idx += incr * i + 1
    end
    return arr
end

# TODO can do a cholesky of w (fast) to check feas first (since logdet part only uses w), then eigen of w/v instead of w
function update_feas(cone::EpiPerSepSpectral{<:MatrixCSqr{T}, F, T}) where {T, F}
    @assert !cone.feas_updated
    cache = cone.cache
    v = cone.point[2]

    cone.is_feas = false
    if v > eps(T)
        w = svec_to_smat!(cache.w, cone.w_view, cache.rt2)
        w_chol = cache.w_chol = cholesky(Hermitian(w, :U), check = false) # TODO use in-place
        if isposdef(w_chol)
            viw = cache.viw
            @. viw = w / v
            # TODO other options? eigen(A; permute::Bool=true, scale::Bool=true, sortby) -> Eigen
            viw_eigen = cache.viw_eigen = eigen(Hermitian(viw, :U), sortby = nothing) # TODO use in-place
            viw_λ = viw_eigen.values
            if all(>(eps(T)), viw_λ)
                cache.ϕ = h_val(F, viw_λ)
                cache.ζ = cone.point[1] - v * cache.ϕ
                cone.is_feas = (cache.ζ > eps(T))
            end
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

# TODO check if this is faster or slower than only using nbhd check
function is_dual_feas(cone::EpiPerSepSpectral{MatrixCSqr{T, R}, F, T}) where {T, R, F}
    u = cone.dual_point[1]
    (u < eps(T)) && return false
    @views w = cone.dual_point[3:end]
    # TODO in-place:
    uiw = zeros(R, cone.d, cone.d)
    svec_to_smat!(uiw, w, cone.cache.rt2)
    @. uiw /= u
    uiw_eigen = eigen(Hermitian(uiw, :U), sortby = nothing)
    uiw_λ = uiw_eigen.values
    h_conj_dom(F, uiw_λ) || return false
    v = cone.dual_point[2]
    return (v - u * h_conj(F, uiw_λ) > eps(T))
end

function update_grad(cone::EpiPerSepSpectral{<:MatrixCSqr, F}) where F
    @assert !cone.grad_updated && cone.is_feas
    grad = cone.grad
    v = cone.point[2]
    cache = cone.cache
    ζi = cache.ζi = inv(cache.ζ)
    viw_λ = cache.viw_eigen.values
    ∇h_viw = cache.∇h_viw
    @. ∇h_viw = h_der1(F, viw_λ)
    cache.σ = cache.ϕ - dot(viw_λ, ∇h_viw) # TODO guessed, just dots vectors
    # cache.wi = inv(cache.w_chol)

    viw_vecs = cache.viw_eigen.vectors
    temp = viw_vecs * Diagonal(ζi * ∇h_viw - inv.(v .* viw_λ)) * viw_vecs' # TODO combines the ∇h_viw and wi
    # @. temp -= cache.wi

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @views smat_to_svec!(cone.grad[3:end], temp, cache.rt2)

    cone.grad_updated = true
    return grad
end

function update_hess(cone::EpiPerSepSpectral{<:MatrixCSqr, F}) where F
    @assert cone.grad_updated && !cone.hess_updated
    d = cone.d
    v = cone.point[2]
    cache = cone.cache
    H = cone.hess.data
    ζi = cache.ζi
    ζi2 = abs2(ζi)
    σ = cache.σ
    # viw = cache.viw
    viw_λ = cache.viw_eigen.values
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    @. ∇2h_viw = h_der2(F, viw_λ)
    # ζi∇h_viw = cache.ζi∇h_viw
    # ζivi = ζi / v
    # ζiσ = ζi * σ
    cache.wi = inv(cache.w_chol) # TODO maybe not needed

    # # Huu
    # H[1, 1] = ζi2
    #
    # # Huv
    # H[1, 2] = -ζi2 * σ
    #
    # # Hvv start
    # Hvv = v^-2 + abs2(ζi * σ)
    #
    # @inbounds for j in 1:d
    #     ζi∇h_viw_j = ζi∇h_viw[j]
    #     term_j = ζivi * viw[j] * ∇2h_viw[j]
    #     Hvv += viw[j] * term_j
    #     j2 = 2 + j
    #
    #     # Huw
    #     H[1, j2] = -ζi * ζi∇h_viw_j
    #
    #     # Hvw
    #     H[2, j2] = ζiσ * ζi∇h_viw_j - term_j
    #
    #     # Hww
    #     for i in 1:(j - 1)
    #         H[2 + i, j2] = ζi∇h_viw_j * ζi∇h_viw[i]
    #     end
    #     H[j2, j2] = abs2(ζi∇h_viw_j) + ζivi * ∇2h_viw[j] + abs2(wi[j])
    # end
    #
    # # Hvv end
    # H[2, 2] = Hvv

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSepSpectral{MatrixCSqr{T, R}, F}) where {T, R, F}
    # cone.hess_aux_updated || update_hess_aux(cone) # TODO
    @assert cone.hess_updated
    v = cone.point[2]
    vi = inv(v)
    cache = cone.cache
    w = Hermitian(cache.w)
    ζi = cache.ζi
    ζi2 = abs2(ζi)
    viw = Hermitian(cache.viw)
    σ = cache.σ
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    wi = Hermitian(cache.wi, :U)
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values

    # TODO prealloc
    d = cone.d
    r = Hermitian(zeros(R, d, d))
    ξ = Hermitian(zeros(R, d, d))
    # ζivi = ζi * vi
    ∇h_viw_mat = Hermitian(viw_vecs * Diagonal(cache.∇h_viw) * viw_vecs')

    diff_mat = zeros(T, d, d)
    rteps = sqrt(eps(T))
    for j in 1:d
        viw_λ_j = viw_λ[j]
        ∇h_viw_j = ∇h_viw[j]
        for i in 1:(j - 1)
            denom = viw_λ[i] - viw_λ_j
            (abs(denom) < rteps) && println("small denom") # TODO
            diff_mat[i, j] = (∇h_viw[i] - ∇h_viw_j) / denom
        end
        diff_mat[j, j] = ∇2h_viw[j]
    end
    diff_mat = Hermitian(diff_mat, :U)


    @inbounds @views for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        svec_to_smat!(r.data, arr[3:end, j], cache.rt2)

        viq = q * vi
        @. ξ.data = vi * (r.data - viq * w.data) # TODO could just do vecs here

        # χ = get_χ(p, q, r, cone)
        χ = p - cache.σ * q - dot(∇h_viw_mat, r)
        ζi2χ = ζi2 * χ

        temp = Hermitian(ζi * (diff_mat .* ξ))

        prod[1, j] = ζi2χ
        prod[2, j] = -σ * ζi2χ - dot(viw_λ, diag(temp)) + viq * vi
        # TODO wrong:
        prod_r = -ζi2χ * ∇h_viw_mat + viw_vecs * temp * viw_vecs' + wi * r * wi
        smat_to_svec!(prod[3:end, j], prod_r, cache.rt2)
    end

    return prod
end

# function get_χ(
#     p::T,
#     q::T,
#     r::AbstractMatrix{T},
#     cone::EpiPerSepSpectral{<:MatrixCSqr{T}},
#     ) where {T <: Real}
#     cache = cone.cache
#     # TODO precompute vecs * cache.∇h_viw * vecs'
#     ∇h_viw_mat = cache.vecs * Diagonal(cache.∇h_viw) * cache.vecs'
#     return p - cache.σ * q - dot(∇h_viw_mat, r)
# end
