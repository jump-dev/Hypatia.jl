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
    diff_mat::Matrix{T}
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
    cache.diff_mat = zeros(T, d, d)
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

function update_hess(cone::EpiPerSepSpectral{<:MatrixCSqr{T}, F, T}) where {T, F}
    @assert cone.grad_updated && !cone.hess_updated
    d = cone.d
    v = cone.point[2]
    cache = cone.cache
    H = cone.hess.data
    ζ = cache.ζ
    ζi = cache.ζi
    ζi2 = abs2(ζi)
    σ = cache.σ
    # viw = cache.viw
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    @. ∇2h_viw = h_der2(F, viw_λ)
    ζivi = ζi / v
    ζiσ = ζi * σ
    wi = cache.wi = inv(cache.w_chol) # TODO maybe not needed


    diff_mat = cache.diff_mat
    rteps = sqrt(eps(T))
    for j in 1:d
        viw_λ_j = viw_λ[j]
        ∇h_viw_j = ∇h_viw[j]
        ∇2h_viw_j = ∇2h_viw[j]
        for i in 1:(j - 1)
            denom = viw_λ[i] - viw_λ_j
            if abs(denom) < rteps
                println("small denom") # TODO
                diff_mat[i, j] = (∇2h_viw[i] + ∇2h_viw_j) / 2 # NOTE or take ∇2h at the average (viw[i] + viw[j]) / 2
            else
                diff_mat[i, j] = (∇h_viw[i] - ∇h_viw_j) / denom
            end
        end
        diff_mat[j, j] = ∇2h_viw_j
    end
    diff_mat = Hermitian(diff_mat, :U)


    # Huu
    H[1, 1] = ζi2

    # Huv
    H[1, 2] = -ζi2 * σ

    # Hvv
    H[2, 2] = v^-2 + abs2(ζi * σ) + ζivi * sum(abs2(viw_λ[j]) * ∇2h_viw[j] for j in 1:d)

    # Huw
    const1 = -ζi^2 * ∇h_viw
    Huw = viw_vecs * Diagonal(const1) * viw_vecs'
    @views smat_to_svec!(H[1, 3:end], Huw, cache.rt2)
    temp1 = H[1, 3:end]

    # Hvw
    Hvw = viw_vecs * Diagonal(-ζivi * ∇2h_viw .* viw_λ - σ * const1) * viw_vecs'
    @views smat_to_svec!(H[2, 3:end], Hvw, cache.rt2)

    # Hww
    @views Hww = H[3:end, 3:end]
    symm_kron(Hww, wi, cache.rt2)
    temp1 .*= -ζ
    mul!(Hww, temp1, temp1', true, true)

    temp2 = similar(Hww)
    temp3 = similar(Hww)
    temp4 = similar(temp1)
    temp5 = similar(Hww)
    symm_kron(temp3, viw_vecs, cache.rt2, upper_only = false)
    smat_to_svec!(temp4, diff_mat, one(T))
    mul!(temp5, temp3, Diagonal(temp4))
    mul!(temp2, temp5, temp3')
    @. Hww += ζivi * temp2

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSepSpectral{MatrixCSqr{T, R}, F}) where {T, R, F}
    # cone.hess_aux_updated || update_hess_aux(cone) # TODO

    hess(cone) # TODO remove
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
    diff_mat = Hermitian(cache.diff_mat, :U)

    # TODO prealloc
    d = cone.d
    r = Hermitian(zeros(R, d, d))
    # ξ = Hermitian(zeros(R, d, d))
    ζivi = ζi * vi

    @inbounds @views for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        svec_to_smat!(r.data, arr[3:end, j], cache.rt2)

        r_vecs = Hermitian(viw_vecs' * r * viw_vecs)

        # χ = get_χ(p, q, r, cone)
        χ = p - cache.σ * q - dot(∇h_viw, diag(r_vecs))
        ζi2χ = ζi2 * χ

        temp = Hermitian(diff_mat .* (r_vecs - Diagonal(q * viw_λ)))

        prod[1, j] = ζi2χ
        prod[2, j] = -σ * ζi2χ - ζivi * dot(diag(temp), viw_λ) + q * vi * vi

        diag_λi = Diagonal([inv(v * viw_λ[i]) for i in 1:d])
        prod_w = viw_vecs * (
            -ζi2χ * Diagonal(∇h_viw) +
            ζivi * temp +
            diag_λi * r_vecs * diag_λi
            ) * viw_vecs'

        smat_to_svec!(prod[3:end, j], prod_w, cache.rt2)
    end

    return prod
end


function correction(cone::EpiPerSepSpectral{<:MatrixCSqr{T, R}, F}, dir::AbstractVector{T}) where {T, R, F}
    # cone.hess_aux_updated || update_hess_aux(cone) # TODO

    hess(cone) # TODO remove
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
    diff_mat = Hermitian(cache.diff_mat, :U)
    d = cone.d

    ∇3h_viw = cache.∇3h_viw
    @. ∇3h_viw = h_der3(F, viw_λ)

    # TODO diff tensor
    # TODO "symmetric", could use a tensor package, or a symmetric matrix of symmetric matrices
    # TODO better to write as an operator though
    diff_ten = zeros(T, d, d, d)
    rteps = sqrt(eps(T))
    for k in 1:d, j in 1:k, i in 1:j
        (viw_λ_i, viw_λ_j, viw_λ_k) = (viw_λ[i], viw_λ[j], viw_λ[k])
        (∇3h_i, ∇3h_j, ∇3h_k) = (∇3h_viw[i], ∇3h_viw[j], ∇3h_viw[k])
        denom_ij = viw_λ_i - viw_λ_j
        denom_ik = viw_λ_i - viw_λ_k

        if abs(denom_ij) < rteps
            println("small denom 1") # TODO
            if abs(denom_ik) < rteps
                println("small denom 2") # TODO
                t = (∇3h_i + ∇3h_j + ∇3h_k) / 6
            else
                t = (diff_mat[i, j] - diff_mat[j, k]) / denom_ik
            end
        else
            t = (diff_mat[i, k] - diff_mat[j, k]) / denom_ij
        end

        diff_ten[i, j, k] = diff_ten[i, k, j] = diff_ten[j, i, k] =
            diff_ten[j, k, i] = diff_ten[k, i, j] = diff_ten[k, j, i] = t
    end


    wi = cache.wi
    corr = cone.correction

    # TODO prealloc
    d = cone.d
    r = Hermitian(zeros(R, d, d))
    # ξ = Hermitian(zeros(R, d, d))

    p = dir[1]
    q = dir[2]
    svec_to_smat!(r.data, dir[3:end], cache.rt2)

    r_vecs = Hermitian(viw_vecs' * r * viw_vecs)

    viq = vi * q
    # χ = get_χ(p, q, r, cone)
    χ = p - cache.σ * q - dot(∇h_viw, diag(r_vecs))
    ζiχ = ζi * χ
    ζiχpviq = ζiχ + viq

    ξ_vecs = Hermitian(vi * (r_vecs - Diagonal(q * viw_λ)))
    temp = Hermitian(diff_mat .* ξ_vecs)

    ξbξ = ζi * v * dot(temp, ξ_vecs) / 2
    c1 = ζi * (ζiχ^2 + ξbξ)

    # TODO too inefficient. don't form diff tensor explicitly
    diff_dot = Hermitian([dot(ξ_vecs[:, p], Diagonal(diff_ten[:, p, q]), ξ_vecs[:, q]) for p in 1:d, q in 1:d])

    corr[1] = c1

    corr[2] = -c1 * σ -
        ζi * ζiχpviq * dot(diag(temp), viw_λ) +
        (ξbξ + viq^2) / v +
        ζi * dot(diag(diff_dot), viw_λ)

    diag_λi = Diagonal([inv(v * viw_λ[i]) for i in 1:d])
    prod_w = viw_vecs * (
        -c1 * Diagonal(∇h_viw) +
        ζi * ζiχpviq * temp +
        -ζi * diff_dot +
        diag_λi * r_vecs * diag_λi * r_vecs * diag_λi
        ) * viw_vecs'

    @views smat_to_svec!(corr[3:end], prod_w, cache.rt2)

    return corr
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
