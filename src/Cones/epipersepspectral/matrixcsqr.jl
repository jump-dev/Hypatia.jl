#=
matrix cone of squares, i.e. 𝕊₊ᵈ for d ≥ 1, with rank d
=#

struct MatrixCSqr{T <: Real, R <: RealOrComplex{T}} <: ConeOfSquares{T} end

vector_dim(::Type{<:MatrixCSqr{T, T} where {T <: Real}}, d::Int) = svec_length(d)
vector_dim(::Type{<:MatrixCSqr{T, Complex{T}} where {T <: Real}}, d::Int) = d^2

mutable struct MatrixCSqrCache{T <: Real, R <: RealOrComplex{T}} <: CSqrCache{T}
    is_complex::Bool
    rt2::T
    w::Matrix{R}
    viw::Matrix{R}
    viw_eigen
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
    cache.∇h_viw = zeros(T, d)
    cache.∇2h_viw = zeros(T, d)
    cache.∇3h_viw = zeros(T, d)
    cache.diff_mat = zeros(T, d, d)
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

function update_feas(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.feas_updated
    cache = cone.cache
    v = cone.point[2]

    cone.is_feas = false
    if v > eps(T)
        w = svec_to_smat!(cache.w, cone.w_view, cache.rt2)
        w_chol = cholesky(Hermitian(w, :U), check = false) # TODO use in-place, check whether it is faster to do this before an eigdecomp
        if isposdef(w_chol)
            viw = cache.viw
            @. viw = w / v
            # TODO other options? eigen(A; permute::Bool=true, scale::Bool=true, sortby) -> Eigen
            viw_eigen = cache.viw_eigen = eigen(Hermitian(viw, :U), sortby = nothing) # TODO use in-place
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

# TODO check if this is faster or slower than only using nbhd check
function is_dual_feas(cone::EpiPerSepSpectral{MatrixCSqr{T, R}}) where {T, R}
    u = cone.dual_point[1]
    (u < eps(T)) && return false
    @views w = cone.dual_point[3:end]
    uiw = zeros(R, cone.d, cone.d)
    if h_conj_dom_pos(cone.h)
        # use cholesky to check conjugate domain feasibility
        # TODO check whether it is faster to do this before an eigdecomp
        svec_to_smat!(uiw, w, cone.cache.rt2)
        w_chol = cholesky!(Hermitian(uiw, :U), check = false)
        isposdef(w_chol) || return false
    end

    svec_to_smat!(uiw, w, cone.cache.rt2)
    # TODO in-place:
    @. uiw /= u
    uiw_eigen = eigen(Hermitian(uiw, :U), sortby = nothing)
    uiw_λ = uiw_eigen.values
    # h_conj_dom(uiw_λ, cone.h) || return false
    return (cone.dual_point[2] - u * h_conj(uiw_λ, cone.h) > eps(T))
end

function update_grad(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
    @assert !cone.grad_updated && cone.is_feas
    grad = cone.grad
    v = cone.point[2]
    cache = cone.cache
    ζi = cache.ζi = inv(cache.ζ)
    viw_λ = cache.viw_eigen.values
    ∇h_viw = cache.∇h_viw
    h_der1(∇h_viw, viw_λ, cone.h)
    cache.σ = cache.ϕ - dot(viw_λ, ∇h_viw) # TODO guessed, just dots vectors

    viw_vecs = cache.viw_eigen.vectors
    temp = viw_vecs * Diagonal(ζi * ∇h_viw - inv.(v .* viw_λ)) * viw_vecs' # TODO combines the ∇h_viw and wi

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @views smat_to_svec!(cone.grad[3:end], temp, cache.rt2)

    cone.grad_updated = true
    return grad
end

function update_hess(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
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
    h_der2(∇2h_viw, viw_λ, cone.h)
    ζivi = ζi / v
    ζiσ = ζi * σ
    rt2 = cache.rt2
    w_λ = v * viw_λ


# TODO refac out?
    diff_mat = cache.diff_mat
    rteps = sqrt(eps(T))
    for j in 1:d
        viw_λ_j = viw_λ[j]
        ∇h_viw_j = ∇h_viw[j]
        ∇2h_viw_j = ∇2h_viw[j]
        for i in 1:(j - 1)
            denom = viw_λ[i] - viw_λ_j
            if abs(denom) < rteps
                # println("small denom") # TODO
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
    @views smat_to_svec!(H[1, 3:end], Huw, rt2)
    temp1 = H[1, 3:end]

    # Hvw
    Hvw = viw_vecs * Diagonal(-ζivi * ∇2h_viw .* viw_λ - σ * const1) * viw_vecs'
    @views smat_to_svec!(H[2, 3:end], Hvw, rt2)

    # Hww
    @views Hww = H[3:end, 3:end]

    # Hww kron parts
    tempa = ζivi * diff_mat + inv.(w_λ) * inv.(w_λ)'
    # @show isposdef(tempa) # true



    # TODO refac, in-place, simplify, precompute parts
    rt2i = inv(rt2)
    col_idx = 1
    for j in 1:d, i in 1:j
        vecsi = viw_vecs[i, :] # TODO to be efficient, make a transposed copy of vecs and index columns
        if i == j
            mat = vecsi * vecsi'
        else
            vecsj = viw_vecs[j, :]
            mat = vecsi * vecsj'
            mat = mat + mat'
            mat .*= rt2i
        end

        mat .*= tempa
        mat = viw_vecs * transpose(mat) * viw_vecs'
        @views smat_to_svec!(Hww[:, col_idx], mat, rt2)
        col_idx += 1

        if cache.is_complex && (i != j)
            vecsj = viw_vecs[j, :]
            mat = vecsi * vecsj'
            mat .*= rt2i * im
            mat = mat + mat'

            mat .*= tempa
            mat = viw_vecs * transpose(mat) * viw_vecs'
            @views smat_to_svec!(Hww[:, col_idx], mat, rt2)
            col_idx += 1
        end
    end



    # Hww vector outer prod part
    temp1 .*= -ζ
    mul!(Hww, temp1, temp1', true, true)



    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSepSpectral{MatrixCSqr{T, R}}) where {T, R}
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

function update_inv_hess(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}) where T
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
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    ζivi = ζi / v
    ζiσ = ζi * σ
    rt2 = cache.rt2
    diff_mat = Hermitian(cache.diff_mat, :U)

    ζivi = ζi / v
    ζivi2 = ζivi / v
    w_λ = v * viw_λ

    m = inv.(ζivi * ∇2h_viw + abs2.(inv.(w_λ)))
    α1 = m .* ∇h_viw
    α = viw_vecs * Diagonal(α1) * viw_vecs' # TODO can use sqrt
    β = dot(∇h_viw, α1)
    ζ2β = ζ^2 + β

    w∇2h_viw = ζivi2 * w_λ .* ∇2h_viw
    γ1 = m .* w∇2h_viw
    γ = viw_vecs * Diagonal(γ1) * viw_vecs' # TODO maybe can use sqrt
    c1 = (σ + dot(∇h_viw, γ1)) / ζ2β

    c3 = ζi2 * σ
    c4 = ζi2 * β
    Zuu = ζi2 - c4 / ζ2β
    Zvu = -c3 + c1 * c4 - ζi2 * dot(γ1, ∇h_viw)
    Zvv = (inv(v) + dot(w_λ, w∇2h_viw)) / v + abs2(ζi * σ) + dot(w∇2h_viw - c3 * ∇h_viw, c1 * α1 - γ1)

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
    tempa = inv.(ζivi * diff_mat + inv.(w_λ) * inv.(w_λ)')
    # @show isposdef(tempa) # true

    # TODO refac, in-place, simplify, precompute parts
    rt2i = inv(rt2)
    col_idx = 1
    for j in 1:d, i in 1:j
        vecsi = viw_vecs[i, :] # TODO to be efficient, make a transposed copy of vecs and index columns
        if i == j
            mat = vecsi * vecsi'
        else
            vecsj = viw_vecs[j, :]
            mat = vecsi * vecsj'
            mat = mat + mat'
            mat .*= rt2i
        end

        mat .*= tempa
        mat = viw_vecs * transpose(mat) * viw_vecs'
        @views smat_to_svec!(Hiww[:, col_idx], mat, rt2)
        col_idx += 1

        if cache.is_complex && (i != j)
            vecsj = viw_vecs[j, :]
            mat = vecsi * vecsj'
            mat .*= rt2i * im
            mat = mat + mat'

            mat .*= tempa
            mat = viw_vecs * transpose(mat) * viw_vecs'
            @views smat_to_svec!(Hiww[:, col_idx], mat, rt2)
            col_idx += 1
        end
    end

    # Hiww vector outer prod part
    Hiww .+= yu * (αvec - HiuW)' - yv * HivW'

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSepSpectral{MatrixCSqr{T, R}}) where {T, R}
    # cone.hess_aux_updated || update_hess_aux(cone) # TODO
    hess(cone) # TODO
    @assert cone.hess_updated
    v = cone.point[2]
    cache = cone.cache



    Hi = update_inv_hess(cone)
    mul!(prod, Hi, arr)

    # # TODO @inbounds
    # for j in 1:size(arr, 2)
    #
    #
    # end

    return prod
end

function correction(cone::EpiPerSepSpectral{MatrixCSqr{T, R}}, dir::AbstractVector{T}) where {T, R}
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
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    diff_mat = Hermitian(cache.diff_mat, :U)
    d = cone.d


# TODO for square, h_der3 is 0 and h_der2 is constant, so can skip much of this


    ∇3h_viw = cache.∇3h_viw
    h_der3(∇3h_viw, viw_λ, cone.h)

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
            # println("small denom 1") # TODO
            if abs(denom_ik) < rteps
                # println("small denom 2") # TODO
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
    ζi2χpviq = ζi * (ζiχ + viq)

    ξ_vecs = Hermitian(vi * (r_vecs - Diagonal(q * viw_λ)))
    temp = Hermitian(diff_mat .* ξ_vecs)

    ξbξ = ζi * v * dot(temp, ξ_vecs) / 2
    c1 = ζi * (ζiχ^2 + ξbξ)

    # TODO too inefficient. don't form diff tensor explicitly
    diff_dot = Hermitian([dot(ξ_vecs[:, p], Diagonal(diff_ten[:, p, q]), ξ_vecs[:, q]) for p in 1:d, q in 1:d])

    corr[1] = c1

    corr[2] = -c1 * σ -
        ζi2χpviq * dot(diag(temp), viw_λ) +
        (ξbξ + viq^2) / v +
        ζi * dot(diag(diff_dot), viw_λ)

    diag_λi = Diagonal([inv(v * viw_λ[i]) for i in 1:d])
    prod_w = viw_vecs * (
        -c1 * Diagonal(∇h_viw) +
        ζi2χpviq * temp +
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
