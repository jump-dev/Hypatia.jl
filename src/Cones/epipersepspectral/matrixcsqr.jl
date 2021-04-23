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

    viw_vecs = cache.viw_eigen.vectors
    temp = viw_vecs * Diagonal(ζi * ∇h_viw - inv.(v .* viw_λ)) * viw_vecs' # TODO combines the ∇h_viw and wi

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @views smat_to_svec!(cone.grad[3:end], temp, cache.rt2)

    cone.grad_updated = true
    return grad
end

function update_hess(cone::EpiPerSepSpectral{MatrixCSqr{T, R}, F, T}) where {T, R, F}
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
    rt2 = cache.rt2

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
    @views smat_to_svec!(H[1, 3:end], Huw, rt2)
    temp1 = H[1, 3:end]

    # Hvw
    Hvw = viw_vecs * Diagonal(-ζivi * ∇2h_viw .* viw_λ - σ * const1) * viw_vecs'
    @views smat_to_svec!(H[2, 3:end], Hvw, rt2)

    # Hww
    @views Hww = H[3:end, 3:end]

    # Hww kron parts
    eigw = v * viw_λ
    tempa = Symmetric(ζivi * diff_mat + inv.(eigw) * inv.(eigw)')
    @assert all(>(0), tempa)


    temp1 .*= -ζ
    vecouter = temp1 * temp1'


# kww = kron(transpose(viw_vecs), viw_vecs)
# bigmat = kww * Diagonal(vec(tempa)) * kww'
# @show bigmat
#
#     D = Diagonal(vec(tempa))
#     Hww .= 0
#
#     col_idx = 1
#     @inbounds for l in 1:d
#         for k in 1:(l - 1)
#             row_idx = 1
#             for j in 1:d
#                 (row_idx > col_idx) && break
#                 for i in 1:(j - 1)
#                     # Hww[row_idx, col_idx] = mat[i, k] * mat[j, l] + mat[i, l] * mat[j, k]
#                     Hww[row_idx, col_idx] = -1
#                     row_idx += 1
#                 end
#                 # Hww[row_idx, col_idx] = rt2 * mat[j, k] * mat[j, l]
#                 Hww[row_idx, col_idx] = -2
#                 row_idx += 1
#             end
#             col_idx += 1
#         end
#
#         row_idx = 1
#         for j in 1:d
#             (row_idx > col_idx) && break
#             for i in 1:(j - 1)
#                 # Hww[row_idx, col_idx] = rt2 * mat[i, l] * mat[j, l]
#                 Hww[row_idx, col_idx] = -3
#                 row_idx += 1
#             end
#             # Hww[row_idx, col_idx] = mat[j, l] * mat[l, j]
#             @show svec_idx(j, l), svec_idx(l, j)
#             Hww[row_idx, col_idx] = bigmat[svec_idx(j, l), svec_idx(l, j)]
#             @show Hww[row_idx, col_idx]
#             row_idx += 1
#         end
#         col_idx += 1
#     end

#
#
#     if cache.is_complex
#         tempa = Hermitian(ComplexF64.(tempa - tempa*im))
#     else
#         tempa = Symmetric(tempa)
#     end
#     # @show isposdef(tempa) # true
#

    dim2 = svec_length(d)
    temp3 = zeros(R, dim2, dim2)
    # temp3 = similar(Hww)

    temp4 = zeros(T, dim2)
    smat_to_svec!(temp4, tempa, one(T))
    # @show temp4
    # temp4 = [tempa[i, j] for j in 1:d for i in 1:j]




    symm_kron(temp3, viw_vecs, rt2, upper_only = false)
    outer = temp3 * temp3'
    @show outer ≈ I
    println("outer")
    display(UpperTriangular(round.(outer, digits=6)))

#
#     @show viw_vecs
#     krvecs = kron(transpose(viw_vecs), viw_vecs)
#     @show krvecs
#     @show temp3
# #
#     try1 = krvecs * Diagonal(vec(tempa)) * krvecs'
# @show try1
    try1 = Hermitian(temp3 * Diagonal(temp4) * temp3')




    if !cache.is_complex
        Hww .+= try1
    else

    try1a = sqrt(Diagonal(temp4)) * temp3'
    try1b = Hermitian(try1a' * try1a)
    @show try1b ≈ try1
    println()
    try1 = try1a

    dot1(a,b) = dot(try1[:, a], try1[:, b])
    dot2(i,k,j,l) = dot1(svec_idx(i,k), svec_idx(j,l))

    Hww .= 0

    col_idx = 1
    col_idx2 = 1
# @inbounds
    for i in 1:d, j in 1:i
        row_idx = 1
        row_idx2 = 1
        if i == j
            for i2 in 1:d, j2 in 1:i2
                if i2 == j2
                    @show 1, row_idx2, col_idx2
                    # Hww[row_idx, col_idx] = abs(dot1(row_idx2, col_idx2))
                    k = i2
                    l = j2
                    @show i,j,k,l
                    Hww[row_idx, col_idx] = (dot2(i,k,j,l) + dot2(i,l,j,k) + dot2(j,l,i,k) + dot2(j,k,i,l)) / 4
                    # Hww[row_idx, col_idx] = (X[i,k] * Y[j,l] + X[i,l] * Y[j,k] + X[j,l] * Y[i,k] + X[j,k] * Y[i,l]) / 4
                else
                    @show 2, row_idx2, col_idx2
                    c = dot1(row_idx2, col_idx2)
                    Hww[row_idx, col_idx] = real(c)
                    row_idx += 1
                    Hww[row_idx, col_idx] = -imag(c)
                end
                row_idx += 1
                row_idx2 += 1
                (row_idx > col_idx) && break
            end
            col_idx += 1
            col_idx2 += 1
        else
            for i2 in 1:d, j2 in 1:i2
                if i2 == j2
                    @show 3, row_idx2, col_idx2
                    c = dot1(row_idx2, col_idx2)
                    Hww[row_idx, col_idx] = real(c)
                    Hww[row_idx, col_idx + 1] = -imag(c)
                else
                    @show 4, row_idx2, col_idx2
                    @show i, j, i2, j2
                    @show svec_idx(i2,i), svec_idx(j,j2), svec_idx(j2,i), svec_idx(j,i2)
                    # b1 = try1[row_idx2, col_idx2]
                    b1 = dot1(svec_idx(j2,i), svec_idx(j,i2))
                    b2 = dot1(svec_idx(i2,i), svec_idx(j,j2))
                    # b2 = dot1(col_idx2, row_idx2)
                    @show b1
                    @show b2
                    b1 -= hypot(b2)
                    c1 = b1 + b2
                    Hww[row_idx, col_idx] = real(c1)
                    Hww[row_idx, col_idx + 1] = imag(c1)
                    row_idx += 1
                    c2 = b1 - b2
                    Hww[row_idx, col_idx] = -imag(c2)
                    Hww[row_idx, col_idx + 1] = real(c2)
                end
                row_idx += 1
                row_idx2 += 1
                (row_idx > col_idx) && break
            end
            col_idx += 2
            col_idx2 += 1
        end
    end

@show try1a


    end
#     # @show try1
#     # @show try2

    # temp3 .= 0
    # symm_kron(temp3, inv(cache.viw_eigen), rt2)
    # @show Hermitian(temp3)
    # temp4 = zero(temp3)
    # symm_kron(temp4, cache.viw, rt2)
    # @show Hermitian(temp4)
    # @show Hermitian(temp3) * Hermitian(temp4)


    # rnd = rand(dim2)
    # Rnd =
    # try1 = vec(viw_vecs * rnd * viw_vecs)
    # try2 = temp3 * [rnd[i, j] for j in 1:d for i in 1:j]
    # @show try1
    # @show try2

    # smat_to_svec!(temp4, tempa, one(T))
    # mul!(temp5, temp3, Diagonal(temp4))
    # mul!(Hww, temp5, temp3')
    # HwwR = temp3 * Diagonal(temp4) * temp3'
    # @show temp4
    # HwwR2 = temp3 * Diagonal(sqrt.(temp4))
    # HwwR3 = HwwR2 * HwwR2'
    # @show HwwR3 - HwwR
    # Hww .+= HwwR
    #
    # # symm_kron(temp3, viw_vecs, rt2, upper_only = false)
    # # @show temp3
    # #
    # # @show HwwR
    #
    # # Hww vector outer prod part
    # temp1 .*= -ζ
    # outer = temp1 * temp1'
    # # @show outer
    mul!(Hww, temp1, temp1', true, true)
    @show vecouter

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
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    diff_mat = Hermitian(cache.diff_mat, :U)
    d = cone.d


# TODO for square, h_der3 is 0 and h_der2 is constant, so can skip much of this


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
