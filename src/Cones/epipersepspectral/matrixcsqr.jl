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
    ϕ::T
    ζ::T
    ζi::T
    σ::T
    ∇h::Vector{T}
    ∇2h::Vector{T}
    ∇3h::Vector{T}
    diff_mat::Matrix{T} # first difference matrix # TODO maybe rename to Δh
    diff_ten::Matrix{T} # some elements of second difference tensor # TODO maybe rename to Δ2h

    w1::Matrix{R} # aux
    w2::Matrix{R} # aux
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
    cache.∇h = zeros(T, d)
    cache.∇2h = zeros(T, d)
    cache.∇3h = zeros(T, d)
    cache.diff_mat = zeros(T, d, d)
    cache.diff_ten = zeros(T, d, svec_length(d))
    cache.w1 = zeros(R, d, d)
    cache.w2 = zeros(R, d, d)
    # cache.m = zeros(R, d, d)
    # cache.α = zeros(R, d, d)
    # cache.γ = zeros(R, d, d)
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

    # TODO in-place; refactor this common V * D * V' structure, where D is diagonal or hermitian
    temp = viw_vecs * Diagonal(ζi * ∇h - inv.(cache.w_λ)) * viw_vecs'

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @views smat_to_svec!(cone.grad[3:end], temp, cache.rt2)

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
                # println("small denom") # TODO
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

    ζ = cache.ζ
    ζi = cache.ζi
    ζi2 = abs2(ζi)
    σ = cache.σ
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    ζivi = ζi / v
    w_λ = cache.w_λ
    diff_mat = Symmetric(cache.diff_mat, :U)


    # Huu
    H[1, 1] = ζi2

    # Huv
    H[1, 2] = -ζi2 * σ

    # Hvv
    @inbounds H[2, 2] = v^-2 + abs2(ζi * σ) + ζivi * sum(abs2(viw_λ[j]) * ∇2h[j] for j in 1:d)

    # Huw
    const1 = -ζi2 * ∇h
    Huw = viw_vecs * Diagonal(const1) * viw_vecs'
    @views smat_to_svec!(H[1, 3:end], Huw, rt2)

    # Hvw
    Hvw = viw_vecs * Diagonal(-ζivi * ∇2h .* viw_λ - σ * const1) * viw_vecs'
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
    temp1 = -ζ * H[1, 3:end] # TODO in place
    mul!(Hww, temp1, temp1', true, true)



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
    w_λ = cache.w_λ
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
        diag_λi = Diagonal(inv.(w_λ))
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
    @inbounds c3 = abs2(inv(v)) + σ * c1 + sum((viw[i] + c1 * α[i] - γ[i]) * w1[i] for i in 1:cone.d)
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

    m = inv.(ζivi * ∇2h + abs2.(inv.(w_λ)))
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

        diff_ten[i, svec_idx(k, j)] = diff_ten[j, svec_idx(k, i)] = diff_ten[k, svec_idx(j, i)] = t
    end

    cone.correction_aux_updated = true
end

function correction(cone::EpiPerSepSpectral{<:MatrixCSqr{T}}, dir::AbstractVector{T}) where T
    cone.correction_aux_updated || update_correction_aux(cone)
    v = cone.point[2]
    corr = cone.correction
    cache = cone.cache
    ζi = cache.ζi
    viw_vecs = cache.viw_eigen.vectors
    viw_λ = cache.viw_eigen.values
    w_λ = cache.w_λ
    σ = cache.σ
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    ∇3h = cache.∇3h
    diff_mat = Symmetric(cache.diff_mat, :U)
    diff_ten = cache.diff_ten
    ζivi = ζi / v

    r = Hermitian(cache.w1, :U)
    # ξ = Hermitian(zeros(R, d, d))

    p = dir[1]
    q = dir[2]
    @views svec_to_smat!(r.data, dir[3:end], cache.rt2)
    r_vecs = Hermitian(viw_vecs' * r * viw_vecs)

    viq = q / v
    ξ_vecs = Hermitian(r_vecs - Diagonal(q * viw_λ))
    ξb = ζivi * Hermitian(diff_mat .* ξ_vecs)
    ζiχ = ζi * (p - σ * q - dot(∇h, diag(r_vecs)))
    ξbξ = dot(ξb, ξ_vecs) / 2
    c1 = -ζi * (ζiχ^2 + ξbξ)

    ξ_vecs.data ./= v

    diff_dot = similar(ξ_vecs.data) # TODO
    d = cone.d
    col = 1
    @inbounds for j in 1:d, i in 1:j
        diff_dot[i, j] = sum(ξ_vecs[k, i]' * ξ_vecs[k, j] * diff_ten[k, col] for k in 1:d)
        col += 1
    end

    w_aux = ξb
    w_aux.data .*= ζiχ + viq
    @. w_aux.data -= ζi * diff_dot

    diag_λi = Diagonal(inv.(w_λ))
    w_aux2 = diag_λi * r_vecs * diag_λi * r_vecs * diag_λi # TODO compute with outer prod?
    w_aux3 = c1 * Diagonal(∇h) + w_aux + w_aux2

    corr[1] = -c1
    corr[2] = c1 * σ - dot(viw_λ, diag(w_aux)) + (ξbξ + viq^2) / v
    prod_w = viw_vecs * w_aux3 * viw_vecs' # TODO in-place
    @views smat_to_svec!(corr[3:end], prod_w, cache.rt2)

    return corr
end
