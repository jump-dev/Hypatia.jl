"""
$(TYPEDEF)

Epigraph of real or complex infinity norm cone of dimension `dim`.

    $(FUNCTIONNAME){T, R}(dim::Int, use_dual::Bool = false)
"""
mutable struct EpiNormInf{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    is_complex::Bool

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
    inv_hess_aux_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, SparseMatrixCSC{T, Int}}
    inv_hess::Symmetric{T, Matrix{T}}

    w::Vector{R}
    mu::Vector{R}
    zeta::Vector{T}
    cu::T
    Zu::T
    wumzi::Vector{R}
    zti::Vector{T}

    s1::Vector{T}
    s2::Vector{T}
    w1::Vector{R}
    w2::Vector{R}

    function EpiNormInf{T, R}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.is_complex = (R <: Complex)
        @assert !cone.is_complex || iseven(dim - 1)
        cone.d = (cone.is_complex ? div(dim - 1, 2) : dim - 1)
        return cone
    end
end

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_aux_updated = false)

use_sqrt_hess_oracles(::Int, cone::EpiNormInf) = false

function setup_extra_data!(
    cone::EpiNormInf{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    d = cone.d
    cone.w = zeros(R, d)
    cone.mu = zeros(R, d)
    cone.zeta = zeros(T, d)
    cone.wumzi = zeros(R, d)
    cone.s1 = zeros(T, d)
    cone.s2 = zeros(T, d)
    cone.w1 = zeros(R, d)
    if cone.is_complex
        cone.w2 = zeros(R, d)
    else
        cone.zti = zeros(T, d)
    end
    return cone
end

get_nu(cone::EpiNormInf) = 1 + cone.d

function set_initial_point!(
    arr::AbstractVector{T},
    cone::EpiNormInf{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormInf{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views vec_copyto!(cone.w, cone.point[2:end])
        cone.is_feas = (u - maximum(abs, cone.w) > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiNormInf{T}) where T
    dp = cone.dual_point
    u = dp[1]
    if u > eps(T)
        if cone.is_complex
            @inbounds norm1 = sum(hypot(dp[2i], dp[2i + 1]) for i in 1:cone.d)
        else
            @views norm1 = sum(abs, dp[2:end])
        end
        return (u - norm1 > eps(T))
    end
    return false
end

remul(a::T, b::T) where {T <: Real} = a * b

remul(a::Complex{T}, b::Complex{T}) where {T <: Real} =
    (real(a) * real(b) + imag(a) * imag(b))

function update_grad(cone::EpiNormInf{T}) where T
    @assert cone.is_feas
    u = cone.point[1]
    w = cone.w
    mu = cone.mu
    zeta = cone.zeta
    g = cone.grad

    @. mu = w / u
    @. zeta = T(0.5) * (u - remul(mu, w))
    cone.cu = (cone.d - 1) / u

    g[1] = cone.cu - sum(inv, zeta)

    if cone.is_complex
        @. cone.w1 = mu / zeta
        @views vec_copyto!(g[2:end], cone.w1)
    else
        @. g[2:end] = mu / zeta
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiNormInf)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d = cone.d
    u = cone.point[1]
    zeta = cone.zeta
    tzi2 = cone.s1
    g = cone.grad
    Hnz = cone.hess.data.nzval # modify nonzeros of upper triangle

    ui = inv(u)
    @. tzi2 = (inv(zeta) - ui) / zeta
    Hnz[1] = sum(tzi2) - cone.cu / u

    k = 2
    if cone.is_complex
        @inbounds for i in 1:d
            z_i = zeta[i]
            (mur, mui) = reim(cone.mu[i])
            mzir = g[2i]
            mzii = g[2i + 1]
            Hnz[k] = -mzir / z_i
            Hnz[k + 1] = (mur * mzir + ui) / z_i
            Hnz[k + 2] = -mzii / z_i
            Hnz[k + 3] = mzir * mzii
            Hnz[k + 4] = (mui * mzii + ui) / z_i
            k += 5
        end
    else
        @inbounds for i in 1:d
            Hnz[k] = -g[1 + i] / zeta[i]
            Hnz[k + 1] = tzi2[i]
            k += 2
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormInf{T, T},
    ) where T
    @assert cone.grad_updated
    u = cone.point[1]
    mu = cone.mu
    zeta = cone.zeta
    s1 = cone.s1

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]

        pui = p / u
        @. s1 = (p - mu * r) / zeta

        prod[1, j] = sum((s1[i] - pui) / zeta[i] for i in 1:cone.d) -
            cone.cu * pui

        @. prod[2:end, j] = (r / u - s1 * mu) / zeta
    end

    return prod
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormInf{T, Complex{T}},
    ) where T
    @assert cone.grad_updated
    u = cone.point[1]
    mu = cone.mu
    zeta = cone.zeta
    r = cone.w1
    w2 = cone.w2
    s1 = cone.s1

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        pui = p / u
        @. s1 = (p - remul(mu, r)) / zeta

        prod[1, j] = sum((s1[i] - pui) / zeta[i] for i in 1:cone.d) -
            cone.cu * pui

        @. w2 = (r / u - s1 * mu) / zeta
        @views vec_copyto!(prod[2:end, j], w2)
    end

    return prod
end

function update_inv_hess_aux(cone::EpiNormInf)
    @assert !cone.inv_hess_aux_updated
    @assert cone.grad_updated
    u = cone.point[1]
    w = cone.w
    wumzi = cone.wumzi
    umz = cone.s1

    @. umz = u - cone.zeta
    cone.Zu = -cone.cu + sum(inv, umz)

    @. wumzi = w / umz

    if !cone.is_complex
        @. cone.zti = u - wumzi * w
    end

    cone.inv_hess_aux_updated = true
end

function update_inv_hess(cone::EpiNormInf)
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    d = cone.d
    u = cone.point[1]
    w = cone.w
    zeta = cone.zeta
    wumzi = cone.wumzi
    Hi = cone.inv_hess.data

    hiuu = Hi[1, 1] = u / cone.Zu

    @views Hiuw = Hi[1, 2:end]
    @views Hiuw2 = Hi[2:end, 1]
    vec_copyto!(Hiuw2, wumzi)
    @. Hiuw = Hiuw2 * hiuu
    @views mul!(Hi[2:end, 2:end], Hiuw2, Hiuw')

    if cone.is_complex
        @inbounds for i in 1:d
            z_i = zeta[i]
            (wr, wi) = reim(w[i])
            (wumzir, wumzii) = reim(wumzi[i])
            k = 2 * i
            k1 = k + 1
            Hi[k, k] += z_i * (u - wumzir * wr)
            Hi[k1, k1] += z_i * (u - wumzii * wi)
            Hi[k, k1] -= z_i * wumzir * wi
        end
    else
        @inbounds for i in 1:d
            k = 1 + i
            Hi[k, k] += zeta[i] * cone.zti[i]
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormInf{T, T},
    ) where T
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    u = cone.point[1]
    wumzi = cone.wumzi

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]

        c1 = u * (p + dot(wumzi, r)) / cone.Zu
        prod[1, j] = c1

        @. prod[2:end, j] = c1 * wumzi + cone.zeta * r * cone.zti
    end

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormInf{T, Complex{T}},
    ) where T
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    u = cone.point[1]
    w = cone.w
    wumzi = cone.wumzi
    s1 = cone.s1
    r = cone.w1
    w2 = cone.w2

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        @. s1 = remul(wumzi, r)

        c1 = u * (p + sum(s1)) / cone.Zu
        prod[1, j] = c1

        @. w2 = c1 * wumzi + cone.zeta * (u * r - s1 * w)
        @views vec_copyto!(prod[2:end, j], w2)
    end

    return prod
end

function dder3(cone::EpiNormInf{T, T}, dir::AbstractVector{T}) where T
    @assert cone.grad_updated
    u = cone.point[1]
    mu = cone.mu
    zeta = cone.zeta
    rui = cone.w1
    s1 = cone.s1
    s2 = cone.s2
    dder3 = cone.dder3

    p = dir[1]
    @views r = dir[2:end]

    pui = p / u
    @. rui = r / u
    @. s1 = (p - mu * r) / zeta
    @. s2 = 0.5 * (p * pui - rui * r) / zeta - abs2(s1)

    @inbounds c1 = sum((s1[i] * pui + s2[i]) / zeta[i] for i in 1:cone.d)
    dder3[1] = -c1 - cone.cu * abs2(pui)

    @. dder3[2:end] = (s1 * rui + s2 * mu) / zeta

    return dder3
end

function dder3(cone::EpiNormInf{T, Complex{T}}, dir::AbstractVector{T}) where T
    @assert cone.grad_updated
    u = cone.point[1]
    mu = cone.mu
    zeta = cone.zeta
    rui = cone.w1
    r = w2 = cone.w2
    s1 = cone.s1
    s2 = cone.s2
    dder3 = cone.dder3

    p = dir[1]
    @views vec_copyto!(r, dir[2:end])

    pui = p / u
    @. rui = r / u
    @. s1 = (p - remul(mu, r)) / zeta
    @. s2 = 0.5 * (p * pui - remul(rui, r)) / zeta - abs2(s1)

    @inbounds c1 = sum((s1[i] * pui + s2[i]) / zeta[i] for i in 1:cone.d)
    dder3[1] = -c1 - cone.cu * abs2(pui)

    @. w2 = (s1 * rui + s2 * mu) / zeta
    @views vec_copyto!(dder3[2:end], w2)

    return dder3
end

function alloc_hess!(cone::EpiNormInf{T, T}) where {T <: Real}
    # initialize sparse idxs for upper triangle of Hessian
    dim = cone.dim
    nnz_tri = 2 * dim - 1
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    @views I[idxs1] .= 1
    @views J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    @views I[idxs2] .= 2:dim
    @views J[idxs2] .= 2:dim
    V = ones(T, nnz_tri)
    cone.hess = Symmetric(sparse(I, J, V, dim, dim), :U)
    return
end

function alloc_hess!(cone::EpiNormInf{T, Complex{T}}) where {T <: Real}
    # initialize sparse idxs for upper triangle of Hessian
    dim = cone.dim
    nnz_tri = 2 * dim - 1 + cone.d
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    @views I[idxs1] .= 1
    @views J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    @views I[idxs2] .= 2:dim
    @views J[idxs2] .= 2:dim
    idxs3 = (2 * dim):nnz_tri
    @views I[idxs3] .= 2:2:dim
    @views J[idxs3] .= 3:2:dim
    V = ones(T, nnz_tri)
    cone.hess = Symmetric(sparse(I, J, V, dim, dim), :U)
    return
end

# TODO remove this in favor of new hess_nz_count etc functions
# that directly use uu, uw, ww etc
hess_nz_count(cone::EpiNormInf{<:Real, <:Real}) =
    3 * cone.dim - 2

hess_nz_count(cone::EpiNormInf{<:Real, <:Complex}) =
    3 * cone.dim - 2 + 2 * cone.d

hess_nz_count_tril(cone::EpiNormInf{<:Real, <:Real}) =
    2 * cone.dim - 1

hess_nz_count_tril(cone::EpiNormInf{<:Real, <:Complex}) =
    2 * cone.dim - 1 + cone.d

hess_nz_idxs_col(cone::EpiNormInf{<:Real, <:Real}, j::Int) =
    (j == 1 ? (1:cone.dim) : [1, j])

hess_nz_idxs_col(cone::EpiNormInf{<:Real, <:Complex}, j::Int) =
    (j == 1 ? (1:cone.dim) : (iseven(j) ? [1, j, j + 1] : [1, j - 1, j]))

hess_nz_idxs_col_tril(cone::EpiNormInf{<:Real, <:Real}, j::Int) =
    (j == 1 ? (1:cone.dim) : [j])

hess_nz_idxs_col_tril(cone::EpiNormInf{<:Real, <:Complex}, j::Int) =
    (j == 1 ? (1:cone.dim) : (iseven(j) ? [j, j + 1] : [j]))
