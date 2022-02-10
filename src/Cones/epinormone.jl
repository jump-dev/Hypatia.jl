"""
$(TYPEDEF)

Epigraph of real or complex infinity norm cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int, use_dual::Bool = false)
"""
mutable struct EpiNormOne{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dual_grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    dual_grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_aux_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, SparseMatrixCSC{T, Int}}

    w::Vector{T}
    mu::Vector{T}
    grad_mu::Vector{T}
    zeta::T
    dual_zeta::Vector{T}
    grad_zeta::Vector{T}
    cu::T
    Zu::T
    wumzi::Vector{T}
    zti::Vector{T}

    s1::Vector{T}
    s2::Vector{T}
    w1::Vector{T}
    w2::Vector{T}

    function EpiNormOne{T}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.d = dim - 1
        return cone
    end
end


reset_data(cone::EpiNormOne) = (cone.feas_updated = cone.grad_updated =
    cone.dual_grad_updated = cone.hess_updated = cone.inv_hess_updated =
    cone.hess_aux_updated = false)

function setup_extra_data!(
    cone::EpiNormOne{T},
    ) where {T <: Real}
    d = cone.d
    cone.w = zeros(T, d)
    cone.mu = zeros(T, d)
    cone.grad_mu = zeros(T, d)
    cone.dual_zeta = zeros(T, d)
    cone.grad_zeta = zeros(T, d)
    cone.wumzi = zeros(T, d)
    cone.s1 = zeros(T, d)
    cone.s2 = zeros(T, d)
    cone.w1 = zeros(T, d)
    cone.zti = zeros(T, d)
    return cone
end

get_nu(cone::EpiNormOne) = 1 + cone.d

function set_initial_point!(
    arr::AbstractVector{T},
    cone::EpiNormOne{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormOne{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views vec_copyto!(cone.w, cone.point[2:end])
        @views norm1 = sum(abs, cone.w)
        cone.zeta = u - norm1
        cone.is_feas = (cone.zeta > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiNormOne{T}) where T
    dp = cone.dual_point
    u = dp[1]
    if u > eps(T)
        @views return (u - maximum(abs, dp[2:end]) > eps(T))
    end
    return false
end

function update_grad(
    cone::EpiNormOne{T},
    ) where {T <: Real}
    u = cone.point[1]
    w = cone.w

    (gu, zw2) = epinorminf_dg(u, w, cone.d, cone.zeta)
    cone.grad[1] = gu
    @views vec_copyto!(cone.grad[2:end], zw2)

    # TODO moves somewhere else
    g = cone.grad
    @views @. cone.grad_mu = g[2:end] / gu
    @views @. cone.grad_zeta = -T(0.5) * (gu - remul(cone.grad_mu, g[2:end]))
    cone.cu = (cone.d - 1) / -gu

    cone.grad_updated = true
    return cone.grad
end

function update_inv_hess(cone::EpiNormOne)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    @assert cone.grad_updated
    d = cone.d
    u = -cone.grad[1]
    zeta = cone.grad_zeta
    tzi2 = cone.s1
    g = -cone.point
    Hnz = cone.inv_hess.data.nzval # modify nonzeros of upper triangle

    ui = inv(u)
    @. tzi2 = (inv(zeta) - ui) / zeta
    Hnz[1] = sum(tzi2) - cone.cu / u

    k = 2
    @inbounds for i in 1:d
        Hnz[k] = -g[1 + i] / zeta[i]
        Hnz[k + 1] = tzi2[i]
        k += 2
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormOne,
    )
    @assert cone.grad_updated
    u = -cone.grad[1]
    mu = cone.grad_mu
    zeta = cone.grad_zeta
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

function update_hess_aux(cone::EpiNormOne)
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    zeta = cone.grad_zeta
    u = -cone.grad[1]
    w = -cone.grad[2:end]
    wumzi = cone.wumzi
    umz = cone.s1

    @. umz = u - zeta
    cone.Zu = -cone.cu + sum(inv, umz)

    @. wumzi = w / umz

    @. cone.zti = u - wumzi * w

    cone.hess_aux_updated = true
end

function update_hess(cone::EpiNormOne)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d = cone.d
    u = -cone.grad[1]
    w = -cone.grad[2:end]
    zeta = cone.grad_zeta
    wumzi = cone.wumzi
    H = cone.hess.data

    huu = H[1, 1] = u / cone.Zu

    @views Huw = H[1, 2:end]
    @views Huw2 = H[2:end, 1]
    vec_copyto!(Huw2, wumzi)
    @. Huw = Huw2 * huu
    @views mul!(H[2:end, 2:end], Huw2, Huw')

    @inbounds for i in 1:d
        k = 1 + i
        H[k, k] += zeta[i] * cone.zti[i]
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormOne{T},
    ) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    u = -cone.grad[1]
    wumzi = cone.wumzi

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]

        c1 = u * (p + dot(wumzi, r)) / cone.Zu
        prod[1, j] = c1

        @. prod[2:end, j] = c1 * wumzi + cone.grad_zeta * r * cone.zti
    end

    return prod
end

function dder3(
    cone::EpiNormOne{T},
    dir::AbstractVector{T},
    ) where {T <: Real}
    cone.dder3 .= -dder3(cone, dir, hess(cone) * dir)
    return cone.dder3
end

function dder3(
    cone::EpiNormOne{T},
    pdir::AbstractVector{T},
    ddir::AbstractVector{T},
    ) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)
    dder3 = cone.dder3
    point = cone.point
    d1 = hess_prod!(zeros(T, cone.dim), pdir, cone)
    d2 = ddir

    u = -cone.grad[1]
    w = -cone.grad[2:end]
    mu = cone.grad_mu
    zeta = cone.grad_zeta

    p = d1[1]
    x = d2[1]
    @views r = d1[2:end]
    @views z = d2[2:end]
    s1 = cone.s1
    @. s1 = (z * mu - x) * (p - r * mu) / zeta + (x * p - r * z) / 2u
    s2 = cone.s2
    @. s2 = -(p * z + x * r) / 2u

    dder3[1] = p * cone.cu / abs2(u) * x + sum((s1[i] + mu[i] * s2[i] +
        x * p / u) / abs2(zeta[i]) for i in 1:cone.d)
    @. dder3[2:end] = (s2 - mu * (s1 - r * z / u)) / abs2(zeta)

    dder3 .= hess_prod!(zeros(T, cone.dim), copy(dder3), cone)

    return dder3
end

function alloc_inv_hess!(cone::EpiNormOne{T}) where {T <: Real}
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
    cone.inv_hess = Symmetric(sparse(I, J, V, dim, dim), :U)
    return
end

function epinorminf_dg(u::T, w::AbstractVector{T}, d::Int, dual_zeta::T) where T
    h(y) = u * y + sum(sqrt(1 + abs2(w[i] * y)) for i in 1:d) + 1
    hp(y) = u + y * sum(abs2(w[i]) / sqrt(1 + abs2(w[i] * y)) for i in 1:d)
    lower = -(d + 1) / dual_zeta
    upper = -inv(dual_zeta)
    dgu = rootnewton(h, hp, lower = lower, upper = upper, init = lower)

    # z * w / 2
    zw2 = copy(w)
    for i in eachindex(w)
        if abs(w[i]) .< eps(T)
            zw2[i] = 0
        else
            zw2[i] = sqrt(1 + abs2(w[i] * dgu)) / w[i] - 1 / w[i]
        end
    end
    return (dgu, zw2)
end

function rootnewton(
    f::Function,
    g::Function;
    lower::T = -Inf,
    upper::T = Inf,
    init::T = (lower + upper) / 2,
    increasing::Bool = true,
    ) where {T <: Real}
    curr = init
    f_new = f(big(curr))
    iter = 0
    while abs(f_new) > 1000eps(T)
        candidate = curr - f_new / g(big(curr))
        if (candidate < lower) || (candidate > upper)
            curr = (lower + upper) / 2
        else
            curr = candidate
        end
        f_new = f(big(curr))
        if (f_new < 0 && increasing) || (f_new >= 0 && !increasing)
            lower = curr
        else
            upper = curr
        end
        iter += 1
        if iter > 200
            @warn "too many iters in dual grad"
            break
        end
    end
    return T(curr)
end
