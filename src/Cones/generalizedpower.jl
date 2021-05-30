"""
$(TYPEDEF)

Generalized power cone parametrized by powers `α` in the unit simplex and
dimension `d` of the normed variables.

    $(FUNCTIONNAME){T}(α::Vector{T}, d::Int, use_dual::Bool = false)
"""
mutable struct GeneralizedPower{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    α::Vector{T}
    n::Int

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
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    produ::T
    produw::T
    produuw::T
    aui::Vector{T}
    auiproduuw::Vector{T}
    tempm::Vector{T}

    function GeneralizedPower{T}(
        α::Vector{T},
        n::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert n >= 1
        dim = length(α) + n
        @assert dim >= 3
        @assert all(ai > 0 for ai in α)
        @assert sum(α) ≈ 1
        cone = new{T}()
        cone.n = n
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.α = α
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

dimension(cone::GeneralizedPower) = length(cone.α) + cone.n

function setup_extra_data!(cone::GeneralizedPower{T}) where {T <: Real}
    m = length(cone.α)
    cone.aui = zeros(T, m)
    cone.auiproduuw = zeros(T, m)
    cone.tempm = zeros(T, m)
    return cone
end

get_nu(cone::GeneralizedPower) = length(cone.α) + 1

function set_initial_point!(arr::AbstractVector, cone::GeneralizedPower)
    m = length(cone.α)
    @. @views arr[1:m] = sqrt(1 + cone.α)
    @views arr[(m + 1):cone.dim] .= 0
    return arr
end

function update_feas(cone::GeneralizedPower{T}) where {T <: Real}
    @assert !cone.feas_updated
    m = length(cone.α)
    @views u = cone.point[1:m]

    if all(>(eps(T)), u)
        @inbounds cone.produ = exp(2 * sum(cone.α[i] * log(u[i])
            for i in eachindex(cone.α)))
        @views cone.produw = cone.produ - sum(abs2, cone.point[(m + 1):end])
        cone.is_feas = (cone.produw > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::GeneralizedPower{T}) where {T <: Real}
    α = cone.α
    m = length(cone.α)
    @views u = cone.dual_point[1:m]

    if all(>(eps(T)), u)
        @inbounds p = exp(2 * sum(α[i] * log(u[i] / α[i])
            for i in eachindex(α)))
        @views w = cone.dual_point[(m + 1):end]
        return (p - sum(abs2, w) > eps(T))
    end

    return false
end

function update_grad(cone::GeneralizedPower)
    @assert cone.is_feas
    m = length(cone.α)
    @views u = cone.point[1:m]
    @views w = cone.point[(m + 1):end]

    @. cone.aui = 2 * cone.α / u
    cone.produuw = cone.produ / cone.produw
    @. cone.auiproduuw = -cone.aui * cone.produuw
    @. @views cone.grad[1:m] = cone.auiproduuw - (1 - cone.α) / u
    produwi2 = 2 / cone.produw
    @. @views cone.grad[(m + 1):end] = produwi2 * w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::GeneralizedPower)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    m = length(cone.α)
    @views u = cone.point[1:m]
    @views w = cone.point[(m + 1):end]
    aui = cone.aui
    auiproduuw = cone.auiproduuw
    g = cone.grad

    produuwm1 = 1 - cone.produuw
    @inbounds for j in 1:m
        auiproduuwm1 = auiproduuw[j] * produuwm1
        @inbounds for i in 1:j
            H[i, j] = aui[i] * auiproduuwm1
        end
        H[j, j] -= g[j] / u[j]
    end

    offset = 2 / cone.produw
    @inbounds for j in m .+ (1:cone.n)
        gj = g[j]
        @inbounds for i in 1:m
            H[i, j] = auiproduuw[i] * gj
        end
        @inbounds for i in (m + 1):j
            H[i, j] = g[i] * gj
        end
        H[j, j] += offset
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::GeneralizedPower,
    )
    @assert cone.grad_updated
    m = length(cone.α)
    @views u = cone.point[1:m]
    @views w = cone.point[(m + 1):end]
    aui = cone.aui
    produuw = cone.produuw
    w_idxs = (m + 1):cone.dim
    produwi2 = 2 / cone.produw
    const1 = 2 * produuw - 1
    tempm = cone.tempm
    @. tempm = (1 + const1 * cone.α) / u / u

    @inbounds @views for i in 1:size(arr, 2)
        arr_u = arr[1:m, i]
        arr_w = arr[w_idxs, i]
        dot1 = -produuw * dot(aui, arr_u)
        dot2 = dot1 + produwi2 * dot(w, arr_w)
        dot3 = dot1 - produuw * dot2
        dot4 = produwi2 * dot2
        @. prod[1:m, i] = dot3 * aui + tempm * arr_u
        @. prod[w_idxs, i] = dot4 * w + produwi2 * arr_w
    end

    return prod
end

function dder3(cone::GeneralizedPower, dir::AbstractVector)
    m = length(cone.α)
    @views u = cone.point[1:m]
    @views w = cone.point[(m + 1):end]
    dder3 = cone.dder3
    @views u_dder3 = dder3[1:m]
    @views w_dder3 = dder3[(m + 1):end]
    @views u_dir = dir[1:m]
    @views w_dir = dir[(m + 1):end]
    α = cone.α
    produw = cone.produw
    produuw = cone.produuw

    wwd = 2 * dot(w, w_dir)
    udu = cone.tempm
    @. udu = u_dir / u
    audu = dot(α, udu)
    const8 = 2 * produuw - 1
    const1 = 2 * const8 * abs2(audu) + sum(ai * udui * udui
        for (ai, udui) in zip(α, udu))
    const15 = wwd / produw
    const10 = sum(abs2, w_dir) + wwd * const15

    const11 = -2 * produuw * (1 - produuw)
    const12 = -2 * produuw / produw
    const13 = const11 * const1 + const12 * (2 * wwd * const8 * audu - const10)
    const14 = const11 * 2 * audu + const12 * wwd
    @. u_dder3 = const14 .+ const8 * udu
    u_dder3 .*= α
    u_dder3 .+= udu
    u_dder3 .*= udu
    @. u_dder3 += const13 * α
    u_dder3 ./= u

    const2 = -2 * const12 * audu
    const6 = 2 * const2 * const15 + const12 * const1 -
        2 / produw * const10 / produw
    const7 = const2 - 2 / produw * wwd / produw
    @. w_dder3 = const7 * w_dir + const6 * w

    return cone.dder3
end
