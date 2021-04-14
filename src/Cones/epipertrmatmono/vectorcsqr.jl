#=
TODO

vector cone of squares, i.e. ℝ₊ᵈ for d ≥ 1, with rank d
=#

struct VectorCSqr{T <: Real} <: ConeOfSquares{T} end

vector_dim(::Type{<:VectorCSqr}, d::Int) = d

mutable struct VectorCSqrCache{T <: Real} <: CSqrCache{T}
    viw::Vector{T}
    wi::Vector{T}
    ϕ::T
    ζ::T
    ζi::T
    ∇h_viw::Vector{T}
    ∇2h_viw::Vector{T}
    ζi∇h_viw::Vector{T}
    σ::T
    VectorCSqrCache{T}() where {T <: Real} = new{T}()
end

function setup_csqr_cache(cone::EpiPerTrMatMono{VectorCSqr{T}}) where T
    cone.cache = cache = VectorCSqrCache{T}()
    cache.viw = zeros(T, cone.d)
    cache.wi = zeros(T, cone.d)
    cache.∇h_viw = zeros(T, cone.d)
    cache.∇2h_viw = zeros(T, cone.d)
    cache.ζi∇h_viw = zeros(T, cone.d)
    return
end

function set_initial_point(arr::AbstractVector, cone::EpiPerTrMatMono{<:VectorCSqr, F}) where F
    (arr[1], arr[2], w0) = get_initial_point(F, cone.d)
    @views fill!(arr[3:end], w0)
    return arr
end

function update_feas(cone::EpiPerTrMatMono{VectorCSqr{T}, F, T}) where {T, F}
    @assert !cone.feas_updated
    cache = cone.cache
    v = cone.point[2]

    if (v > eps(T)) && all(>(eps(T)), cone.w_view)
        @. cache.viw = cone.w_view / v
        cache.ϕ = h_sum(F, cache.viw)
        cache.ζ = cone.point[1] - v * cache.ϕ
        cone.is_feas = (cache.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

# TODO implement
# is_dual_feas(cone::EpiPerTrMatMono{VectorCSqr}) =

function update_grad(cone::EpiPerTrMatMono{<:VectorCSqr, F}) where F
    @assert !cone.grad_updated && cone.is_feas
    grad = cone.grad
    v = cone.point[2]
    cache = cone.cache
    ζi = cache.ζi = inv(cache.ζ)
    ∇h_viw = cache.∇h_viw
    @. ∇h_viw = h_der1(F, cache.viw)
    cache.σ = cache.ϕ - dot(cache.viw, ∇h_viw)
    @. cache.wi = inv(cone.w_view)
    @. cache.ζi∇h_viw = ζi * ∇h_viw

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @. grad[3:end] = -cache.wi + cache.ζi∇h_viw

    cone.grad_updated = true
    return grad
end

function update_hess(cone::EpiPerTrMatMono{<:VectorCSqr, F}) where F
    @assert cone.grad_updated && !cone.hess_updated
    d = cone.d
    v = cone.point[2]
    cache = cone.cache
    H = cone.hess.data
    ζi = cache.ζi
    ζi2 = abs2(ζi)
    σ = cache.σ
    viw = cache.viw
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    @. ∇2h_viw = h_der2(F, cache.viw)
    ζi∇h_viw = cache.ζi∇h_viw
    wi = cache.wi
    ζivi = ζi / v
    ζiσ = ζi * σ

    # Huu
    H[1, 1] = ζi2

    # Huv
    H[1, 2] = -ζi2 * σ

    # Hvv start
    Hvv = v^-2 + abs2(ζi * σ)

    @inbounds for j in 1:d
        ζi∇h_viw_j = ζi∇h_viw[j]
        term_j = ζivi * viw[j] * ∇2h_viw[j]
        Hvv += viw[j] * term_j
        j2 = 2 + j

        # Huw
        H[1, j2] = -ζi * ζi∇h_viw_j

        # Hvw
        H[2, j2] = ζiσ * ζi∇h_viw_j - term_j

        # Hww
        for i in 1:(j - 1)
            H[2 + i, j2] = ζi∇h_viw_j * ζi∇h_viw[i]
        end
        H[j2, j2] = abs2(ζi∇h_viw_j) + ζivi * ∇2h_viw[j] + abs2(wi[j])
    end

    # Hvv end
    H[2, 2] = Hvv

    cone.hess_updated = true
    return cone.hess
end

# function update_inv_hess(cone::EpiPerTrMatMono{<:VectorCSqr, F}) where F
#
# end

# function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerTrMatMono{<:VectorCSqr, F}) where F
#     # cone.hess_aux_updated || update_hess_aux(cone) # TODO
#     v = cone.point[2]
#     cache = cone.cache
#
#     # TODO @inbounds
#     for j in 1:size(arr, 2)
#
#
#     end
#
#     return prod
# end

# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerTrMatMono{<:VectorCSqr, F}) where F
#     # cone.hess_aux_updated || update_hess_aux(cone) # TODO
#     v = cone.point[2]
#     cache = cone.cache
#
#     # TODO @inbounds
#     for j in 1:size(arr, 2)
#
#
#     end
#
#     return prod
# end

# function correction(cone::EpiPerTrMatMono{<:VectorCSqr, F}, primal_dir::AbstractVector) where F
#     @assert cone.hess_updated
#     corr = cone.correction
#     cache = cone.cache
#
#
#
#     return corr
# end
