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
    zh::Vector{T}
    cu::T
    u2ti::Vector{T}
    zti1::T

    w1::Vector{R}
    w2::Vector{R}
    s1::Vector{T}
    s2::Vector{T}

    function EpiNormInf{T, R}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.is_complex = (R <: Complex)
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
    cone.zh = zeros(T, d)
    cone.u2ti = zeros(T, d)
    cone.w1 = zeros(R, d)
    cone.w2 = zeros(R, d)
    cone.s1 = zeros(T, d)
    cone.s2 = zeros(T, d)
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
    u = cone.dual_point[1]
    if u > eps(T)
        @views svec_to_smat!(cone.w1, cone.dual_point[2:end])
        return (u - sum(abs, cone.w1) > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormInf{T}) where T
    @assert cone.is_feas
    u = cone.point[1]
    w = cone.w
    zh = cone.zh
    w1 = cone.w1
    g = cone.grad

    cone.cu = (cone.d - 1) / u
    u2 = abs2(u)
    @. zh = T(0.5) * (u2 - abs2(w))

    g[1] = cone.cu - sum(u / zh[i] for i in 1:cone.d)
    @. w1 = w / zh
    @views vec_copyto!(g[2:end], w1)

    cone.grad_updated = true
    return cone.grad
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormInf,
    )
    @assert cone.grad_updated
    d = cone.d
    u = cone.point[1]
    w = cone.w
    zh = cone.zh
    cu = cone.cu
    r = cone.w1
    w2 = cone.w2
    s1 = cone.s1

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        @. s1 = (real(conj(w) * r) - u * p) / zh

        prod[1, j] = -sum((p + u * s1[i]) / zh[i] for i in 1:d) - cu * p / u

        @. w2 = (r + s1 * w) / zh
        @views vec_copyto!(prod[2:end, j], w2)
    end

    return prod
end

function update_inv_hess_aux(cone::EpiNormInf)
    @assert !cone.inv_hess_aux_updated
    @assert cone.grad_updated
    zh = cone.zh
    th = cone.s1

    u2 = abs2(cone.point[1])
    @. th = u2 - zh
    @. cone.u2ti = u2 / th
    @inbounds cone.zti1 = 1 + sum(zh[i] / th[i] for i in 1:cone.d)

    cone.inv_hess_aux_updated = true
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormInf,
    )
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    u = cone.point[1]
    w = cone.w
    zh = cone.zh
    zti1 = cone.zti1
    u2ti = cone.u2ti
    r = cone.w1
    w2 = cone.w2
    s1 = cone.s1

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        @. s1 = real(conj(w) * r)

        c1 = (u * p + dot(u2ti, s1)) / zti1
        prod[1, j] = c1 * u

        @. w2 = zh * r + ((c1 - s1) * u2ti + s1) * w
        @views vec_copyto!(prod[2:end, j], w2)
    end

    return prod
end

function dder3(cone::EpiNormInf{T}, dir::AbstractVector{T}) where T
    @assert cone.grad_updated
    u = cone.point[1]
    w = cone.w
    zh = cone.zh
    dder3 = cone.dder3
    r = cone.w1
    s1 = cone.s1
    s2 = cone.s2

    p = dir[1]
    @views vec_copyto!(r, dir[2:end])

    @. s1 = (u * p - real(conj(w) * r)) / zh
    @. s2 = T(0.5) * (abs2(p) - abs2(r)) / zh - abs2(s1)

    @inbounds c1 = sum((p * s1[i] + u * s2[i]) / zh[i] for i in 1:cone.d)
    dder3[1] = -c1 - cone.cu * abs2(p / u)

    @. cone.w2 = (s1 * r + s2 * w) / zh
    @views vec_copyto!(dder3[2:end], cone.w2)

    return dder3
end
