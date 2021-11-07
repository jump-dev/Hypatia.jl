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
    hess_aux_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, SparseMatrixCSC{T, Int}}
    inv_hess::Symmetric{T, Matrix{T}}

    w::Vector{R}
    s::Vector{T}
    V::Vector{R}
    z::Vector{T}
    zi::Vector{T}
    uzi::Vector{T}
    szi::Vector{T}

    huu::T
    huw::Vector{T}
    hiuui::T
    hiuw::Vector{T}
    hiww #?

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
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated = false)

use_sqrt_hess_oracles(::Int, cone::EpiNormInf) = false

function setup_extra_data!(
    cone::EpiNormInf{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    d = cone.d
    cone.w = zeros(R, d)
    cone.s = zeros(T, d)
    cone.z = zeros(T, d)
    cone.zi = zeros(T, d)
    cone.uzi = zeros(T, d)
    cone.szi = zeros(T, d)
    cone.V = zeros(R, d)
    cone.huw = zeros(T, d)
    cone.hiuw = zeros(T, d)
    # cone.hiww = zeros(T, d, d)

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

function update_grad(cone::EpiNormInf)
    @assert cone.is_feas
    u = cone.point[1]
    w = cone.w
    z = cone.z
    uzi = cone.uzi
    w1 = cone.w1
    g = cone.grad

    u2 = abs2(u)
    @. z = u2 - abs2(w)
    @. uzi = 2 * u / z

    g[1] = -sum(uzi) + (cone.d - 1) / u
    @. w1 = 2 * w / z
    @views vec_copyto!(g[2:end], w1)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiNormInf)
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    u = cone.point[1]
    w = cone.w
    V = cone.V
    s = cone.s
    z = cone.z
    zi = cone.zi
    uzi = cone.uzi
    szi = cone.szi
    d = cone.d
    # hiww = cone.hiww

    @. s = abs(w)
    @. V = w / s
    @. zi = inv(z)
    @. szi = 2 * s / z
    u2 = abs2(u)
    cu = (cone.d - 1) / u2
    t = u2 .+ abs2.(s) # TODO
    cone.huu = 2 * sum(t[i] / z[i] * zi[i] for i in 1:d) - cu
    @. cone.huw = -uzi * szi
    cone.hiuui = 2 * sum(inv, t) - cu
    @. cone.hiuw = 2 * u * s ./ t

    # for j in 1:d
    #     s_j = s[j]
    #     z_j = z[j]
    #     for i in 1:(j - 1)
    #         dij = u2 + s[i] * s_j
    #         hiww[i, j] = z[i] / (2 * dij) * z_j
    #     end
    #     hiww[j, j] = z_j / (2 * t[j]) * z_j
    # end

    cone.hess_aux_updated = true
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormInf,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    V = cone.V
    u = cone.point[1]
    w = cone.w
    s = cone.s
    z = cone.z
    # TODO delete not using
    zi = cone.zi
    uzi = cone.uzi
    szi = cone.szi
    huw = cone.huw
    huu = cone.huu
    r = cone.w1

    simszi = zero(s)
    HW = zero(w)
    Vszi = 2 * conj.(w) ./ z

# TODO don't use huu (for all spectral cones)

    # TODO try to do not column by column, like old code
    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])
        @. simszi = real(Vszi * r)

        prod[1, j] = huu * p - dot(uzi, simszi)

        @. HW = 2 * (r + (simszi - p * uzi) * w) / z
        @views vec_copyto!(prod[2:end, j], HW)
    end

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormInf,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    V = cone.V
    u = cone.point[1]
    w = cone.w
    s = cone.s
    z = cone.z
    # TODO delete not using
    zi = cone.zi
    uzi = cone.uzi
    szi = cone.szi
    hiuui = cone.hiuui
    r = cone.w1

    u2 = abs2(u)
    t = u2 .+ abs2.(s)

    simsz = zero(s)
    HiW = zero(w)
    wti = w ./ t
    T = eltype(z)

    # TODO try to do not column by column, like old code
    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])
        @. simsz = real(conj(wti) * r)

        c1 = (p + 2 * u * sum(simsz)) / hiuui
        prod[1, j] = c1

        dconst = 2 * u * c1
        @. HiW = dconst * wti + z * (T(0.5) * r - simsz * w)
        @views vec_copyto!(prod[2:end, j], HiW)
    end

    return prod
end

function dder3(cone::EpiNormInf, dir::AbstractVector)
    cone.hess_aux_updated || update_hess_aux(cone)
    u = cone.point[1]
    w = cone.w
    d = cone.d
    z = cone.z
    dder3 = cone.dder3
    r = cone.w1
    s1 = cone.s1
    s2 = cone.s2

    p = dir[1]
    @views vec_copyto!(r, dir[2:end])

    @. s1 = -2 * (u * p - real(conj(w) * r)) / z
    @. s2 = (abs2(r) - abs2(p)) / z + abs2(s1)

    dder3[1] = 2 * sum((p * s1[i] + u * s2[i]) / z[i] for i in 1:d) -
        (cone.d - 1) / u * abs2(p / u)

    @. cone.w2 = -2 * (s1 * r + s2 * w) / z
    @views vec_copyto!(dder3[2:end], cone.w2)

    return dder3
end
