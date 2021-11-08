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




function update_hess(cone::EpiNormInf{T}) where T
    isdefined(cone, :hess) || alloc_hess!(cone)
    d = cone.d

    # modify nonzeros of upper triangle of Hessian
    nzval = cone.hess.data.nzval
    # nzval[1] = cone.Huu

    if !cone.is_complex
        nz_idx = 2
        @inbounds for i in 1:d
            # nzval[nz_idx] = cone.Hure[i]
            # nzval[nz_idx + 1] = cone.Hrere[i]
            nz_idx += 2
        end
    else
        nz_idx = 1
        @inbounds for i in 1:d
            # @. nzval[nz_idx .+ (1:5)] = (cone.Hure[i], cone.Hrere[i],
                # cone.Huim[i], cone.Hreim[i], cone.Himim[i])
            nz_idx += 5
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiNormInf)
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    Hi = cone.inv_hess.data

    # wden = cone.wden
    # u = cone.point[1]
    # schur = cone.schur
    #
    # Hi[1, 1] = inv(schur)
    # @inbounds for j in 1:d
    #     if cone.is_complex
    #         Hi[2j, 1] = cone.Hiure[j]
    #         Hi[2j + 1, 1] = cone.Hiuim[j]
    #     else
    #         Hi[j + 1, 1] = cone.Hiure[j]
    #     end
    # end
    # @. Hi[1, 2:end] = Hi[2:end, 1] / schur
    #
    # @inbounds for j in 2:cone.dim, i in 2:j
    #     Hi[i, j] = Hi[j, 1] * Hi[1, i]
    # end
    #
    # if cone.is_complex
    #     @inbounds for j in 1:d
    #         detj = cone.idet[j]
    #         vj = 2j
    #         wj = vj + 1
    #         Hi[vj, vj] += cone.Himim[j] / detj
    #         Hi[wj, wj] += cone.Hrere[j] / detj
    #         Hi[vj, wj] -= cone.Hreim[j] / detj
    #     end
    # else
    #     @inbounds for (j, rerej) in enumerate(cone.Hrere)
    #         vj = j + 1
    #         Hi[vj, vj] += inv(rerej)
    #     end
    # end

    cone.inv_hess_updated = true
    return cone.inv_hess
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
