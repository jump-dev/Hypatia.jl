#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

TODO
- write description
- assumes first A matrix is PSD (eg identity)
=#

mutable struct LinMatrixIneq{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    side::Int
    As::Vector
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    sumA
    fact
    sumAinvAs::Vector

    function LinMatrixIneq{T}(
        As::Vector;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        dim = length(As)
        @assert dim > 1
        side = 0
        for A_i in As
            if A_i isa AbstractMatrix
                if iszero(side)
                    side = size(A_i, 1)
                else
                    @assert size(A_i, 1) == side
                end
            end
            # @assert eltype(A_i) <: RealOrComplex{T}
            @assert ishermitian(A_i)
        end
        @assert side > 0
        @assert div(side * (side + 1), 2) >= dim # TODO necessary to ensure linear independence of As (but not sufficient)
        @assert isposdef(first(As))
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.side = side
        cone.As = As
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

reset_data(cone::LinMatrixIneq) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::LinMatrixIneq{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    return
end

get_nu(cone::LinMatrixIneq) = cone.side

use_correction(cone::LinMatrixIneq) = true

function set_initial_point(arr::AbstractVector, cone::LinMatrixIneq{T}) where {T <: Real}
    arr .= 0
    arr[1] = 1
    return arr
end

lmi_fact(arr::Union{UniformScaling{R}, Diagonal{R}}) where {R} = arr # NOTE could use SymTridiagonal here when that type gets a isposdef and ldiv in Julia
lmi_fact(arr::AbstractSparseMatrix{R}) where {R} = cholesky(Hermitian(arr), shift=false, check=false)
lmi_fact(arr::AbstractMatrix{R}) where {R} = cholesky!(Hermitian(arr), check=false)

function update_feas(cone::LinMatrixIneq{T}) where {T <: Real}
    @assert !cone.feas_updated

    # NOTE not in-place because typeof(A) is AbstractMatrix eg sparse
    # TODO if sumA is dense, can do in-place
    cone.sumA = sum(w_i * A_i for (w_i, A_i) in zip(cone.point, cone.As))
    @assert ishermitian(cone.sumA) # TODO delete
    @assert eltype(cone.sumA) <: RealOrComplex{T}
    cone.fact = lmi_fact(cone.sumA)
    cone.is_feas = isposdef(cone.fact)

    cone.feas_updated = true
    return cone.is_feas
end

update_dual_feas(cone::LinMatrixIneq) = true # TODO use a dikin ellipsoid condition?

function update_grad(cone::LinMatrixIneq{T}) where {T <: Real}
    @assert cone.is_feas

    # grad[i] = -tr(inv(sumA) * A[i])
    cone.sumAinvAs = [cone.fact \ A_i for A_i in cone.As] # TODO if dense, can do in-place
    @inbounds for (i, sumAinvAs_i) in enumerate(cone.sumAinvAs)
        cone.grad[i] = -sum(real(sumAinvAs_i[k, k]) for k in 1:cone.side)
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::LinMatrixIneq)
    @assert cone.is_feas
    sumAinvAs = cone.sumAinvAs
    H = cone.hess.data

    # H[i, j] = tr((cone.fact \ A_i) * (cone.fact \ A_j))
    @inbounds for i in 1:cone.dim, j in i:cone.dim
        H[i, j] = real(dot(sumAinvAs[i], sumAinvAs[j]'))
    end

    cone.hess_updated = true
    return cone.hess
end

function correction2(cone::LinMatrixIneq, primal_dir::AbstractVector)
    sumAinvAs = cone.sumAinvAs
    corr = cone.correction
    dim = cone.dim

    tmp = similar(sumAinvAs[1])
    tmp .= 0
    @inbounds for j in 1:dim, k in 1:dim
        mul!(tmp, sumAinvAs[j], sumAinvAs[k], primal_dir[j] * primal_dir[k], true)
    end
    @inbounds for i in 1:dim
        corr[i] = real(dot(sumAinvAs[i], tmp'))
    end

    return corr
end
