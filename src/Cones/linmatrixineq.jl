#=
TODO
- write description
- assumes first A matrix is PSD (eg identity)
=#

mutable struct LinMatrixIneq{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    dim::Int
    side::Int
    As::Vector

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    correction::Vector{T}
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

    sumA::AbstractMatrix
    fact
    sumAinvAs::Vector

    function LinMatrixIneq{T}(
        As::Vector;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
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
            @assert ishermitian(A_i)
        end
        @assert side > 0
        @assert div(side * (side + 1), 2) >= dim # TODO necessary to ensure linear independence of As (but not sufficient)
        @assert isposdef(first(As))
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.dim = dim
        cone.side = side
        cone.As = As
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

# TODO only allocate the fields we use
function setup_extra_data(cone::LinMatrixIneq{T}) where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    return cone
end

get_nu(cone::LinMatrixIneq) = cone.side

function set_initial_point(arr::AbstractVector, cone::LinMatrixIneq{T}) where {T <: Real}
    arr .= 0
    arr[1] = 1
    return arr
end

lmi_fact(arr::AbstractSparseMatrix) = cholesky(Hermitian(arr), shift=false, check=false)
lmi_fact(arr::AbstractMatrix) = cholesky!(Hermitian(arr), check=false)

function update_feas(cone::LinMatrixIneq{T}) where {T <: Real}
    @assert !cone.feas_updated

    cone.sumA = sum(w_i * A_i for (w_i, A_i) in zip(cone.point, cone.As))
    cone.fact = lmi_fact(cone.sumA)
    cone.is_feas = isposdef(cone.fact)

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::LinMatrixIneq) = true # TODO use a dikin ellipsoid condition?

function update_grad(cone::LinMatrixIneq{T}) where {T <: Real}
    @assert cone.is_feas

    sumAinvAs = cone.sumAinvAs = [Hermitian(cone.fact.L \ (cone.fact.L \ A_i)') for A_i in cone.As]
    @inbounds for (i, mat_i) in enumerate(sumAinvAs)
        cone.grad[i] = -tr(mat_i)
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::LinMatrixIneq)
    @assert cone.grad_updated
    sumAinvAs = cone.sumAinvAs
    H = cone.hess.data

    @inbounds for i in 1:cone.dim, j in i:cone.dim
        H[i, j] = real(dot(sumAinvAs[i], sumAinvAs[j]'))
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::LinMatrixIneq)
    @assert cone.grad_updated
    sumAinvAs = cone.sumAinvAs

    @inbounds for j in 1:size(arr, 2)
        j_mat = Hermitian(sum(arr[i, j] * sumAinvAs[i] for i in 1:cone.dim))
        for i in 1:cone.dim
            prod[i, j] = real(dot(j_mat, sumAinvAs[i]))
        end
    end

    return prod
end

function correction(cone::LinMatrixIneq, primal_dir::AbstractVector)
    @assert cone.grad_updated
    corr = cone.correction
    sumAinvAs = cone.sumAinvAs

    dir_mat = sum(d_i * mat_i for (d_i, mat_i) in zip(primal_dir, sumAinvAs))
    Z = Hermitian(dir_mat * dir_mat')
    @inbounds for i in 1:cone.dim
        corr[i] = real(dot(Z, sumAinvAs[i]))
    end

    return corr
end
