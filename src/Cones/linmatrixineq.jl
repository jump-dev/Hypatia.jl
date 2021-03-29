#=
TODO
- write description
- assumes first A matrix is PSD (eg identity)
=#

import SuiteSparse
# TODO remove if https://github.com/JuliaLang/julia/pull/40250 is merged
import LinearAlgebra.dot
dot(A::AbstractMatrix, J::UniformScaling) = tr(A) * J.λ

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

    sumA
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

lmi_fact(arr::Union{UniformScaling{R}, Diagonal{R}}) where {R} = arr # NOTE could use SymTridiagonal here when that type gets a isposdef and ldiv in Julia
lmi_fact(arr::AbstractSparseMatrix{R}) where {R} = cholesky(Hermitian(arr), shift=false, check=false)
lmi_fact(arr::AbstractMatrix{R}) where {R} = cholesky!(Hermitian(arr), check=false)

rdiv_sqrt!(arr::AbstractMatrix{R}, fact::UniformScaling{R}) where {R} = arr ./ sqrt(fact.λ)
rdiv_sqrt!(arr::AbstractMatrix{R}, fact::Diagonal{R}) where {R} = @. arr / sqrt(fact)
rdiv_sqrt!(arr::AbstractMatrix{R}, fact::Cholesky) where {R} = rdiv!(arr, fact.U)
rdiv_sqrt!(arr::AbstractMatrix{R}, fact::SuiteSparse.CHOLMOD.Factor{R}) where {R} = (fact.L \ arr)'

function update_feas(cone::LinMatrixIneq{T}) where {T <: Real}
    @assert !cone.feas_updated

    # NOTE not in-place because typeof(A) is AbstractMatrix eg sparse
    # TODO if sumA is dense, can do in-place
    cone.sumA = sum(w_i * A_i for (w_i, A_i) in zip(cone.point, cone.As))
    cone.fact = lmi_fact(cone.sumA)
    cone.is_feas = isposdef(cone.fact)

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::LinMatrixIneq) = true # TODO use a dikin ellipsoid condition?

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

function correction(cone::LinMatrixIneq, primal_dir::AbstractVector)
    @assert cone.feas_updated
    corr = cone.correction
    As = cone.As
    fact = cone.fact

    # TODO specialize if As are all Hermitian AbstractMatrix
    # dir_mat = sum(d_i * A_i for (d_i, A_i) in zip(primal_dir, As)).data
    # LinearAlgebra.copytri!(dir_mat, As[1].uplo)
    # ldiv!(fact, dir_mat)
    # rdiv!(dir_mat, fact.U)
    # M = dir_mat * dir_mat'
    dir_mat = sum(d_i * A_i for (d_i, A_i) in zip(primal_dir, As))
    Y1 = fact \ dir_mat
    Y2 = rdiv_sqrt!(Y1, fact)
    M = Y2 * Y2'
    for i in 1:cone.dim
        corr[i] = real(dot(M, As[i]))
    end

    return corr
end
