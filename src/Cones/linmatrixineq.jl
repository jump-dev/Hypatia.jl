"""
$(TYPEDEF)

Linear matrix inequality cone parametrized by list of real symmetric or complex
Hermitian matrices `mats` of equal dimension.

    $(FUNCTIONNAME){T}(mats::Vector, use_dual::Bool = false)
"""
mutable struct LinMatrixIneq{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    side::Int
    As::Vector

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
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}
    use_hess_prod_slow::Bool
    use_hess_prod_slow_updated::Bool

    sumA::AbstractMatrix
    fact
    sumAinvAs::Vector

    function LinMatrixIneq{T}(
        As::Vector;
        use_dual::Bool = false,
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
        # necessary to ensure linear independence of As (but not sufficient)
        @assert svec_length(side) >= dim
        @assert isposdef(first(As))
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.side = side
        cone.As = As
        return cone
    end
end

reset_data(cone::LinMatrixIneq) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated =
    cone.use_hess_prod_slow = cone.use_hess_prod_slow_updated = false)

get_nu(cone::LinMatrixIneq) = cone.side

function set_initial_point!(
    arr::AbstractVector,
    cone::LinMatrixIneq{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = 1
    return arr
end

lmi_fact(arr::AbstractSparseMatrix) = cholesky(Hermitian(arr),
    shift=false, check=false)
lmi_fact(arr::AbstractMatrix) = cholesky!(Hermitian(arr), check=false)

function update_feas(cone::LinMatrixIneq{T}) where {T <: Real}
    @assert !cone.feas_updated

    cone.sumA = sum(w_i * A_i for (w_i, A_i) in zip(cone.point, cone.As))
    cone.fact = lmi_fact(cone.sumA)
    cone.is_feas = isposdef(cone.fact)

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::LinMatrixIneq{T}) where {T <: Real}
    @assert cone.is_feas

    L = cone.fact.L
    cone.sumAinvAs = [Hermitian(L \ (L \ A_i)', :U) for A_i in cone.As]
    @inbounds for (i, mat_i) in enumerate(cone.sumAinvAs)
        cone.grad[i] = -tr(mat_i)
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::LinMatrixIneq)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    sumAinvAs = cone.sumAinvAs

    @inbounds for i in 1:cone.dim, j in i:cone.dim
        H[i, j] = real(dot(sumAinvAs[i], sumAinvAs[j]'))
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod_slow!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::LinMatrixIneq,
    )
    cone.use_hess_prod_slow_updated || update_use_hess_prod_slow(cone)
    @assert cone.hess_updated
    cone.use_hess_prod_slow || return hess_prod!(prod, arr, cone)

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

function dder3(cone::LinMatrixIneq, dir::AbstractVector)
    @assert cone.grad_updated
    dder3 = cone.dder3
    sumAinvAs = cone.sumAinvAs

    dir_mat = sum(d_i * mat_i for (d_i, mat_i) in zip(dir, sumAinvAs))
    Z = Hermitian(dir_mat * dir_mat')
    @inbounds for i in 1:cone.dim
        dder3[i] = real(dot(Z, sumAinvAs[i]))
    end

    return dder3
end
