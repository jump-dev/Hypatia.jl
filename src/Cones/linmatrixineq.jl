#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

TODO
- write description
- assumes first A matrix is PSD (eg identity)
- loosen type signature - some of the matrices could be eg diagonal or identity types
- reduce allocs etc
=#

mutable struct LinMatrixIneq{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    As::Vector{HermOrSym{R, Matrix{R}} where {R <: RealOrComplex{T}}}
    is_complex::Bool
    point::Vector{T}
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

    sumA::HermOrSym
    fact
    sumAinvAs

    function LinMatrixIneq{T}(
        As::Vector,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        dim = length(As)
        @assert dim > 1
        side = size(first(As), 1)
        @assert side > 1
        for A_i in As
            @assert A_i isa HermOrSym{R, Matrix{R}} where {R <: RealOrComplex{T}}
            @assert size(A_i, 1) == side
            @assert eltype(A_i) <: RealOrComplex{T}
        end
        @assert isposdef(first(As)) # TODO reuse factorization from here later if useful
        # TODO check all are upper tri hermitian if useful?
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.side = side
        cone.As = As
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

LinMatrixIneq{T}(As::Vector) where {T <: Real} = LinMatrixIneq{T}(As, false)

# TODO only allocate the fields we use
function setup_data(cone::LinMatrixIneq{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    return
end

get_nu(cone::LinMatrixIneq) = cone.side

function set_initial_point(arr::AbstractVector, cone::LinMatrixIneq{T}) where {T <: Real}
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::LinMatrixIneq)
    @assert !cone.feas_updated

    cone.sumA = sum(w_i * A_i for (w_i, A_i) in zip(cone.point, cone.As)) # TODO in-place with 5-arg mul
    @assert ishermitian(cone.sumA) # TODO delete
    cone.fact = cholesky!(cone.sumA, check = false)
    cone.is_feas = isposdef(cone.fact)

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::LinMatrixIneq)
    @assert cone.is_feas

    # grad[i] = -tr(inv(sumA) * A[i])
    cone.sumAinvAs = [cone.fact \ A_i for A_i in cone.As] # TODO make efficient and save ldiv
    for i in 1:cone.dim
        sumAinvAsi = cone.sumAinvAs[i]
        @inbounds cone.grad[i] = -sum(real(sumAinvAsi[k, k]) for k in 1:cone.side)
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::LinMatrixIneq)
    @assert cone.is_feas
    sumAinvAs = cone.sumAinvAs
    H = cone.hess.data

    # H[i, j] = tr((cone.fact \ A_i) * (cone.fact \ A_j))
    for i in 1:cone.dim, j in i:cone.dim
        @inbounds H[i, j] = real(dot(sumAinvAs[i], sumAinvAs[j]'))
    end

    cone.hess_updated = true
    return cone.hess
end
