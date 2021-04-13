#=
svec-scaled sparse positive semidefinite matrix cone
mat(w) PSD

specified with ordered lists of row and column indices for elements in lower triangle
must include all diagonal elements
dual cone is cone of PSD-completable matrices with the given sparsity pattern
real symmetric or complex Hermitian cases
NOTE in complex Hermitian case, on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector, but row and column indices are not repeated
=#

abstract type PSDSparseImpl end

abstract type PSDSparseCache{T <: Real, R <: RealOrComplex{T}} end

mutable struct PosSemidefTriSparse{I <: PSDSparseImpl, T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    side::Int
    row_idxs::Vector{Int}
    col_idxs::Vector{Int}
    is_complex::Bool
    rt2::T

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
    use_hess_prod_slow::Bool
    use_hess_prod_slow_updated::Bool

    cache::PSDSparseCache{T, R}

    function PosSemidefTriSparse{I, T, R}(
        side::Int,
        row_idxs::Vector{Int},
        col_idxs::Vector{Int};
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {I <: PSDSparseImpl, R <: RealOrComplex{T}} where {T <: Real}
        # check validity of inputs
        num_nz = length(row_idxs)
        @assert length(col_idxs) == num_nz
        # TODO maybe also check no off-diags appear twice?
        diag_present = falses(side)
        for (row_idx, col_idx) in zip(row_idxs, col_idxs)
            @assert col_idx <= row_idx <= side
            if row_idx == col_idx
                @assert !diag_present[row_idx] # don't count element twice
                diag_present[row_idx] = true
            end
        end
        @assert all(diag_present)
        cone = new{I, T, R}()
        if R <: Real
            cone.dim = num_nz
            cone.is_complex = false
        else
            cone.dim = 2 * num_nz - side
            cone.is_complex = true
        end
        @assert cone.dim >= 1
        cone.use_dual_barrier = use_dual
        cone.side = side # side dimension of sparse matrix
        cone.row_idxs = row_idxs
        cone.col_idxs = col_idxs
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

reset_data(cone::PosSemidefTriSparse) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = cone.use_hess_prod_slow = cone.use_hess_prod_slow_updated = false)

function setup_extra_data(cone::PosSemidefTriSparse{<:PSDSparseImpl, T, <:RealOrComplex{T}}) where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    setup_psdsparse_cache(cone)
    return cone
end

get_nu(cone::PosSemidefTriSparse) = cone.side

function set_initial_point(arr::AbstractVector, cone::PosSemidefTriSparse)
    idx = 1
    incr = (cone.is_complex ? 2 : 1)
    fill!(arr, 0)
    @inbounds for (row_idx, col_idx) in zip(cone.row_idxs, cone.col_idxs)
        if row_idx == col_idx
            arr[idx] = 1
            idx += 1
        else
            idx += incr
        end
    end
    return arr
end

is_dual_feas(cone::PosSemidefTriSparse) = true # TODO try completable matrix test

include("possemideftrisparse/denseimpl.jl")
include("possemideftrisparse/cholmodimpl.jl")
const PSDSparseImplList = [
    (Cones.PSDSparseDense, Real),
    (Cones.PSDSparseCholmod, LinearAlgebra.BlasReal),
    ]
