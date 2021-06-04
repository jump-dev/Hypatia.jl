"""
$(TYPEDEF)

An implementation type for the sparse positive semidefinite cone
[`PosSemidefTriSparse`](@ref).
"""
abstract type PSDSparseImpl end

# a cache for an implementation for the sparse positive semidefinite cone
abstract type PSDSparseCache{T <: Real, R <: RealOrComplex{T}} end

"""
$(TYPEDEF)

Real symmetric or complex Hermitian sparse positive semidefinite cone of side
dimension `side` and sparse lower triangle row and column indices `rows`, `cols`
in svec format. Note all diagonal elements must be present.

    $(FUNCTIONNAME){T, R}(side::Int, rows::Vector{Int}, cols::Vector{Int}, use_dual::Bool = false)
"""
mutable struct PosSemidefTriSparse{I <: PSDSparseImpl, T <: Real,
    R <: RealOrComplex{T}} <: Cone{T}
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

    cache::PSDSparseCache{T, R}

    function PosSemidefTriSparse{I, T, R}(
        side::Int,
        row_idxs::Vector{Int},
        col_idxs::Vector{Int};
        use_dual::Bool = false,
        ) where {I <: PSDSparseImpl, T <: Real, R <: RealOrComplex{T}}
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
        return cone
    end
end

reset_data(cone::PosSemidefTriSparse) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated =
    cone.use_hess_prod_slow = cone.use_hess_prod_slow_updated = false)

get_nu(cone::PosSemidefTriSparse) = cone.side

function set_initial_point!(arr::AbstractVector, cone::PosSemidefTriSparse)
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

include("denseimpl.jl")
# cholmod implementation only works after this commit
# see https://github.com/JuliaLang/julia/pull/40560
if VERSION >= v"1.7.0-DEV.1025"
    include("cholmodimpl.jl")
    const PSDSparseImplList = [
        (PSDSparseDense, Real),
        (PSDSparseCholmod, LinearAlgebra.BlasReal),
        ]
else
    const PSDSparseCholmod = PSDSparseDense
    const PSDSparseImplList = [(PSDSparseDense, Real),]
end
