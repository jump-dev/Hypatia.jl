#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

sparse lower triangle of positive semidefinite matrix cone (unscaled "smat" form)
W \in S^n : 0 >= eigmin(W)

NOTE on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector

dual is sparse PSD completable

TODO
- describe
- hermitian case
- reference
=#

mutable struct PosSemidefTriSparse{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    row_idxs::Vector{Int}
    col_idxs::Vector{Int}
    is_complex::Bool
    point::Vector{T}
    rt2::T
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

    fact_mat
    inv_mat

    function PosSemidefTriSparse{T, R}(
        side::Int,
        row_idxs::Vector{Int},
        col_idxs::Vector{Int},
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T), # TODO get inverse hessian directly
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert R <: Real # TODO generalize and delete
        dim = length(row_idxs) # TODO doesn't work for complex
        @assert dim >= 1
        @assert length(col_idxs) == dim
        @assert all(col_idxs .<= row_idxs .<= side) # TODO improve efficiency
        # TODO check diagonals are all present. maybe diagonals go first in point, then remaining elements are just the nonzeros off diag, and row/col idxs are only the off diags
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.dim = dim # real vector dimension
        cone.side = side # side dimension of sparse matrix
        cone.row_idxs = row_idxs
        cone.col_idxs = col_idxs
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

PosSemidefTriSparse{T, R}(side::Int, row_idxs::Vector{Int}, col_idxs::Vector{Int}) where {R <: RealOrComplex{T}} where {T <: Real} = PosSemidefTriSparse{T, R}(side, row_idxs, col_idxs, false)

# reset_data(cone::PosSemidefTriSparse) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function setup_data(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    return
end

get_nu(cone::PosSemidefTriSparse) = cone.side

function set_initial_point(arr::AbstractVector, cone::PosSemidefTriSparse)
    # TODO set diagonal elements to 1
    # TODO improve efficiency - maybe order diag elements at start
    for i in 1:cone.dim
        if cone.row_idxs[i] == cone.col_idxs[i]
            arr[i] = 1
        else
            arr[i] = 0
        end
    end
    return arr
end

function update_feas(cone::PosSemidefTriSparse)
    @assert !cone.feas_updated

    scal_point = copy(cone.point)
    for i in 1:cone.dim
        if cone.row_idxs[i] != cone.col_idxs[i]
            scal_point[i] /= cone.rt2
        end
    end
    mat = Symmetric(Matrix(sparse(cone.row_idxs, cone.col_idxs, scal_point, cone.side, cone.side)), :L) # TODO not dense
    cone.fact_mat = cholesky(mat, check = false)
    cone.is_feas = isposdef(cone.fact_mat)

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidefTriSparse)
    @assert cone.is_feas

    cone.inv_mat = inv(cone.fact_mat)
    for i in 1:cone.dim
        cone.grad[i] = -cone.inv_mat[cone.row_idxs[i], cone.col_idxs[i]]
        if cone.row_idxs[i] != cone.col_idxs[i]
            cone.grad[i] *= cone.rt2
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::PosSemidefTriSparse)
    @assert cone.grad_updated
    H = cone.hess.data

    svec_dim = svec_length(cone.side)
    full_kron = Symmetric(symm_kron(zeros(eltype(H), svec_dim, svec_dim), cone.inv_mat, cone.rt2), :U)
    for i in 1:cone.dim, j in i:cone.dim
        H[i, j] = full_kron[svec_idx(cone.row_idxs[i], cone.col_idxs[i]), svec_idx(cone.row_idxs[j], cone.col_idxs[j])]
    end

    cone.hess_updated = true
    return cone.hess
end
