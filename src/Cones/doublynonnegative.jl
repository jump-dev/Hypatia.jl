#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

smat(w) in S_+^d intersected with w in R_+^(sdim(d))

barrier -logdet(W) - sum(log(W_ij) for i in 1:n, j in 1:(i-1))
where W = smat(w)

TODO
initial point
better description

=#

mutable struct DoublyNonnegative{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    side::Int
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
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    mat::Matrix{T}
    mat2::Matrix{T}
    mat3::Matrix{T}
    offdiag_idxs
    inv_mat::Matrix{T}
    inv_vec::Vector{T}
    fact_mat

    function DoublyNonnegative{T}(
        dim::Int;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        side = round(Int, sqrt(0.25 + 2 * dim) - 0.5)
        @assert side * (side + 1) == 2 * dim
        cone.side = side
        cone.offdiag_idxs = vcat([(sum(1:(i - 1)) + 1):(sum(1:i) - 1) for i in 2:side]...) # TODO better way without splatting?
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

use_heuristic_neighborhood(cone::DoublyNonnegative) = false

reset_data(cone::DoublyNonnegative) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

function setup_data(cone::DoublyNonnegative{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.mat = zeros(T, cone.side, cone.side)
    cone.mat2 = similar(cone.mat)
    cone.mat3 = similar(cone.mat)
    cone.inv_vec = zeros(T, svec_length(cone.side))
    return
end

get_nu(cone::DoublyNonnegative) = cone.dim

function set_initial_point(arr::AbstractVector, cone::DoublyNonnegative)
    if cone.dim == 3
        arr .= [1.1180340045943178, 0.5000000098333939 * sqrt(2), 1.1180340045943178]
    else
        arr .= 1
        k = 1
        @inbounds for i in 1:cone.side
            arr[k] = cone.side
            k += i + 1
        end
    end
    return arr
end

function update_feas(cone::DoublyNonnegative)
    @assert !cone.feas_updated

    if all(u -> (u > 0), cone.point)
        svec_to_smat!(cone.mat, cone.point, cone.rt2)
        copyto!(cone.mat2, cone.mat)
        cone.fact_mat = cholesky!(Symmetric(cone.mat2, :U), check = false)
        cone.is_feas = isposdef(cone.fact_mat)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::DoublyNonnegative)
    @assert cone.is_feas

    cone.inv_mat = inv(cone.fact_mat)
    smat_to_svec!(cone.grad, cone.inv_mat, cone.rt2)
    cone.grad .*= -1
    copytri!(cone.mat, 'U')
    @. @views cone.inv_vec[cone.offdiag_idxs] = inv(cone.point[cone.offdiag_idxs])
    @. cone.grad -= cone.inv_vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::DoublyNonnegative)
    @assert cone.grad_updated
    symm_kron(cone.hess.data, cone.inv_mat, cone.rt2)
    cone.hess.data .+= Diagonal(abs2.(cone.inv_vec))
    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::DoublyNonnegative)
    @assert is_feas(cone)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat3, view(arr, :, i), cone.rt2)
        copytri!(cone.mat3, 'U')
        rdiv!(cone.mat3, cone.fact_mat)
        ldiv!(cone.fact_mat, cone.mat3)
        smat_to_svec!(view(prod, :, i), cone.mat3, cone.rt2)
    end
    @. @views prod[cone.offdiag_idxs, :] += arr[cone.offdiag_idxs, :] / cone.point[cone.offdiag_idxs, :] / cone.point[cone.offdiag_idxs, :]
    return prod
end
