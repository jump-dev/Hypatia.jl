#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

row-wise lower triangle of positive semidefinite matrix cone (scaled "svec" form)
W \in S^n : 0 >= eigmin(W)

NOTE on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector
barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO
- describe svec scaling
- describe hermitian complex PSD cone
- try to derive faster neighborhood calculations for this cone specifically
=#

mutable struct PosSemidefTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    max_neighborhood::T
    dim::Int
    side::Int
    is_complex::Bool
    point::Vector{T}
    rt2::T
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}
    inv_mat::Matrix{R}
    fact_mat

    function PosSemidefTri{T, R}(
        dim::Int,
        is_dual::Bool,
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 1
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.max_neighborhood = default_max_neighborhood()
        cone.dim = dim # real vector dimension
        cone.rt2 = sqrt(T(2))
        if R <: Complex
            side = isqrt(dim) # real lower triangle and imaginary under diagonal
            @assert side^2 == dim
            cone.is_complex = true
        else
            side = round(Int, sqrt(0.25 + 2 * dim) - 0.5) # real lower triangle
            @assert side * (side + 1) == 2 * dim
            cone.is_complex = false
        end
        cone.side = side
        return cone
    end
end

PosSemidefTri{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = PosSemidefTri{T, R}(dim, false)

reset_data(cone::PosSemidefTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function setup_data(cone::PosSemidefTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.mat = zeros(R, cone.side, cone.side)
    cone.mat2 = similar(cone.mat)
    cone.mat3 = similar(cone.mat)
    cone.mat4 = similar(cone.mat)
    return
end

get_nu(cone::PosSemidefTri) = cone.side

function set_initial_point(arr::AbstractVector, cone::PosSemidefTri)
    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    @inbounds for i in 1:cone.side
        arr[k] = 1
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::PosSemidefTri)
    @assert !cone.feas_updated

    svec_to_smat!(cone.mat, cone.point, cone.rt2)
    copyto!(cone.mat2, cone.mat)
    cone.fact_mat = cholesky!(Hermitian(cone.mat2, :U), check = false)
    cone.is_feas = isposdef(cone.fact_mat)

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidefTri)
    @assert cone.is_feas

    cone.inv_mat = inv(cone.fact_mat)
    smat_to_svec!(cone.grad, cone.inv_mat, cone.rt2)
    cone.grad .*= -1
    copytri!(cone.mat, 'U', cone.is_complex)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::PosSemidefTri)
    @assert cone.grad_updated
    symm_kron(cone.hess.data, cone.inv_mat, cone.rt2)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::PosSemidefTri)
    @assert is_feas(cone)
    symm_kron(cone.inv_hess.data, cone.mat, cone.rt2)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert is_feas(cone)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', cone.is_complex)
        rdiv!(cone.mat4, cone.fact_mat)
        ldiv!(cone.fact_mat, cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert is_feas(cone)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        mul!(cone.mat3, Hermitian(cone.mat4, :U), cone.mat)
        mul!(cone.mat4, Hermitian(cone.mat, :U), cone.mat3)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end
    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert is_feas(cone)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', cone.is_complex)
        rdiv!(cone.mat4, cone.fact_mat.U)
        ldiv!(cone.fact_mat.U', cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end
    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert is_feas(cone)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', cone.is_complex)
        rmul!(cone.mat4, cone.fact_mat.U')
        lmul!(cone.fact_mat.U, cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end
    return prod
end
