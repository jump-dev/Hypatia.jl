"""
$(TYPEDEF)

Real symmetric or complex Hermitian positive semidefinite cone of dimension
`dim` in svec format.

    $(FUNCTIONNAME){T, R}(dim::Int)
"""
mutable struct PosSemidefTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    dim::Int
    side::Int
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
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}
    inv_mat::Matrix{R}
    fact_mat::Cholesky{R}

    function PosSemidefTri{T, R}(
        dim::Int,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert dim >= 1
        cone = new{T, R}()
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        cone.is_complex = (R <: Complex)
        cone.side = svec_side(R, dim)
        return cone
    end
end

use_dual_barrier(::PosSemidefTri) = false

reset_data(cone::PosSemidefTri) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = false)

use_sqrt_hess_oracles(::Int, cone::PosSemidefTri) = true

function setup_extra_data!(
    cone::PosSemidefTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    cone.mat = zeros(R, cone.side, cone.side)
    cone.mat2 = zero(cone.mat)
    cone.mat3 = zero(cone.mat)
    cone.mat4 = zero(cone.mat)
    cone.inv_mat = zero(cone.mat)
    return cone
end

get_nu(cone::PosSemidefTri) = cone.side

function set_initial_point!(arr::AbstractVector, cone::PosSemidefTri)
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

function is_dual_feas(cone::PosSemidefTri)
    svec_to_smat!(cone.mat3, cone.dual_point, cone.rt2)
    return isposdef(cholesky!(Hermitian(cone.mat3, :U), check = false))
end

function update_grad(cone::PosSemidefTri)
    @assert cone.is_feas

    inv_fact!(cone.inv_mat, cone.fact_mat)
    smat_to_svec!(cone.grad, cone.inv_mat, cone.rt2)
    cone.grad .*= -1
    copytri!(cone.mat, 'U', true)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::PosSemidefTri)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    copytri!(cone.inv_mat, 'U', true)
    symm_kron!(cone.hess.data, cone.inv_mat, cone.rt2)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::PosSemidefTri)
    @assert is_feas(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    symm_kron!(cone.inv_hess.data, cone.mat, cone.rt2)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTri,
    )
    @assert is_feas(cone)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', true)
        rdiv!(cone.mat4, cone.fact_mat)
        ldiv!(cone.fact_mat, cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTri,
    )
    @assert is_feas(cone)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        mul!(cone.mat3, Hermitian(cone.mat4, :U), cone.mat)
        mul!(cone.mat4, Hermitian(cone.mat, :U), cone.mat3)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end

    return prod
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTri,
    )
    @assert is_feas(cone)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', true)
        rdiv!(cone.mat4, cone.fact_mat.U)
        ldiv!(cone.fact_mat.U', cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end

    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTri,
    )
    @assert is_feas(cone)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', true)
        rmul!(cone.mat4, cone.fact_mat.U')
        lmul!(cone.fact_mat.U, cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end

    return prod
end

function dder3(cone::PosSemidefTri, dir::AbstractVector)
    @assert cone.grad_updated

    S = copytri!(svec_to_smat!(cone.mat4, dir, cone.rt2), 'U', true)
    ldiv!(cone.fact_mat, S)
    rdiv!(S, cone.fact_mat.U)
    mul!(cone.mat3, S, S') # TODO use outer prod function
    smat_to_svec!(cone.dder3, cone.mat3, cone.rt2)

    return cone.dder3
end
