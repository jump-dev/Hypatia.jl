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
    use_dual_barrier::Bool
    max_neighborhood::T
    dim::Int
    side::Int
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}
    rt2::T
    timer::TimerOutput

    feas_updated::Bool
    dual_feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    scal_hess_updated::Bool
    nt_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    mat::Matrix{R}
    dual_mat::Matrix{R} # NOTE in old branch this was named dual_fact_mat
    mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}
    inv_mat::Matrix{R}
    fact_mat
    dual_fact_mat # NOTE in old branch this was named dual_fact

    scalmat_sqrt::Matrix{R}
    scalmat_sqrti::Matrix{R}
    correction::Vector{T}

    function PosSemidefTri{T, R}(
        dim::Int;
        use_dual::Bool = false, # TODO self-dual so maybe remove this option/field?
        max_neighborhood::Real = default_max_neighborhood(),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 1
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.max_neighborhood = max_neighborhood
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

use_heuristic_neighborhood(cone::PosSemidefTri) = false

reset_data(cone::PosSemidefTri) = (cone.feas_updated = cone.dual_feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.scal_hess_updated = cone.nt_updated = false)

use_nt(::PosSemidefTri) = true

use_correction(::PosSemidefTri) = true

use_scaling(::PosSemidefTri) = true

function setup_data(cone::PosSemidefTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.mat = zeros(R, cone.side, cone.side)
    cone.dual_mat = zeros(R, cone.side, cone.side)
    cone.mat2 = similar(cone.mat)
    cone.mat3 = similar(cone.mat)
    cone.mat4 = similar(cone.mat)
    cone.scalmat_sqrt = Matrix{T}(I, cone.side, cone.side)
    cone.scalmat_sqrti = Matrix{T}(I, cone.side, cone.side)
    cone.correction = zeros(T, dim)
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

function update_dual_feas(cone::PosSemidefTri)
    svec_to_smat!(cone.dual_mat, cone.point, cone.rt2)
    copyto!(cone.mat2, cone.dual_mat)
    cone.dual_fact_mat = cholesky!(Hermitian(cone.mat2, :U), check = false)
    cone.dual_feas_updated = true
    return isposdef(cone.dual_fact_mat)
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

function update_scal_hess(cone::PosSemidefTri{T}, mu::T) where {T}
    @assert cone.grad_updated
    cone.nt_updated || update_nt(cone)
    symm_kron(cone.hess.data, cone.scalmat_sqrti' * cone.scalmat_sqrti, cone.rt2) # TODO fix mul
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

function scal_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert is_feas(cone)
    cone.nt_updated || update_nt(cone)
    mul!(cone.mat4, cone.scalmat_sqrti', cone.scalmat_sqrti) # TODO fix inefficiency of the mul
    herm_congruence_prod!(prod, arr, cone.mat4, cone)
    return prod
end

# PXP where P is Hermitian
function herm_congruence_prod!(prod::AbstractVecOrMat, inner::AbstractVecOrMat, outer::AbstractVecOrMat, cone::PosSemidefTri)
    @inbounds for i in 1:size(inner, 2)
        svec_to_smat!(cone.mat3, view(inner, :, i), cone.rt2)
        mul!(cone.mat2, Hermitian(cone.mat3, :U), outer)
        mul!(cone.mat3, Hermitian(outer, :U), cone.mat2)
        smat_to_svec!(view(prod, :, i), cone.mat3, cone.rt2)
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

function scal_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert cone.is_feas
    cone.nt_updated || update_nt(cone)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat3, view(arr, :, i), cone.rt2)
        mul!(cone.work_mat, Hermitian(cone.mat3, :U), cone.scalmat_sqrti')
        mul!(cone.mat3, cone.scalmat_sqrti, cone.work_mat)
        smat_to_svec!(view(prod, :, i), cone.mat3, cone.rt2)
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

# old step_and_update_scaling
function update_nt(cone::PosSemidefTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.feas_updated
    cone.dual_feas_updated || update_dual_feas(cone)
    # TODO preallocate
    (U, lambda, V) = svd(cone.dual_fact_mat.U * cone.fact_mat.L)
    cone.scalmat_sqrt = cone.fact_mat.L * V * Diagonal(inv.(sqrt.(lambda)))
    cone.scalmat_sqrti = Diagonal(inv.(sqrt.(lambda))) * U' * cone.dual_fact_mat.U

    return
end

function correction(cone::PosSemidefTri, primal_dir::AbstractVector, dual_dir::AbstractVector)#, primal_point)
    @assert cone.grad_updated

    S = copytri!(svec_to_smat!(cone.mat2, primal_dir, cone.rt2), 'U', cone.is_complex)
    Z = Hermitian(svec_to_smat!(cone.mat3, dual_dir, cone.rt2))

    # TODO compare the following numerically
    # Pinv_S_Z = mul!(cone.work_mat3, ldiv!(cone.fact, S), Z)
    # Pinv_S_Z = ldiv!(cone.fact, mul!(cone.work_mat3, S, Z))
    # TODO reuse factorization if useful
    # fact = cholesky(Hermitian(svec_to_smat!(cone.work_mat3, primal_point, cone.rt2), :U))
    fact = cholesky(Hermitian(svec_to_smat!(cone.mat4, cone.point, cone.rt2), :U))
    Pinv_S_Z = mul!(cone.mat4, ldiv!(fact, S), Z)

    Pinv_S_Z_symm = cone.mat2
    @. Pinv_S_Z_symm = (Pinv_S_Z + Pinv_S_Z') / 2
    smat_to_svec!(cone.correction, Pinv_S_Z_symm, cone.rt2)

    return cone.correction
end
