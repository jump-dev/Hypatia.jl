#=
Copyright 2018, Chris Coey and contributors

TODO describe hermitian complex PSD cone
on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector

row-wise lower triangle of positive semidefinite matrix cone
W \in S^n : 0 >= eigmin(W)
(see equivalent MathOptInterface PositiveSemidefiniteConeTriangle definition)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO
- eliminate allocations for inverse-finding
=#

mutable struct PosSemidefTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    point::AbstractVector{T}
    is_complex::Bool

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    inv_mat::Matrix{R}
    fact_mat

    function PosSemidefTri{T, R}(dim::Int, is_dual::Bool) where {R <: RealOrComplex{T}} where {T <: Real}
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.dim = dim # real vector dimension
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

reset_data(cone::PosSemidefTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_prod_updated = false)

function setup_data(cone::PosSemidefTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.mat = zeros(R, cone.side, cone.side)
    cone.mat2 = similar(cone.mat)
    cone.mat3 = similar(cone.mat)
    return
end

get_nu(cone::PosSemidefTri) = cone.side

function set_initial_point(arr::AbstractVector, cone::PosSemidefTri)
    incr = cone.is_complex ? 2 : 1
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
    vec_to_mat_U!(cone.mat, cone.point)
    copyto!(cone.mat2, cone.mat)
    cone.fact_mat = hyp_chol!(Hermitian(cone.mat2, :U)) # TODO eliminate allocs
    cone.is_feas = isposdef(cone.fact_mat)
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidefTri)
    @assert cone.is_feas
    cone.inv_mat = inv(cone.fact_mat) # TODO eliminate allocs
    mat_U_to_vec_scaled!(cone.grad, cone.inv_mat)
    cone.grad .*= -1
    cone.grad_updated = true
    return cone.grad
end

# TODO parallelize
function _build_hess(H::Matrix{T}, mat::Matrix{T}, is_inv::Bool) where {T <: Real}
    side = size(mat, 1)
    scale1 = (is_inv ? inv(T(2)) : T(2))
    scale2 = (is_inv ? one(T) : scale1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        @inbounds for i2 in 1:side, j2 in 1:i2
            if (i == j) && (i2 == j2)
                H[k2, k] = abs2(mat[i2, i])
            elseif (i != j) && (i2 != j2)
                H[k2, k] = scale1 * (mat[i2, i] * mat[j, j2] + mat[j2, i] * mat[j, i2])
            else
                H[k2, k] = scale2 * mat[i2, i] * mat[j, j2]
            end
            if k2 == k
                break
            end
            k2 += 1
        end
        k += 1
    end
    return H
end

function _build_hess(H::Matrix{T}, mat::Matrix{Complex{T}}, is_inv::Bool) where {T <: Real}
    side = size(mat, 1)
    scale1 = (is_inv ? inv(T(2)) : T(2))
    scale2 = (is_inv ? one(T) : scale1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        if i == j
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    H[k2, k] = abs2(mat[i2, i])
                    k2 += 1
                else
                    c = scale2 * mat[i, i2] * mat[j2, j]
                    H[k2, k] = real(c)
                    k2 += 1
                    H[k2, k] = -imag(c)
                    k2 += 1
                end
                if k2 > k
                    break
                end
            end
            k += 1
        else
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    c = scale2 * mat[i2, i] * mat[j, j2]
                    H[k2, k] = real(c)
                    H[k2, k + 1] = -imag(c)
                    k2 += 1
                else
                    b1 = mat[i2, i] * mat[j, j2]
                    b2 = mat[j2, i] * mat[j, i2]
                    c1 = scale1 * (b1 + b2)
                    H[k2, k] = real(c1)
                    H[k2, k + 1] = -imag(c1)
                    k2 += 1
                    c2 = scale1 * (b1 - b2)
                    H[k2, k] = imag(c2)
                    H[k2, k + 1] = real(c2)
                    k2 += 1
                end
                if k2 > k
                    break
                end
            end
            k += 2
        end
    end
    return H
end

function update_hess(cone::PosSemidefTri)
    @assert cone.grad_updated
    _build_hess(cone.hess.data, cone.inv_mat, false)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::PosSemidefTri)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone) # need cone.mat to be symmetric/Hermitian
    end
    _build_hess(cone.inv_hess.data, cone.mat, true)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function update_inv_hess_prod(cone::PosSemidefTri)
    @assert is_feas(cone)
    copytri!(cone.mat, 'U', cone.is_complex)
    cone.inv_hess_prod_updated = true
    return nothing
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert cone.grad_updated
    @inbounds for i in 1:size(arr, 2)
        vec_to_mat_U!(cone.mat2, view(arr, :, i))
        mul!(cone.mat3, Hermitian(cone.mat2, :U), cone.inv_mat)
        mul!(cone.mat2, Hermitian(cone.inv_mat, :U), cone.mat3)
        mat_U_to_vec_scaled!(view(prod, :, i), cone.mat2)
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    @inbounds for i in 1:size(arr, 2)
        vec_to_mat_U_scaled!(cone.mat2, view(arr, :, i))
        mul!(cone.mat3, Hermitian(cone.mat2, :U), cone.mat)
        mul!(cone.mat2, Hermitian(cone.mat, :U), cone.mat3)
        mat_U_to_vec!(view(prod, :, i), cone.mat2)
    end
    return prod
end
