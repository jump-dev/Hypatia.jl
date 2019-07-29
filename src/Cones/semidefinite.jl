#=
Copyright 2018, Chris Coey and contributors

TODO describe hermitian complex PSD cone
on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector

row-wise lower triangle (svec space) of positive semidefinite matrix cone
(smat space) W \in S^n : 0 >= eigmin(W)
(see equivalent MathOptInterface PositiveSemidefiniteConeTriangle definition)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO
- eliminate allocations for inverse-finding

- not make multiplication operators the default
=#

mutable struct PosSemidef{T <: HypReal, R <: HypRealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    point::AbstractVector{T}
    is_complex::Bool
    use_hess_prod_ops

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    mat::HermOrSym{R, Matrix{R}}
    mat2::HermOrSym{R, Matrix{R}}
    mat3::HermOrSym{R, Matrix{R}}
    inv_mat::HermOrSym{R, Matrix{R}}
    fact_mat

    function PosSemidef{T, R}(dim::Int, is_dual::Bool, use_hess_prod_ops::Bool) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.dim = dim # real vector dimension
        cone.use_hess_prod_ops = use_hess_prod_ops
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

PosSemidef{T, R}(dim::Int; use_hess_prod_ops::Bool = true) where {R <: HypRealOrComplex{T}} where {T <: HypReal} = PosSemidef{T, R}(dim, false, use_hess_prod_ops)

reset_data(cone::PosSemidef) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function setup_data(cone::PosSemidef{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
    reset_data(cone)
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    if cone.is_complex
        cone.mat = Hermitian(zeros(R, cone.side, cone.side), :L)
    else
        cone.mat = Symmetric(zeros(R, cone.side, cone.side), :L)
    end
    cone.mat2 = similar(cone.mat)
    cone.mat3 = similar(cone.mat)
    return
end

get_nu(cone::PosSemidef) = cone.side

function set_initial_point(arr::AbstractVector, cone::PosSemidef)
    incr = cone.is_complex ? 2 : 1
    arr .= 0
    k = 1
    for i in 1:cone.side
        arr[k] = 1
        k += incr * i + 1
    end
    return arr
end

# TODO only work with upper triangle
function update_feas(cone::PosSemidef)
    @assert !cone.feas_updated
    vec_to_mat_L!(cone.mat.data, cone.point)
    # @show cone.mat.data
    # @show cone.mat
    copyto!(cone.mat2, cone.mat)
    cone.fact_mat = hyp_chol!(cone.mat2)
    cone.is_feas = isposdef(cone.fact_mat)
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidef)
    @assert cone.is_feas
    if cone.is_complex
        # @show inv(cone.fact_mat)
        cone.inv_mat = Hermitian(inv(cone.fact_mat), :L) # TODO eliminate allocs
    else
        cone.inv_mat = Symmetric(inv(cone.fact_mat), :L) # TODO eliminate allocs
    end
    # @show cone.inv_mat
    # @show transpose(cone.inv_mat.data)
    smat_to_svec!(cone.grad, (cone.inv_mat))
    cone.grad .*= -1
    cone.grad_updated = true
    return cone.grad
end

# TODO parallelize
function _build_hess_real(H::Symmetric{T, Matrix{T}}, mat::Symmetric{T, Matrix{T}}, is_inv::Bool) where {T}
    side = size(mat, 1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        for i2 in 1:side, j2 in 1:i2
            if (i == j) && (i2 == j2)
                H.data[k2, k] = abs2(mat[i2, i])
            elseif (i != j) && (i2 != j2)
                fact = (is_inv ? inv(2) : 2)
                H.data[k2, k] = fact * (mat[i2, i] * mat[j, j2] + mat[j2, i] * mat[j, i2])
            else
                fact = (is_inv ? 1 : 2)
                H.data[k2, k] = fact * mat[i2, i] * mat[j, j2]
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

function _build_hess_complex(H::Symmetric{T, Matrix{T}}, mat::Hermitian{Complex{T}, Matrix{Complex{T}}}) where {T <: HypReal}
    side = size(mat, 1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        if i == j
            for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    H.data[k2, k] = abs2(mat[i2, i])
                    k2 += 1
                else
                    c = T(2) * mat[i, i2] * mat[j2, j]
                    H.data[k2, k] = real(c)
                    k2 += 1
                    H.data[k2, k] = -imag(c)
                    k2 += 1
                end
                if k2 > k
                    break
                end
            end
            k += 1
        else
            for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    c = T(2) * mat[i2, i] * mat[j, j2]
                    H.data[k2, k] = real(c)
                    H.data[k2, k + 1] = -imag(c)
                    k2 += 1
                else
                    b1 = mat[i2, i] * mat[j, j2]
                    b2 = mat[j2, i] * mat[j, i2]
                    c1 = T(2) * (b1 + b2)
                    H.data[k2, k] = real(c1)
                    H.data[k2, k + 1] = -imag(c1)
                    k2 += 1
                    c2 = T(2) * (b1 - b2)
                    H.data[k2, k] = imag(c2)
                    H.data[k2, k + 1] = real(c2)
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

function _build_inv_hess_complex(H::Symmetric{T, Matrix{T}}, mat::Hermitian{Complex{T}, Matrix{Complex{T}}}) where {T <: HypReal}
    side = size(mat, 1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        if i == j
            for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    H.data[k2, k] = abs2(mat[i2, i])
                    k2 += 1
                else
                    c = mat[i, i2] * mat[j2, j]
                    H.data[k2, k] = real(c)
                    k2 += 1
                    H.data[k2, k] = -imag(c)
                    k2 += 1
                end
                if k2 > k
                    break
                end
            end
            k += 1
        else
            for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    c = mat[i2, i] * mat[j, j2]
                    H.data[k2, k] = real(c)
                    H.data[k2, k + 1] = -imag(c)
                    k2 += 1
                else
                    b1 = mat[i2, i] * mat[j, j2]
                    b2 = mat[j2, i] * mat[j, i2]
                    c1 = T(0.5) * (b1 + b2)
                    H.data[k2, k] = real(c1)
                    H.data[k2, k + 1] = -imag(c1)
                    k2 += 1
                    c2 = T(0.5) * (b1 - b2)
                    H.data[k2, k] = imag(c2)
                    H.data[k2, k + 1] = real(c2)
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

function update_hess(cone::PosSemidef{T, T}) where {T <: HypReal}
    @assert cone.grad_updated
    _build_hess_real(cone.hess, cone.inv_mat, false)
    cone.hess_updated = true
    return cone.hess
end

function update_hess(cone::PosSemidef)
    @assert cone.grad_updated
    _build_hess_complex(cone.hess, cone.inv_mat)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::PosSemidef{T, T}) where {T <: HypReal}
    @assert cone.is_feas
    _build_hess_real(cone.inv_hess, cone.mat, true)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function update_inv_hess(cone::PosSemidef)
    @assert cone.is_feas
    _build_inv_hess_complex(cone.inv_hess, cone.mat)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_inv_hess_prod(cone::PosSemidef) = nothing

# function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidef)
#     if !cone.hess_updated
#         update_hess(cone)
#     end
#     return mul!(prod, cone.hess, arr)
# end
#
# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidef)
#     if !cone.inv_hess_updated
#         update_inv_hess(cone)
#     end
#     return mul!(prod, cone.inv_hess, arr)
# end

# function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidef{T, T}) where {T <: HypReal}
#     @assert cone.grad_updated
#     @inbounds for i in 1:size(arr, 2)
#         vec_to_mat_L!(cone.mat2.data, view(arr, :, i))
#         mul!(cone.mat3.data, cone.mat2, cone.inv_mat)
#         # hyp_symm!(T(1), cone.mat3.data, cone.mat2.data, cone.inv_mat.data)
#         mul!(cone.mat2.data, cone.inv_mat, cone.mat3.data)
#         # cone.mat2.data .= cone.inv_mat * cone.mat2 * cone.inv_mat
#         smat_to_svec!(view(prod, :, i), cone.mat2)
#     end
#     return prod
# end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidef)
    @assert cone.grad_updated
    @inbounds for i in 1:size(arr, 2)
        vec_to_mat_L!(cone.mat2.data, view(arr, :, i))
        # mul!(cone.mat3.data, cone.mat2, transpose(cone.inv_mat))
        # mul!(cone.mat2.data, transpose(cone.inv_mat), cone.mat3.data)
        cone.mat2.data .= cone.inv_mat * cone.mat2 * cone.inv_mat
        smat_to_svec!(view(prod, :, i), cone.mat2)
    end
    return prod
end

# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidef{T, T}) where {T <: HypReal}
#     @assert is_feas(cone)
#     @inbounds for i in 1:size(arr, 2)
#         svec_to_smat!(cone.mat2.data, view(arr, :, i))
#         # TODO make muls use Symmetric/Hermitian methods
#         mul!(cone.mat3.data, Symmetric(cone.mat2, :L), Symmetric(cone.mat, :L))
#         mul!(cone.mat2.data, Symmetric(cone.mat, :L), cone.mat3.data)
#         mat_L_to_vec!(view(prod, :, i), cone.mat2)
#     end
#     return prod
# end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidef{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
    @assert is_feas(cone)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat2.data, view(arr, :, i))
        # TODO make muls use Symmetric/Hermitian methods
        # mul!(cone.mat3.data, Hermitian(cone.mat2, :L), Hermitian(cone.mat, :L))
        # mul!(cone.mat2.data, Hermitian(cone.mat, :L), cone.mat3.data)
        cone.mat2.data .= cone.mat * cone.mat2 * cone.mat
        mat_L_to_vec!(view(prod, :, i), cone.mat2)
    end
    return prod
end
