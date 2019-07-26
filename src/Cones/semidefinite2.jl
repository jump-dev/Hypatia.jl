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
- eliminate redundant svec_to_smat calls

- complex version of cone
- remove "remove me"- references to it need to be replaced from scalings to no-ops, just using as a placeholder as to not rewrite functions too soon
=#

mutable struct PosSemidefUnsc{T <: HypReal, R <: HypRealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    point::AbstractVector{T}
    is_complex::Bool

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    two::T
    remove_me::T
    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    inv_mat::Matrix{R}
    fact_mat

    function PosSemidefUnsc{T, R}(dim::Int, is_dual::Bool) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
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

PosSemidefUnsc{T, R}(dim::Int) where {R <: HypRealOrComplex{T}} where {T <: HypReal} = PosSemidefUnsc{T, R}(dim, false)

reset_data(cone::PosSemidefUnsc) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function setup_data(cone::PosSemidefUnsc{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
    reset_data(cone)
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.mat = Matrix{R}(undef, cone.side, cone.side)
    cone.mat2 = similar(cone.mat)
    cone.mat3 = similar(cone.mat)
    cone.two = T(2)
    cone.remove_me = one(T) # TODO remove this and references, not needed
    return
end

get_nu(cone::PosSemidefUnsc) = cone.side

function set_initial_point(arr::AbstractVector, cone::PosSemidefUnsc)
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
function update_feas(cone::PosSemidefUnsc)
    @assert !cone.feas_updated
    svec_to_smat!(cone.mat, cone.point, cone.remove_me)
    copyto!(cone.mat2, cone.mat)
    cone.fact_mat = hyp_chol!(Hermitian(cone.mat2))
    cone.is_feas = isposdef(cone.fact_mat)
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidefUnsc)
    @assert cone.is_feas
    cone.inv_mat = inv(cone.fact_mat) # TODO eliminate allocs
    smat_to_svec!(cone.grad, transpose(cone.inv_mat), cone.two)
    cone.grad .*= -1
    cone.grad_updated = true
    return cone.grad
end

# TODO parallelize
function _build_hess_real2(H::Matrix, mat::Matrix, scaling::T, is_inv::Bool) where {T <:HypReal}
    side = size(mat, 1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        for i2 in 1:side, j2 in 1:i2
            if (i == j) && (i2 == j2)
                H[k2, k] = abs2(mat[i2, i])
            elseif (i != j) && (i2 != j2)
                H[k2, k] = scaling * (mat[i2, i] * mat[j, j2] + mat[j2, i] * mat[j, i2])
            else
                fact = (is_inv ? one(T) : scaling)
                H[k2, k] = fact * mat[i2, i] * mat[j, j2]
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

function update_hess(cone::PosSemidefUnsc{T, T}) where {T <:HypReal}
    @assert cone.grad_updated
    _build_hess_real2(cone.hess.data, cone.inv_mat, cone.two, false)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::PosSemidefUnsc{T, T}) where {T <:HypReal}
    @assert cone.is_feas
    _build_hess_real2(cone.inv_hess.data, cone.mat, inv(cone.two), true)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_inv_hess_prod(cone::PosSemidefUnsc) = nothing

# TODO complex case as an operator
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefUnsc{T}) where {T <: HypReal}
    @assert cone.grad_updated
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat2, view(arr, :, i), one(T))
        # TODO make muls use Symmetric/Hermitian methods
        mul!(cone.mat3, Symmetric(cone.mat2, :L), Symmetric(cone.inv_mat, :L))
        mul!(cone.mat2, Symmetric(cone.inv_mat, :L), cone.mat3)
        smat_to_svec!(view(prod, :, i), cone.mat2, cone.two)
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefUnsc{T}) where {T <: HypReal}
    @assert is_feas(cone)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat2, view(arr, :, i), T(0.5))
        # TODO make muls use Symmetric/Hermitian methods
        mul!(cone.mat3, Symmetric(cone.mat2, :L), Symmetric(cone.mat, :L))
        mul!(cone.mat2, cone.mat, cone.mat3)
        smat_to_svec!(view(prod, :, i), cone.mat2, one(T))
    end
    return prod
end
