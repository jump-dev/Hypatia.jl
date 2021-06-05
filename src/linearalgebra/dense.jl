#=
helpers for dense factorizations and linear solves
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.BlasFloat
import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.LAPACK.liblapack
import LinearAlgebra.copytri!

# helpers for in-place matrix inverses (updates upper triangle only in some cases)

# Cholesky, BlasFloat
function inv_fact!(
    mat::Matrix{R},
    fact::Cholesky{R, Matrix{R}},
    ) where {R <: BlasFloat}
    copyto!(mat, fact.factors)
    LAPACK.potri!(fact.uplo, mat)
    return mat
end

# Cholesky, generic
function inv_fact!(
    mat::Matrix{R},
    fact::Cholesky{R, Matrix{R}},
    ) where {R <: RealOrComplex{<:Real}}
    # this is how Julia computes the inverse, but it could be implemented better
    copyto!(mat, I)
    ldiv!(fact, mat)
    return mat
end

# BunchKaufman, BlasReal
function inv_fact!(
    mat::Matrix{T},
    fact::BunchKaufman{T, Matrix{T}},
    ) where {T <: BlasReal}
    @assert fact.rook
    copyto!(mat, fact.LD)
    LAPACK.sytri_rook!(fact.uplo, mat, fact.ipiv),
    return mat
end

# LU, BlasReal
function inv_fact!(
    mat::Matrix{T},
    fact::LU{T, Matrix{T}},
    ) where {T <: BlasReal}
    copyto!(mat, fact.factors)
    LAPACK.getri!(mat, fact.ipiv)
    return mat
end

# LU, generic
function inv_fact!(
    mat::Matrix{T},
    fact::LU{T, Matrix{T}},
    ) where {T <: Real}
    # this is how Julia computes the inverse, but it could be implemented better
    copyto!(mat, I)
    ldiv!(fact, mat)
    return mat
end

# helpers for updating symmetric/Hermitian eigendecomposition

update_eigen!(X::Matrix{<:BlasFloat}) = LAPACK.syev!('V', 'U', X)[1]

function update_eigen!(X::Matrix{<:RealOrComplex{<:Real}})
    F = eigen(Hermitian(X, :U))
    copyto!(X, F.vectors)
    return F.values
end

# helpers for symmetric outer product (upper triangle only)
# B = alpha * A' * A + beta * B

outer_prod!(
    A::Matrix{T},
    B::Matrix{T},
    alpha::Real,
    beta::Real,
    ) where {T <: LinearAlgebra.BlasReal} =
    BLAS.syrk!('U', 'T', alpha, A, beta, B)

outer_prod!(
    A::AbstractMatrix{Complex{T}},
    B::AbstractMatrix{Complex{T}},
    alpha::Real,
    beta::Real,
    ) where {T <: LinearAlgebra.BlasReal} =
    BLAS.herk!('U', 'C', alpha, A, beta, B)

outer_prod!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    alpha::Real,
    beta::Real,
    ) where {R <: RealOrComplex} =
    mul!(B, A', A, alpha, beta)


# ensure diagonal terms in square matrix are not too small
function increase_diag!(A::Matrix{T}) where {T <: Real}
    diag_pert = 1 + T(1e-5)
    diag_min = 1000 * eps(T)
    @inbounds for j in 1:size(A, 1)
        A[j, j] = diag_pert * max(A[j, j], diag_min)
    end
    return A
end

# helpers for spectral outer products

function spectral_outer!(
    mat::AbstractMatrix{T},
    vecs::Union{Matrix{T}, Adjoint{T, Matrix{T}}},
    diag::AbstractVector{T},
    temp::Matrix{T},
    ) where {T <: Real}
    mul!(temp, vecs, Diagonal(diag))
    mul!(mat, temp, vecs')
    return mat
end

function spectral_outer!(
    mat::AbstractMatrix{T},
    vecs::Union{Matrix{T}, Adjoint{T, Matrix{T}}},
    symm::Symmetric{T},
    temp::Matrix{T},
    ) where {T <: Real}
    mul!(temp, vecs, symm)
    mul!(mat, temp, vecs')
    return mat
end

#=
nonsymmetric square: LU
=#

function nonsymm_fact_copy!(
    mat2::Matrix{T},
    mat::Matrix{T},
    ) where {T <: Real}
    copyto!(mat2, mat)
    fact = lu!(mat2, check = false)

    if !issuccess(fact)
        copyto!(mat2, mat)
        increase_diag!(mat2)
        fact = lu!(mat2, check = false)
    end

    return fact
end

#=
symmetric indefinite: BunchKaufman (rook pivoting) and LU for generic fallback
NOTE if better fallback becomes available (eg dense LDL), use that
=#

symm_fact!(A::Symmetric{T, Matrix{T}}) where {T <: BlasReal} =
    bunchkaufman!(A, true, check = false)

symm_fact!(A::Symmetric{T, Matrix{T}}) where {T <: Real} =
    lu!(A, check = false)

function symm_fact_copy!(
    mat2::Symmetric{T, Matrix{T}},
    mat::Symmetric{T, Matrix{T}},
    ) where {T <: Real}
    copyto!(mat2, mat)
    fact = symm_fact!(mat2)

    if !issuccess(fact)
        copyto!(mat2, mat)
        increase_diag!(mat2.data)
        fact = symm_fact!(mat2)
    end

    return fact
end

#=
symmetric positive definite: unpivoted Cholesky
NOTE pivoted seems slower than BunchKaufman
=#

posdef_fact!(A::Symmetric{T, Matrix{T}}) where {T <: Real} =
    cholesky!(A, check = false)

function posdef_fact_copy!(
    mat2::Symmetric{T, Matrix{T}},
    mat::Symmetric{T, Matrix{T}},
    try_shift::Bool = true,
    ) where {T <: Real}
    copyto!(mat2, mat)
    fact = posdef_fact!(mat2)

    if !issuccess(fact)
        # try using symmetric factorization instead
        copyto!(mat2, mat)
        fact = symm_fact!(mat2)

        if try_shift && !issuccess(fact)
            copyto!(mat2, mat)
            increase_diag!(mat2.data)
            fact = symm_fact!(mat2)
        end
    end

    return fact
end
