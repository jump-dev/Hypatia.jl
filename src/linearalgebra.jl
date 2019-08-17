#=
Copyright 2019, Chris Coey and contributors
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.BlasFloat
import LinearAlgebra.HermOrSym
import LinearAlgebra.mul!
import Base.adjoint
import Base.eltype
import Base.size
import Base.*
import Base.-

hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: BlasReal} = BLAS.syrk!('U', 'T', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{Complex{T}}, A::Matrix{Complex{T}}) where {T <: BlasReal} = BLAS.herk!('U', 'C', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: RealOrComplex{<:Real}} = mul!(U, A', A)

hyp_AAt!(U::Matrix{T}, A::Matrix{T}) where {T <: BlasReal} = BLAS.syrk!('U', 'N', one(T), A, zero(T), U)
hyp_AAt!(U::Matrix{Complex{T}}, A::Matrix{Complex{T}}) where {T <: BlasReal} = BLAS.herk!('U', 'N', one(T), A, zero(T), U)
hyp_AAt!(U::Matrix{T}, A::Matrix{T}) where {T <: RealOrComplex{<:Real}} = mul!(U, A, A')

# hyp_syrk!(trans::Char, A::Matrix{T}, U::Matrix{T}) where {T <: BlasReal} = BLAS.syrk!('U', trans, one(T), A, zero(T), U)
# hyp_syrk!(trans::Char, A::Matrix{Complex{T}}, U::Matrix{Complex{T}}) where {T <: BlasReal} = BLAS.herk!('U', trans, one(T), A, zero(T), U)
# hyp_syrk!(trans::Char, A::Matrix{T}, U::Matrix{T}) where {T <: HypRealOrComplex{<:HypReal}} = t == 'T' && mul!(U, A', A) || mul!(U, A, A')

hyp_chol!(A::HermOrSym{T, Matrix{T}}) where {T <: BlasFloat} = cholesky!(A, Val(true), check = false)
hyp_chol!(A::HermOrSym{T, Matrix{T}}) where {T <: RealOrComplex{<:Real}} = cholesky!(A, check = false)

function hyp_ldiv_chol_L!(B::Matrix, F::CholeskyPivoted, A::AbstractMatrix)
    copyto!(B, view(A, F.p, :))
    ldiv!(LowerTriangular(F.L), B)
    return B
end
function hyp_ldiv_chol_L!(B::Matrix, F::Cholesky, A::AbstractMatrix)
    copyto!(B, A)
    ldiv!(LowerTriangular(F.L), B)
    return B
end

struct BlockMatrix{T <: Real}
    nrows::Int
    ncols::Int
    blocks::Vector
    rows::Vector{UnitRange{Int}}
    cols::Vector{UnitRange{Int}}

    function BlockMatrix{T}(nrows::Int, ncols::Int, blocks::Vector, rows::Vector{UnitRange{Int}}, cols::Vector{UnitRange{Int}}) where {T <: Real}
        @assert length(blocks) == length(rows) == length(cols)
        return new{T}(nrows, ncols, blocks, rows, cols)
    end
end

eltype(A::BlockMatrix{T}) where {T <: Real} = T

size(A::BlockMatrix) = (A.nrows, A.ncols)
size(A::BlockMatrix, d) = (d == 1 ? A.nrows : A.ncols)

adjoint(A::BlockMatrix{T}) where {T <: Real} = BlockMatrix{T}(A.ncols, A.nrows, adjoint.(A.blocks), A.cols, A.rows)

# TODO try to speed up by using better logic for alpha and beta (see Julia's 5-arg mul! code)
# TODO check that this eliminates allocs when using IterativeSolvers methods, and that it is as fast as possible
function mul!(y::AbstractVector{T}, A::BlockMatrix{T}, x::AbstractVector{T}, alpha::Number, beta::Number) where {T <: Real}
    @assert size(x, 1) == A.ncols
    @assert size(y, 1) == A.nrows
    @assert size(x, 2) == size(y, 2)

    @. y *= beta
    for (b, r, c) in zip(A.blocks, A.rows, A.cols)
        if isempty(r) || isempty(c)
            continue
        end
        xk = view(x, c)
        yk = view(y, r)
        mul!(yk, b, xk, alpha, true)
    end
    return y
end

*(A::BlockMatrix{T}, x::AbstractVector{T}) where {T <: Real} = mul!(similar(x, size(A, 1)), A, x)

-(A::BlockMatrix{T}) where {T <: Real} = BlockMatrix{T}(A.nrows, A.ncols, -A.blocks, A.rows, A.cols)
