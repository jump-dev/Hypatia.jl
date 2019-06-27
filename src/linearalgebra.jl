#=
Copyright 2019, Chris Coey and contributors
=#

using LinearAlgebra
import LinearAlgebra.BlasReal
import LinearAlgebra.BlasFloat
import LinearAlgebra.HermOrSym
import LinearAlgebra.mul!
import Base.adjoint
import Base.eltype
import Base.size
import Base.*


hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: BlasReal} = BLAS.syrk!('U', 'T', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{Complex{T}}, A::Matrix{Complex{T}}) where {T <: BlasReal} = BLAS.herk!('U', 'C', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: HypRealOrComplex{<:HypReal}} = mul!(U, A', A)


hyp_chol!(A::HermOrSym{T, Matrix{T}}) where {T <: BlasFloat} = cholesky!(A, Val(true), check = false)
hyp_chol!(A::HermOrSym{T, Matrix{T}}) where {T <: HypRealOrComplex{<:HypReal}} = cholesky!(A, check = false)


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


struct HypBlockMatrix{T <: HypReal}
    blocks::Vector{Any}
    rows::Vector{UnitRange{Int}}
    cols::Vector{UnitRange{Int}}
    function HypBlockMatrix{T}(blocks, rows, cols) where {T <: HypReal}
        @assert length(blocks) == length(rows) == length(cols)
        return new{T}(blocks, rows, cols)
    end
end

eltype(A::HypBlockMatrix{T}) where {T <: HypReal} = T

size(A::HypBlockMatrix) = (last(A.rows[end]), last(A.cols[end]))
function size(A::HypBlockMatrix, d)
    if d == 1
        return last(A.rows[end])
    elseif d == 2
        return last(A.cols[end])
    else
        error("HypBlockMatrix has two dimensions")
    end
end

adjoint(A::HypBlockMatrix{T}) where {T <: HypReal} = HypBlockMatrix{T}(adjoint.(A.blocks), A.cols, A.rows)

function mul!(y::AbstractVecOrMat{T}, A::HypBlockMatrix{T}, x::AbstractVecOrMat{T}) where {T <: HypReal}
    for (b, r, c) in zip(A.blocks, A.rows, A.cols)
        xk = view(x, c)
        yk = view(y, r)
        mul!(yk, b, xk)
    end
    return y
end

function mul!(y::AbstractVecOrMat{T}, A::Adjoint{T, HypBlockMatrix{T}}, x::AbstractVecOrMat{T}) where {T <: HypReal}
    for (b, r, c) in zip(A.blocks, A.rows, A.cols)
        xk = view(x, r)
        yk = view(y, c)
        mul!(yk, b', xk)
    end
    return y
end

*(A::HypBlockMatrix{T}, x::Vector{T}) where {T <: HypReal} = mul!(similar(x, size(A, 1)), A, x)
