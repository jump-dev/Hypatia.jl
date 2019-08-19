#=
Copyright 2019, Chris Coey and contributors
=#

import LinearAlgebra.mul!
import Base.adjoint
import Base.eltype
import Base.size
import Base.*
import Base.-

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
