#=
Copyright 2019, Chris Coey and contributors
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.BlasFloat
import LinearAlgebra.HermOrSym
import LinearAlgebra.mul!
import LinearAlgebra.gemv!
import Base.adjoint
import Base.eltype
import Base.size
import Base.*
import Base.-


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


mutable struct HypBlockMatrix{T <: HypReal} # TODO undo
    nrows::Int
    ncols::Int
    blocks::Vector
    rows::Vector{UnitRange{Int}}
    cols::Vector{UnitRange{Int}}
end

function HypBlockMatrix{T}(blocks::Vector, rows::Vector{UnitRange{Int}}, cols::Vector{UnitRange{Int}}) where {T <: HypReal}
    @assert length(blocks) == length(rows) == length(cols)
    nrows = maximum(last, rows)
    ncols = maximum(last, cols)
    return HypBlockMatrix{T}(nrows, ncols, blocks, rows, cols)
end

eltype(A::HypBlockMatrix{T}) where {T <: HypReal} = T

size(A::HypBlockMatrix) = (A.nrows, A.ncols)
size(A::HypBlockMatrix, d) = (d == 1 ? A.nrows : A.ncols)

adjoint(A::HypBlockMatrix{T}) where {T <: HypReal} = HypBlockMatrix{T}(A.ncols, A.nrows, adjoint.(A.blocks), A.cols, A.rows)

# no method matching *(::Array{Float64,1}, ::SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true})

# A = Hypatia.HypBlockMatrix{Float64}(1, 5, [ones(Float64, 1, 3)], [1:1], [1:3])
# y = zeros(1)
# x = randn(5)
# mul!(y, A, x)

function mul!(y::AbstractVector{T}, A::HypBlockMatrix{T}, x::AbstractVector{T}) where {T <: HypReal}
    @assert size(x, 1) == A.ncols
    @assert size(y, 1) == A.nrows
    @assert size(x, 2) == size(y, 2)

    y .= zero(T)

    for (b, r, c) in zip(A.blocks, A.rows, A.cols)
        if isempty(r) || isempty(c)
            continue
        end
        # println()
        # if b isa UniformScaling
        #     println("I")
        # else
        #     println(size(b))
        # end
        # println(r, " , ", c)
        xk = view(x, c)
        yk = view(y, r)
        yk .+= b * xk # TODO need inplace mul+add
        # mul!(yk, b, xk, α = one(T), β = one(T))
    end

    return y
end

# function mul!(y::AbstractVector{T}, A::Adjoint{T, HypBlockMatrix{T}}, x::AbstractVector{T}) where {T <: HypReal}
#     @assert size(x, 1) == A.ncols
#     @assert size(y, 1) == A.nrows
#     @assert size(x, 2) == size(y, 2)
#
#     y .= zero(T)
#
#     for (b, r, c) in zip(A.blocks, A.rows, A.cols)
#         if isempty(r) || isempty(c)
#             continue
#         end
#         # println()
#         # if b isa UniformScaling
#         #     println("I")
#         # else
#         #     println(size(b))
#         # end
#         # println(r, " , ", c)
#         xk = view(x, r)
#         yk = view(y, c)
#         yk .+= b' * xk # TODO need inplace mul+add
#         # mul!(yk, b', xk)
#         # mul!(yk, b', xk, α = one(T), β = one(T))
#     end
#
#     return y
# end

*(A::HypBlockMatrix{T}, x::AbstractVector{T}) where {T <: HypReal} = mul!(similar(x, size(A, 1)), A, x)

-(A::HypBlockMatrix{T}) where {T <: HypReal} = HypBlockMatrix{T}(A.nrows, A.ncols, -A.blocks, A.rows, A.cols)
