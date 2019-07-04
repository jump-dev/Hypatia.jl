#=
Copyright 2019, Chris Coey and contributors
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.BlasFloat
using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra.HermOrSym
import LinearAlgebra.mul!
import LinearAlgebra.gemv!
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
    nrows::Int
    ncols::Int
    blocks::Vector
    rows::Vector{UnitRange{Int}}
    cols::Vector{UnitRange{Int}}
    function HypBlockMatrix{T}(blocks::Vector, rows::Vector{UnitRange{Int}}, cols::Vector{UnitRange{Int}}) where {T <: HypReal}
        @assert length(blocks) == length(rows) == length(cols)
        nrows = maximum(last, rows)
        ncols = maximum(last, cols)
        return new{T}(nrows, ncols, blocks, rows, cols)
    end
end

eltype(A::HypBlockMatrix{T}) where {T <: HypReal} = T

size(A::HypBlockMatrix) = (A.nrows, A.ncols)
size(A::HypBlockMatrix, d) = (d == 1 ? A.nrows : A.ncols)

adjoint(A::HypBlockMatrix{T}) where {T <: HypReal} = HypBlockMatrix{T}(adjoint.(A.blocks), A.cols, A.rows)

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

*(A::HypBlockMatrix{T}, x::Vector{T}) where {T <: HypReal} = mul!(similar(x, size(A, 1)), A, x)



# LAPACK helper functions for linear systems
# TODO try posvxx and sysvxx
# TODO allocate all data in a different struct for these two

# call LAPACK dposvx function (compare to dposv and dposvxx)
# performs equilibration and iterative refinement
function hyp_posvx!(
    X::Matrix{Float64},
    A::Matrix{Float64},
    B::Matrix{Float64},
    ferr,
    berr,
    work,
    iwork,
    AF,
    S,
    )
    n = size(A, 1)
    @assert n == size(A, 2) == size(B, 1)

    lda = stride(A, 2)
    nrhs = size(B, 2)
    ldb = stride(B, 2)
    rcond = Ref{Float64}()

    fact = 'E'
    uplo = 'U'
    equed = 'Y'

    info = Ref{BlasInt}()

    ccall((@blasfunc(dposvx_), Base.liblapack_name), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
        Ref{UInt8}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
        Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}),
        fact, uplo, n, nrhs, A, lda, AF, lda, equed, S, B,
        ldb, X, n, rcond, ferr, berr, work, iwork, info)

    if info[] != 0 && info[] != n+1
        # println("failure to solve linear system (posvx status $(info[]))")
        return false
    end
    return true
end

# call LAPACK dsysvx function (compare to dsysv and dsysvxx)
# performs equilibration and iterative refinement
function hyp_sysvx!(
    X::Matrix{Float64},
    A::Matrix{Float64},
    B::Matrix{Float64},
    ferr,
    berr,
    work,
    iwork,
    AF,
    ipiv,
    )
    n = size(A, 1)
    @assert n == size(A, 2) == size(B, 1)

    lda = stride(A, 2)
    nrhs = size(B, 2)
    ldb = stride(B, 2)
    rcond = Ref{Float64}()
    lwork = Ref{BlasInt}(5n)
    fact = 'N'
    uplo = 'U'

    info = Ref{BlasInt}()

    ccall((@blasfunc(dsysvx_), Base.liblapack_name), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
        Ptr{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
        Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
        Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        fact, uplo, n, nrhs, A, lda, AF, lda, ipiv, B,
        ldb, X, n, rcond, ferr, berr, work, lwork, iwork, info)

    if lwork[] > 5n
        println("in sysvx, lwork increased from $(5n) to $(lwork[])")
    end
    if info[] != 0 && info[] != n+1
        println("failure to solve linear system (sysvx status $(info[]))")
        return false
    end
    return true
end
