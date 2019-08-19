#=
Copyright 2019, Chris Coey and contributors
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.BlasFloat
import LinearAlgebra.HermOrSym

hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: BlasReal} = BLAS.syrk!('U', 'T', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{Complex{T}}, A::Matrix{Complex{T}}) where {T <: BlasReal} = BLAS.herk!('U', 'C', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: RealOrComplex{<:Real}} = mul!(U, A', A)

hyp_AAt!(U::Matrix{T}, A::Matrix{T}) where {T <: BlasReal} = BLAS.syrk!('U', 'N', one(T), A, zero(T), U)
hyp_AAt!(U::Matrix{T}, A::Matrix{T}) where {T <: RealOrComplex{<:Real}} = mul!(U, A, A')


import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.LAPACK.liblapack

# cache for LAPACK cholesky
struct HypCholCache{R <: BlasReal, T <: RealOrComplex{R}}
    uplo
    n
    lda
    piv
    rank
    work
    info
    tol

    function HypCholCache(use_upper::Bool, A::Matrix{T}; tol = zero(R)) where {T <: RealOrComplex{R}} where {R <: BlasReal}
        LinearAlgebra.chkstride1(A)
        uplo = (use_upper ? 'U' : 'L')
        n = LinearAlgebra.checksquare(A)
        lda = max(1, stride(A, 2))
        piv = similar(A, BlasInt, n)
        rank = Vector{BlasInt}(undef, 1)
        work = Vector{R}(undef, 2 * n)
        info = Ref{BlasInt}()
        return new{R, T}(uplo, n, lda, piv, rank, work, info, tol)
    end
end

for (potri, pstrf, elty, rtyp) in (
    (:dpotri_, :dpstrf_, :Float64, :Float64),
    (:spotri_, :spstrf_, :Float32, :Float32),
    (:zpotri_, :zpstrf_, :ComplexF64, :Float64),
    (:cpotri_, :cpstrf_, :ComplexF32, :Float32),
    )

    # TODO implement inverse for pivoted

    @eval begin
        function hyp_chol!(c::HypCholCache{$rtyp, $elty}, A::Matrix{$elty})
            ccall((@blasfunc($pstrf), liblapack), Cvoid, (
                Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                Ptr{BlasInt}, Ref{$rtyp}, Ptr{$rtyp}, Ptr{BlasInt}
                ), c.uplo, c.n, A, c.lda, c.piv, c.rank, c.tol, c.work, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            end

            C = CholeskyPivoted{$elty, Matrix{$elty}}(A, c.uplo, c.piv, c.rank[1], c.tol, c.info[])

            return C
        end
    end
end


# function pstrf!(uplo::AbstractChar, A::AbstractMatrix{$elty}, tol::Real)
#     chkstride1(A)
#     n = checksquare(A)
#     chkuplo(uplo)
#     piv  = similar(A, BlasInt, n)
#     rank = Vector{BlasInt}(undef, 1)
#     work = Vector{$rtyp}(undef, 2n)
#     info = Ref{BlasInt}()
#     ccall((@blasfunc($pstrf), liblapack), Cvoid, (
#         Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
#         Ptr{BlasInt}, Ref{$rtyp}, Ptr{$rtyp}, Ptr{BlasInt}
#         ),
#         uplo, n, A, max(1,stride(A,2)), piv, rank, tol, work, info)
#     if info[] < 0
#       throw(ArgumentError("invalid argument #$(-ret) to LAPACK call"))
#     end
#
#     A, piv, rank[1], info[] #Stored in CholeskyPivoted
# end




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
