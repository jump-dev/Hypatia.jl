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
mutable struct HypCholCache{R <: Real, T <: RealOrComplex{R}}
    uplo
    n
    lda
    piv
    rank
    work
    info
    tol
    HypCholCache{R, T}() where {T <: RealOrComplex{R}} where {R <: Real} = new{R, T}()
end

function HypCholCache(use_upper::Bool, A::Matrix{T}; tol = zero(R)) where {T <: RealOrComplex{R}} where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypCholCache{R, T}()
    c.uplo = (use_upper ? 'U' : 'L')
    c.n = LinearAlgebra.checksquare(A)
    c.lda = max(1, stride(A, 2))
    c.piv = similar(A, BlasInt, c.n)
    c.rank = Vector{BlasInt}(undef, 1)
    c.work = Vector{R}(undef, 2 * c.n)
    c.info = Ref{BlasInt}()
    c.tol = tol
    return c
end

function HypCholCache(use_upper::Bool, A::Matrix{T}; tol = zero(R)) where {T <: RealOrComplex{R}} where {R <: Real}
    c = HypCholCache{R, T}()
    c.uplo = (use_upper ? :U : :L)
    return c
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

hyp_chol!(c::HypCholCache{R, T}, A::Matrix{T}) where {T <: RealOrComplex{R}} where {R <: Real} = cholesky!(Hermitian(A, c.uplo), check = false)

# TODO delete later
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
