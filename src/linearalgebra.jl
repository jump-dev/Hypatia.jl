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

# cache for LAPACK pivoted cholesky (like PSTRF)
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

function HypCholCache(use_upper::Bool, A::AbstractMatrix{T}; tol = zero(R)) where {T <: RealOrComplex{R}} where {R <: BlasReal}
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

for (pstrf, elty, rtyp) in (
    (:dpstrf_, :Float64, :Float64),
    (:spstrf_, :Float32, :Float32),
    (:zpstrf_, :ComplexF64, :Float64),
    (:cpstrf_, :ComplexF32, :Float32),
    )
    @eval begin
        function hyp_chol!(c::HypCholCache{$rtyp, $elty}, A::AbstractMatrix{$elty})
            ccall((@blasfunc($pstrf), liblapack), Cvoid, (
                Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                Ptr{BlasInt}, Ref{$rtyp}, Ptr{$rtyp}, Ptr{BlasInt}
                ), c.uplo, c.n, A, c.lda, c.piv, c.rank, c.tol, c.work, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            end

            return CholeskyPivoted{$elty, typeof(A)}(A, c.uplo, c.piv, c.rank[1], c.tol, c.info[])
        end
    end
end

function HypCholCache(use_upper::Bool, A::AbstractMatrix{T}; tol = zero(R)) where {T <: RealOrComplex{R}} where {R <: Real}
    c = HypCholCache{R, T}()
    c.uplo = (use_upper ? :U : :L)
    return c
end

hyp_chol!(c::HypCholCache{R, T}, A::AbstractMatrix{T}) where {T <: RealOrComplex{R}} where {R <: Real} = cholesky!(Hermitian(A, c.uplo), check = false)

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



# cache for LAPACK cholesky with linear solve (like POSVX)
mutable struct HypCholSolveCache{R <: Real}
    uplo
    n
    lda
    ldaf
    nrhs
    ldb
    rcond
    lwork
    ferr
    berr
    work
    iwork
    AF
    ipiv
    S
    info
    HypCholSolveCache{R}() where {R <: Real} = new{R}()
end

function HypCholSolveCache(use_upper::Bool, X::AbstractMatrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypCholSolveCache{R}()
    c.uplo = (use_upper ? 'U' : 'L')
    c.n = LinearAlgebra.checksquare(A)
    @assert c.n == size(X, 1) == size(B, 1)
    @assert size(X, 2) == size(B, 2)
    c.lda = stride(A, 2)
    c.ldaf = c.n
    c.nrhs = size(B, 2)
    c.ldb = stride(B, 2)
    c.rcond = Ref{R}()
    c.lwork = Ref{BlasInt}(5 * c.n)
    c.ferr = Vector{R}(undef, c.nrhs)
    c.berr = Vector{R}(undef, c.nrhs)
    c.work = Vector{R}(undef, 5 * c.n)
    c.iwork = Vector{Int}(undef, c.n)
    c.AF = Matrix{R}(undef, c.n, c.n)
    c.ipiv = Vector{Int}(undef, c.n)
    c.S = Vector{R}(undef, c.n)
    c.info = Ref{BlasInt}()
    return c
end

# access LAPACK functions
for (posvx, elty, rtyp) in (
    (:dposvx_, :Float64, :Float64),
    (:sposvx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_chol_solve!(c::HypCholSolveCache{$elty}, X::AbstractMatrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            ccall((@blasfunc($posvx), liblapack), Cvoid, (
                Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ref{UInt8}, Ptr{$elty},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt},
                ), 'E', c.uplo, c.n, c.nrhs, A, c.lda, c.AF, c.ldaf, 'Y', c.S, B,
                c.ldb, X, c.n, c.rcond, c.ferr, c.berr, c.work, c.iwork, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif 0 < c.info[] <= c.n
                println("factorization failed")
            elseif c.info[] == c.n
                println("RCOND is small")
            end

            return X
        end
    end
end

function HypCholSolveCache(use_upper::Bool, X::AbstractMatrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    c = HypCholSolveCache{R}()
    c.uplo = (use_upper ? :U : :L)
    return c
end

function hyp_chol_solve!(c::HypCholSolveCache{R}, X::AbstractMatrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    F = cholesky!(Symmetric(A, c.uplo), check = false)
    if !isposdef(F)
        error("Cholesky factorization failed")
    end
    ldiv!(X, F, B)
    return X
end



# cache for LAPACK Bunch-Kaufman with linear solve (like SYSVX)
mutable struct HypBKSolveCache{R <: BlasReal}
    uplo
    n
    lda
    ldaf
    nrhs
    ldb
    rcond
    lwork
    ferr
    berr
    work
    iwork
    AF
    ipiv
    info
    HypBKSolveCache{R}() where {R <: BlasReal} = new{R}()
end

function HypBKSolveCache(use_upper::Bool, X::AbstractMatrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypBKSolveCache{R}()
    c.uplo = (use_upper ? 'U' : 'L')
    c.n = LinearAlgebra.checksquare(A)
    @assert c.n == size(X, 1) == size(B, 1)
    @assert size(X, 2) == size(B, 2)
    c.lda = stride(A, 2)
    c.ldaf = c.n
    c.nrhs = size(B, 2)
    c.ldb = stride(B, 2)
    c.rcond = Ref{R}()
    c.lwork = Ref{BlasInt}(5 * c.n)
    c.ferr = Vector{R}(undef, c.nrhs)
    c.berr = Vector{R}(undef, c.nrhs)
    c.work = Vector{R}(undef, 5 * c.n)
    c.iwork = Vector{Int}(undef, c.n)
    c.AF = Matrix{R}(undef, c.n, c.n)
    c.ipiv = Vector{Int}(undef, c.n)
    c.info = Ref{BlasInt}()
    return c
end

# access LAPACK functions
for (sysvx, elty, rtyp) in (
    (:dsysvx_, :Float64, :Float64),
    (:ssysvx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_bk_solve!(c::HypBKSolveCache{$elty}, X::AbstractMatrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            # ccall((@blasfunc($posvx), liblapack), Cvoid, (
            #     Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            #     Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
            #     Ref{UInt8}, Ptr{$elty},
            #     Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
            #     Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
            #     Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt},
            #     ), 'E', c.uplo, c.n, c.nrhs, A, c.lda, c.AF, c.ldaf, 'Y', c.S, B,
            #     c.ldb, X, c.n, c.rcond, c.ferr, c.berr, c.work, c.iwork, c.info)
            ccall((@blasfunc(dsysvx_), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
                Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                'N', c.uplo, c.n, c.nrhs, A, c.lda, c.AF, c.ldaf, c.ipiv, B,
                c.ldb, X, c.n, c.rcond, c.ferr, c.berr, c.work, c.lwork, c.iwork, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif 0 < c.info[] <= c.n
                println("factorization failed")
            elseif c.info[] == c.n
                println("RCOND is small")
            end

            return X
        end
    end
end
