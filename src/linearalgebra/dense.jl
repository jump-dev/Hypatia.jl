#=
Copyright 2019, Chris Coey and contributors

helpers for dense factorizations and linear solves
NOTE: factorization routines destroy the LHS matrix
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.LAPACK.liblapack
import LinearAlgebra.issuccess

#=
nonsymmetric: LU / GESVX
=#

abstract type DenseNonSymCache{T <: Real} end

reset_fact(cache::DenseNonSymCache) = (cache.is_factorized = false)

mutable struct GESVXNonSymCache{T <: BlasReal} <: DenseNonSymCache{T}
    is_factorized::Bool
    n
    lda
    AF
    ipiv
    equed
    rvec
    cvec
    rcond
    ferr
    berr
    work
    iwork
    info
    GESVXNonSymCache{T}() where {T <: BlasReal} = new{T}()
end

function load_dense_matrix(cache::GESVXNonSymCache{T}, A::Matrix{T}) where {T <: BlasReal}
    cache.is_factorized = false
    LinearAlgebra.chkstride1(A)
    n = cache.n = LinearAlgebra.checksquare(A)
    cache.lda = stride(A, 2)
    cache.AF = Matrix{T}(undef, n, n)
    cache.ipiv = Vector{Int}(undef, n)
    cache.equed = Ref{UInt8}('E')
    cache.rvec = Vector{T}(undef, n)
    cache.cvec = Vector{T}(undef, n)
    cache.rcond = Ref{T}()
    cache.ferr = Vector{T}(undef, 1) # NOTE ferr and berr are resized if too small
    cache.berr = Vector{T}(undef, 1)
    cache.work = Vector{T}(undef, 4 * n)
    cache.iwork = Vector{Int}(undef, n)
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK function
for (gesvx, elty) in [(:dgesvx_, :Float64), (:sgesvx_, :Float32)]
    @eval begin
        function solve_dense_system(cache::GESVXNonSymCache{$elty}, X::StridedVecOrMat{$elty}, A::Matrix{$elty}, B::StridedVecOrMat{$elty})
            nrhs = size(B, 2)
            @assert size(X, 2) == nrhs
            if length(cache.ferr) < nrhs
                resize!(cache.rvec, nrhs)
                resize!(cache.cvec, nrhs)
            end

            if cache.is_factorized
                cache.is_factorized = true
                do_fact = 'F'
            else
                do_fact = 'E'
            end

            # call dgesvx( fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info )
            ccall((@blasfunc($gesvx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ref{UInt8}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                Ptr{BlasInt}, Ptr{BlasInt}),
                do_fact, 'N', cache.n, nrhs,
                A, cache.lda, cache.AF, cache.n,
                cache.ipiv, cache.equed, cache.rvec, cache.cvec,
                B, stride(B, 2), X, cache.n,
                cache.rcond, cache.ferr, cache.berr, cache.work,
                cache.iwork, cache.info)

            if cache.info[] < 0
                throw(ArgumentError("invalid argument #$(-cache.info[]) to LAPACK call"))
            elseif 0 < cache.info[] <= cache.n
                @warn("factorization failed: #$(cache.info[])")
                return false
            elseif cache.info[] > cache.n
                @warn("condition number is small: $(cache.rcond[])")
            end

            return true
        end
    end
end

mutable struct LUNonSymCache{T <: Real} <: DenseNonSymCache{T}
    is_factorized::Bool
    fact
    LUNonSymCache{T}() where {T <: Real} = new{T}()
end

function load_dense_matrix(cache::LUNonSymCache{T}, A::AbstractMatrix{T}) where {T <: Real}
    cache.is_factorized = false
    return cache
end


function solve_dense_system(cache::LUNonSymCache{T}, X::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Real}
    if !cache.is_factorized
        cache.is_factorized = true
        cache.fact = lu!(A, check = false)
        if !issuccess(cache.fact)
            @warn("numerical failure: LU factorization failed")
            return false
        end
    end

    ldiv!(X, cache.fact, B)

    return true
end

# default to GESVXNonSymCache for BlasReals, otherwise generic LUNonSymCache
DenseNonSymCache{T}() where {T <: BlasReal} = GESVXNonSymCache{T}()
DenseNonSymCache{T}() where {T <: Real} = LUNonSymCache{T}()







LinearAlgebra.issuccess(F::Union{Cholesky, CholeskyPivoted}) = isposdef(F)

# ensure diagonal terms in symm/herm that should be PSD are not too small
function set_min_diag!(A::Matrix{<:RealOrComplex{T}}, tol::T) where {T <: Real}
    if tol <= 0
        return A
    end
    @inbounds for j in 1:size(A, 1)
        Ajj = A[j, j]
        if Ajj < tol
            A[j, j] = tol
        end
    end
    return A
end

# TODO equilibration (radix) functions DPOEQUB/DSYEQUB/DGEEQUB

#=
UNpivoted Cholesky factorization (like POTRF) and inverse (like POTRI)
=#

mutable struct HypCholCache{R <: Real, T <: RealOrComplex{R}}
    uplo
    n
    lda
    info
    HypCholCache{R, T}() where {T <: RealOrComplex{R}} where {R <: Real} = new{R, T}()
end

function HypCholCache(uplo::Char, A::Matrix{T}) where {T <: RealOrComplex{R}} where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    Base.require_one_based_indexing(A)
    c = HypCholCache{R, T}()
    c.uplo = uplo
    c.n = LinearAlgebra.checksquare(A)
    c.lda = max(1, stride(A, 2))
    c.info = Ref{BlasInt}()
    return c
end

for (potrf, potri, elty, rtyp) in (
    (:dpotrf_, :dpotri_, :Float64, :Float64),
    (:spotrf_, :spotri_, :Float32, :Float32),
    (:zpotrf_, :zpotri_, :ComplexF64, :Float64),
    (:cpotrf_, :cpotri_, :ComplexF32, :Float32),
    )
    @eval begin
        function hyp_chol!(c::HypCholCache{$rtyp, $elty}, A::Matrix{$elty})
            # call dposvx( fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info )
            ccall((@blasfunc($potrf), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                c.uplo, c.n, A, c.lda, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            end

            return Cholesky{$elty, typeof(A)}(A, c.uplo, c.info[])
        end
    end

    @eval begin
        function hyp_chol_inv!(c::HypCholCache{$rtyp, $elty}, fact_A::Cholesky{$elty, <:Matrix{$elty}})
            # call dpotri( uplo, n, a, lda, info )
            ccall((@blasfunc($potri), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                c.uplo, c.n, fact_A.factors, c.lda, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif c.info[] > 0
                @warn("inverse from Cholesky failed: $(c.info[])")
            end

            return fact_A.factors
        end
    end
end

function HypCholCache(uplo::Char, A::AbstractMatrix{T}) where {T <: RealOrComplex{R}} where {R <: Real}
    c = HypCholCache{R, T}()
    c.uplo = uplo
    return c
end

hyp_chol!(c::HypCholCache{R, T}, A::AbstractMatrix{T}) where {T <: RealOrComplex{R}} where {R <: Real} = cholesky!(Hermitian(A, Symbol(c.uplo)), check = false)

hyp_chol_inv!(c::HypCholCache{R, T}, fact_A::Cholesky{T, <:AbstractMatrix{T}}) where {T <: RealOrComplex{R}} where {R <: Real} = inv(fact_A)

#=
pivoted BunchKaufman factorization (like SYTRF_ROOK)
only use for PSD matrices, since generic fallback is to Cholesky
=#

mutable struct HypBKCache{R <: Real}
    uplo
    n
    lda
    ipiv
    work
    lwork
    info
    HypBKCache{R}() where {R <: Real} = new{R}()
end

function HypBKCache(uplo::Char, A::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypBKCache{R}()
    c.uplo = uplo
    c.n = LinearAlgebra.checksquare(A)
    c.lda = max(1, stride(A, 2))
    c.ipiv = similar(A, BlasInt, c.n)
    c.work = Vector{R}(undef, 1)
    c.lwork = BlasInt(-1)
    c.info = Ref{BlasInt}()
    c.lwork = hyp_bk!(c, A) # gets the optimal size from an empty call to LAPACK
    c.work = Vector{R}(undef, c.lwork)
    return c
end

for (sytrf, elty, rtyp) in (
    (:dsytrf_rook_, :Float64, :Float64),
    (:ssytrf_rook_, :Float32, :Float32),
    )
    @eval begin
        function hyp_bk!(c::HypBKCache{$elty}, A::AbstractMatrix{$elty})
            # call dsytrf_rook( uplo, n, a, lda, ipiv, work, lwork, info )
            ccall((@blasfunc($sytrf), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                c.uplo, c.n, A, c.lda,
                c.ipiv, c.work, c.lwork, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif c.lwork == -1
                return BlasInt(real(c.work[1])) # returns the optimal size
            elseif c.info[] == c.n
                @warn("condition number is small: $(c.rcond[])")
            end

            return BunchKaufman{$elty, typeof(A)}(A, c.ipiv, c.uplo, true, true, c.info[])
        end
    end
end

function HypBKCache(uplo::Char, A::AbstractMatrix{R}) where {R <: Real}
    c = HypBKCache{R}()
    c.uplo = uplo
    return c
end

# fall back to generic Cholesky
hyp_bk!(c::HypBKCache{R}, A::AbstractMatrix{R}) where {R <: Real} = cholesky!(Symmetric(A, Symbol(c.uplo)), check = false)

#=
BunchKaufman solve (like SYSVX)
TODO try Aasen's version (http://www.netlib.org/lapack/lawnspdf/lawn294.pdf) and others
=#

mutable struct HypBKSolveCache{R <: Real}
    uplo
    F
    n
    rcond
    AF
    ipiv
    lwork
    ferr
    berr
    work
    iwork
    info
    HypBKSolveCache{R}() where {R <: Real} = new{R}()
end

function HypBKSolveCache(uplo::Char, A::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypBKSolveCache{R}()
    c.uplo = uplo
    c.n = LinearAlgebra.checksquare(A)
    c.rcond = Ref{R}()
    c.lwork = Ref{BlasInt}(5 * c.n)
    c.ferr = Vector{R}(undef, 0)
    c.berr = Vector{R}(undef, 0)
    c.work = Vector{R}(undef, 5 * c.n)
    c.iwork = Vector{Int}(undef, c.n)
    c.AF = Matrix{R}(undef, c.n, c.n)
    c.ipiv = Vector{Int}(undef, c.n)
    c.info = Ref{BlasInt}()
    return c
end

for (sysvx, elty, rtyp) in (
    (:dsysvx_, :Float64, :Float64),
    (:ssysvx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_bk_solve!(c::HypBKSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}; factorize::Bool = true)
            nrhs = size(B, 2)
            if length(c.ferr) < nrhs
                resize!(c.ferr, nrhs)
                resize!(c.berr, nrhs)
            end

            # call dsysvx( fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, lwork, iwork, info )
            ccall((@blasfunc($sysvx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                (factorize ? 'N' : 'F'), c.uplo, c.n, nrhs,
                A, stride(A, 2), c.AF, c.n,
                c.ipiv, B, stride(B, 2), X,
                stride(X, 2), c.rcond, c.ferr, c.berr,
                c.work, c.lwork, c.iwork, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif 0 < c.info[] <= c.n
                @warn("factorization failed: #$(c.info[])")
                return false
            elseif c.info[] > c.n
                @warn("condition number is small: $(c.rcond[])")
            end
            return true
        end
    end
end

function HypBKSolveCache(uplo::Char, A::AbstractMatrix{R}) where {R <: Real}
    c = HypBKSolveCache{R}()
    c.uplo = uplo
    return c
end

# fall back to generic LU solve
function hyp_bk_solve!(c::HypBKSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}; factorize::Bool = true) where {R <: Real}
    if factorize
        c.F = lu!(Symmetric(A, Symbol(c.uplo)), check = false)
        if !issuccess(c.F)
            return false
        end
    end
    ldiv!(X, c.F, B)
    return true
end

#=
Cholesky solve (like POSVX)
=#

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

function HypCholSolveCache(uplo::Char, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypCholSolveCache{R}()
    c.uplo = uplo
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

for (posvx, elty, rtyp) in (
    (:dposvx_, :Float64, :Float64),
    (:sposvx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_chol_solve!(c::HypCholSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            # call dposvx( fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info )
            ccall((@blasfunc($posvx), liblapack), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ref{UInt8}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                'E', c.uplo, c.n, c.nrhs,
                A, c.lda, c.AF, c.ldaf,
                'Y', c.S, B, c.ldb,
                X, c.n, c.rcond, c.ferr,
                c.berr, c.work, c.iwork, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif 0 < c.info[] <= c.n
                @warn("factorization failed: #$(c.info[])")
                return false
            elseif c.info[] > c.n
                @warn("condition number is small: $(c.rcond[])")
            end
            return true
        end
    end
end

function HypCholSolveCache(uplo::Char, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    c = HypCholSolveCache{R}()
    c.uplo = uplo
    return c
end

function hyp_chol_solve!(c::HypCholSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    F = cholesky!(Symmetric(A, Symbol(c.uplo)), check = false)
    if !isposdef(F)
        return false
    end
    ldiv!(X, F, B)
    return true
end
