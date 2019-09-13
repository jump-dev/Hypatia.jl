#=
Copyright 2019, Chris Coey and contributors
=#

#=
helpers for outer products
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.HermOrSym

hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: BlasReal} = BLAS.syrk!('U', 'T', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{Complex{T}}, A::Matrix{Complex{T}}) where {T <: BlasReal} = BLAS.herk!('U', 'C', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: RealOrComplex{<:Real}} = mul!(U, A', A)

hyp_AAt!(U::Matrix{T}, A::Matrix{T}) where {T <: BlasReal} = BLAS.syrk!('U', 'N', one(T), A, zero(T), U)
hyp_AAt!(U::Matrix{T}, A::Matrix{T}) where {T <: RealOrComplex{<:Real}} = mul!(U, A, A')

#=
helpers for factorizations and linear solves
TODO cleanup by
- removing rtyp when not used
- removing unnecessary fields in caches
=#

import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.LAPACK.liblapack
import LinearAlgebra.issuccess

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

# TODO equilibration (radix) functions for cholesky and bk and LU - use DPOEQUB/DSYEQUB/DGEEQUB

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

function HypCholCache(uplo::Char, A::StridedMatrix{T}) where {T <: RealOrComplex{R}} where {R <: BlasReal}
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
        function hyp_chol!(c::HypCholCache{$rtyp, $elty}, A::StridedMatrix{$elty})
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
        function hyp_chol_inv!(c::HypCholCache{$rtyp, $elty}, fact_A::Cholesky{$elty, <:StridedMatrix{$elty}})
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
LU solve (like GESVX)
=#

mutable struct HypLUSolveCache{R <: Real}
    n
    nrhs
    lda
    AF
    ldaf
    ipiv
    rvec
    cvec
    ldb
    rcond
    ferr
    berr
    work
    iwork
    info
    HypLUSolveCache{R}() where {R <: Real} = new{R}()
end

function HypLUSolveCache(X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypLUSolveCache{R}()
    c.n = LinearAlgebra.checksquare(A)
    @assert c.n == size(X, 1) == size(B, 1)
    @assert size(X, 2) == size(B, 2)
    c.nrhs = size(B, 2)
    c.lda = stride(A, 2)
    c.AF = Matrix{R}(undef, c.n, c.n)
    c.ldaf = c.n
    c.ipiv = Vector{Int}(undef, c.n)
    c.rvec = Vector{R}(undef, c.n)
    c.cvec = Vector{R}(undef, c.n)
    c.ldb = stride(B, 2)
    c.rcond = Ref{R}()
    c.ferr = Vector{R}(undef, c.nrhs)
    c.berr = Vector{R}(undef, c.nrhs)
    c.work = Vector{R}(undef, 4 * c.n)
    c.iwork = Vector{Int}(undef, c.n)
    c.info = Ref{BlasInt}()
    return c
end

for (gesvx, elty, rtyp) in (
    (:dgesvx_, :Float64, :Float64),
    (:sgesvx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_lu_solve!(c::HypLUSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            # call dgesvx( fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info )
            ccall((@blasfunc($gesvx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ref{UInt8}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                'E', 'N', c.n, c.nrhs,
                A, c.lda, c.AF, c.ldaf,
                c.ipiv, 'E', c.rvec, c.cvec,
                B, c.ldb, X, c.n, c.rcond,
                c.ferr, c.berr, c.work, c.iwork, c.info)

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

HypLUSolveCache(X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real} = HypLUSolveCache{R}()

function hyp_lu_solve!(c::HypLUSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    F = lu!(A, check = false)
    if !issuccess(F)
        return false
    end
    ldiv!(X, F, B)
    return true
end

#=
BunchKaufman solve (like SYSVX)
TODO try Aasen's version (http://www.netlib.org/lapack/lawnspdf/lawn294.pdf) and others
=#

mutable struct HypBKSolveCache{R <: Real}
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
    HypBKSolveCache{R}() where {R <: Real} = new{R}()
end

function HypBKSolveCache(uplo::Char, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypBKSolveCache{R}()
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
    c.info = Ref{BlasInt}()
    return c
end

for (sysvx, elty, rtyp) in (
    (:dsysvx_, :Float64, :Float64),
    (:ssysvx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_bk_solve!(c::HypBKSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            # call dsysvx( fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, lwork, iwork, info )
            ccall((@blasfunc($sysvx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                'N', c.uplo, c.n, c.nrhs,
                A, c.lda, c.AF, c.ldaf,
                c.ipiv, B, c.ldb, X,
                c.n, c.rcond, c.ferr, c.berr,
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

function HypBKSolveCache(uplo::Char, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    c = HypBKSolveCache{R}()
    c.uplo = uplo
    return c
end

# fall back to generic LU solve
function hyp_bk_solve!(c::HypBKSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    F = lu!(Symmetric(A, Symbol(c.uplo)), check = false)
    if !issuccess(F)
        return false
    end
    ldiv!(X, F, B)
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

#=
extra precise LU solve (like GESVXX) - requires MKL
=#

mutable struct HypLUxSolveCache{R <: Real}
    n
    nrhs
    lda
    AF
    ldaf
    ipiv
    rvec
    cvec
    ldb
    rcond
    rpvgrw
    berr
    err_bnds_norm
    err_bnds_comp
    params
    work
    iwork
    info
    HypLUxSolveCache{R}() where {R <: Real} = new{R}()
end

function HypLUxSolveCache(X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypLUxSolveCache{R}()
    c.n = LinearAlgebra.checksquare(A)
    @assert c.n == size(X, 1) == size(B, 1)
    @assert size(X, 2) == size(B, 2)
    c.nrhs = size(B, 2)
    c.lda = stride(A, 2)
    c.AF = Matrix{R}(undef, c.n, c.n)
    c.ldaf = c.n
    c.ipiv = Vector{Int}(undef, c.n)
    c.rvec = Vector{R}(undef, c.n)
    c.cvec = Vector{R}(undef, c.n)
    c.ldb = stride(B, 2)
    c.rcond = Ref{R}()
    c.rpvgrw = Ref{R}()
    c.berr = Vector{R}(undef, c.nrhs)
    c.err_bnds_norm = Matrix{R}(undef, c.nrhs, 0)
    c.err_bnds_comp = Matrix{R}(undef, c.nrhs, 0)
    c.params = Matrix{R}(undef, 1, 0)
    c.work = Vector{R}(undef, 4 * c.n)
    c.iwork = Vector{Int}(undef, c.n)
    c.info = Ref{BlasInt}()
    return c
end

for (gesvxx, elty, rtyp) in (
    (:dgesvxx_, :Float64, :Float64),
    (:sgesvxx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_lu_xsolve!(c::HypLUxSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            # call dgesvxx( fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params, work, iwork, info )
            ccall((@blasfunc($gesvxx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ref{UInt8}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                'E', 'N', c.n, c.nrhs,
                A, c.lda, c.AF, c.ldaf,
                c.ipiv, 'B', c.rvec, c.cvec,
                B, c.ldb, X, c.n,
                c.rcond, c.rpvgrw, c.berr, Ref{BlasInt}(0),
                c.err_bnds_norm, c.err_bnds_comp, Ref{BlasInt}(0), c.params,
                c.work, c.iwork, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif 0 < c.info[] <= c.n
                @warn("solve failed: #$(c.info[])")
                return false
            elseif c.info[] > c.n
                @warn("condition number is small: $(c.rcond[])")
            end
            return true
        end
    end
end

HypLUxSolveCache(X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real} = HypLUxSolveCache{R}()

function hyp_lu_xsolve!(c::HypLUxSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    F = lu!(A, check = false)
    if !issuccess(F)
        return false
    end
    ldiv!(X, F, B)
    return true
end

#=
extra precise BunchKaufman solve (like SYSVXX) - requires MKL
=#

mutable struct HypBKxSolveCache{R <: Real}
    uplo
    n
    nrhs
    lda
    AF
    ldaf
    ipiv
    svec
    ldb
    rcond
    rpvgrw
    berr
    err_bnds_norm
    err_bnds_comp
    params
    work
    iwork
    info
    HypBKxSolveCache{R}() where {R <: Real} = new{R}()
end

function HypBKxSolveCache(uplo::Char, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypBKxSolveCache{R}()
    c.n = LinearAlgebra.checksquare(A)
    @assert c.n == size(X, 1) == size(B, 1)
    @assert size(X, 2) == size(B, 2)
    c.nrhs = size(B, 2)
    c.uplo = uplo
    c.lda = stride(A, 2)
    c.AF = Matrix{R}(undef, c.n, c.n)
    c.ldaf = c.n
    c.ipiv = Vector{Int}(undef, c.n)
    c.svec = Vector{R}(undef, c.n)
    c.ldb = stride(B, 2)
    c.rcond = Ref{R}()
    c.rpvgrw = Ref{R}()
    c.berr = Vector{R}(undef, c.nrhs)
    c.err_bnds_norm = Matrix{R}(undef, c.nrhs, 0)
    c.err_bnds_comp = Matrix{R}(undef, c.nrhs, 0)
    c.params = Matrix{R}(undef, 1, 0)
    c.work = Vector{R}(undef, 4 * c.n)
    c.iwork = Vector{Int}(undef, c.n)
    c.info = Ref{BlasInt}()
    return c
end

for (sysvxx, elty, rtyp) in (
    (:dsysvxx_, :Float64, :Float64),
    (:ssysvxx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_bk_xsolve!(c::HypBKxSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            # call dsysvxx( fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, equed, s, b, ldb, x, ldx, rcond, rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params, work, iwork, info )
            ccall((@blasfunc($sysvxx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ref{UInt8}, Ptr{$elty},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                'E', c.uplo, c.n, c.nrhs,
                A, c.lda, c.AF, c.ldaf,
                c.ipiv, 'Y', c.svec,
                B, c.ldb, X, c.n,
                c.rcond, c.rpvgrw, c.berr, Ref{BlasInt}(0),
                c.err_bnds_norm, c.err_bnds_comp, Ref{BlasInt}(0), c.params,
                c.work, c.iwork, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif 0 < c.info[] <= c.n
                @warn("solve failed: #$(c.info[])")
                return false
            elseif c.info[] > c.n
                @warn("condition number is small: $(c.rcond[])")
            end
            return true
        end
    end
end

HypBKxSolveCache(X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real} = HypBKxSolveCache{R}()

# fall back to generic LU solve
function hyp_bk_xsolve!(c::HypBKxSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    F = lu!(Symmetric(A, Symbol(c.uplo)), check = false)
    if !issuccess(F)
        return false
    end
    ldiv!(X, F, B)
    return true
end

#=
extra precise Cholesky solve (like POSVXX) - requires MKL
TODO?
=#
