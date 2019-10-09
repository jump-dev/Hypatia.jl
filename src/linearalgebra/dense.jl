#=
Copyright 2019, Chris Coey and contributors

helpers for dense factorizations and linear solves
NOTE: factorization routines destroy the LHS matrix
TODO use optimal sizes of work arrays etc from LAPACK
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.LAPACK.liblapack

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

#=
nonsymmetric: LU
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

function load_matrix(cache::GESVXNonSymCache{T}, A::Matrix{T}) where {T <: BlasReal}
    cache.is_factorized = false
    LinearAlgebra.require_one_based_indexing(A)
    LinearAlgebra.chkstride1(A)
    n = cache.n = LinearAlgebra.checksquare(A)
    cache.lda = stride(A, 2)
    cache.AF = Matrix{T}(undef, n, n)
    cache.ipiv = Vector{Int}(undef, n)
    cache.equed = Ref{UInt8}('E')
    cache.rvec = Vector{T}(undef, n)
    cache.cvec = Vector{T}(undef, n)
    cache.rcond = Ref{T}()
    cache.ferr = Vector{T}(undef, 0) # NOTE ferr and berr are resized if too small
    cache.berr = Vector{T}(undef, 0)
    cache.work = Vector{T}(undef, 4 * n)
    cache.iwork = Vector{Int}(undef, n)
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK function
for (gesvx, elty) in [(:dgesvx_, :Float64), (:sgesvx_, :Float32)]
    @eval begin
        function solve_system(cache::GESVXNonSymCache{$elty}, X::AbstractVecOrMat{$elty}, A::Matrix{$elty}, B::AbstractVecOrMat{$elty})
            LinearAlgebra.require_one_based_indexing(X, B)
            LinearAlgebra.chkstride1(X, B)

            nrhs = size(B, 2)
            @assert size(X, 2) == nrhs
            if length(cache.ferr) < nrhs
                resize!(cache.ferr, nrhs)
                resize!(cache.berr, nrhs)
            end

            do_fact = (cache.is_factorized ? 'F' : 'E')

            # call dgesvx( fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info )
            ccall((@blasfunc($gesvx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{UInt8}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                Ptr{BlasInt}, Ptr{BlasInt}),
                do_fact, 'N', cache.n, nrhs,
                A, cache.lda, cache.AF, cache.n,
                cache.ipiv, cache.equed, cache.rvec, cache.cvec,
                B, stride(B, 2), X, stride(X, 2),
                cache.rcond, cache.ferr, cache.berr, cache.work,
                cache.iwork, cache.info)

            if cache.info[] < 0
                throw(ArgumentError("invalid argument #$(-cache.info[]) to LAPACK call"))
            elseif 0 < cache.info[] <= cache.n
                @warn("factorization failed: #$(cache.info[])")
                return false
            # elseif cache.info[] > cache.n
            #     @warn("condition number is small: $(cache.rcond[])")
            end

            cache.is_factorized = true
            return true
        end
    end
end

mutable struct LUNonSymCache{T <: Real} <: DenseNonSymCache{T}
    is_factorized::Bool
    fact
    LUNonSymCache{T}() where {T <: Real} = new{T}()
end

function load_matrix(cache::LUNonSymCache{T}, A::AbstractMatrix{T}) where {T <: Real}
    cache.is_factorized = false
    return cache
end

function solve_system(cache::LUNonSymCache{T}, X::AbstractVecOrMat{T}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {T <: Real}
    if !cache.is_factorized
        cache.is_factorized = true
        cache.fact = lu!(A, check = false)
        if !issuccess(cache.fact)
            @warn("numerical failure: LU factorization failed")
            return false
        end
    end

    ldiv!(X, cache.fact, B)

    return issuccess(cache.fact)
end

# default to GESVXNonSymCache for BlasReals, otherwise generic LUNonSymCache
DenseNonSymCache{T}() where {T <: BlasReal} = GESVXNonSymCache{T}()
DenseNonSymCache{T}() where {T <: Real} = LUNonSymCache{T}()

#=
symmetric indefinite: BunchKaufman (and LU fallback)
TODO add a generic BunchKaufman implementation to Julia and use that instead of LU for generic case
TODO try Aasen's version (http://www.netlib.org/lapack/lawnspdf/lawn294.pdf) and others
=#

abstract type DenseSymCache{T <: Real} end

mutable struct LAPACKSymCache{T <: BlasReal} <: DenseSymCache{T}
    uplo
    AF
    ipiv
    work
    lwork
    info
    LAPACKSymCache{T}() where {T <: BlasReal} = new{T}()
end

function load_matrix(cache::LAPACKSymCache{T}, A::Symmetric{T, Matrix{T}}) where {T <: BlasReal}
    LinearAlgebra.require_one_based_indexing(A.data)
    LinearAlgebra.chkstride1(A.data)
    n = LinearAlgebra.checksquare(A.data)
    cache.uplo = A.uplo
    cache.AF = Matrix{T}(undef, n, n)
    cache.ipiv = Vector{Int}(undef, n)
    cache.work = Vector{T}(undef, n) # NOTE this will be resized according to query
    cache.lwork = BlasInt(-1) # NOTE -1 initiates a query for optimal size of work
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK functions
for (sytrf_rook, elty) in [(:dsytrf_rook_, :Float64), (:ssytrf_rook_, :Float32)]
    @eval begin
        function update_fact(cache::LAPACKSymCache{$elty}, A::Symmetric{$elty, <:AbstractMatrix{$elty}})
            copyto!(cache.AF, A.data)

            # call dsytrf_rook( uplo, n, a, lda, ipiv, work, lwork, info )
            ccall((@blasfunc($sytrf_rook), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                cache.uplo, size(cache.AF, 1), cache.AF, stride(cache.AF, 2),
                cache.ipiv, cache.work, cache.lwork, cache.info)

            if cache.lwork < 0 # query for optimal work size, and resize work before solving
                cache.lwork = BlasInt(real(cache.work[1]))
                resize!(cache.work, cache.lwork)
                return update_fact(cache, A)
            end

            if cache.info[] < 0
                throw(ArgumentError("invalid argument #$(-cache.info[]) to LAPACK call"))
            elseif 0 < cache.info[] <= size(cache.AF, 1)
                @warn("factorization failed: #$(cache.info[])")
                return false
            # elseif cache.info[] > size(cache.AF, 1)
            #     @warn("condition number is small: $(cache.rcond[])")
            end

            return true
        end
    end
end

for (sytrs_rook, elty) in [(:dsytrs_rook_, :Float64), (:ssytrs_rook_, :Float32)]
    @eval begin
        function solve_system(cache::LAPACKSymCache{$elty}, X::AbstractVecOrMat{$elty})
            LinearAlgebra.require_one_based_indexing(X)
            LinearAlgebra.chkstride1(X)

            # call dsytrs_rook( uplo, n, nrhs, a, lda, ipiv, b, ldb, info )
            ccall((@blasfunc($sytrs_rook), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}),
                cache.uplo, size(cache.AF, 1), size(X, 2), cache.AF,
                stride(cache.AF, 2), cache.ipiv, X, stride(X, 2),
                cache.info)

            return X
        end
    end
end

for (sytri_rook, elty) in [(:dsytri_rook_, :Float64), (:ssytri_rook_, :Float32)]
    @eval begin
        function invert(cache::LAPACKSymCache{$elty}, X::Symmetric{$elty, <:Matrix{$elty}})
            copyto!(X.data, cache.AF)

            # call dsytri_rook( uplo, n, a, lda, ipiv, work, info )
            ccall((@blasfunc($sytri_rook), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                X.uplo, size(X.data, 1), X.data, stride(X.data, 2),
                cache.ipiv, cache.work, cache.info)

            return X
        end
    end
end

mutable struct LUSymCache{T <: Real} <: DenseSymCache{T}
    is_factorized::Bool
    fact
    LUSymCache{T}() where {T <: Real} = new{T}()
end

function load_matrix(cache::LUSymCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    cache.is_factorized = false
    return cache
end

function solve_system(cache::LUSymCache{T}, X::AbstractVecOrMat{T}, A::Symmetric{T, <:AbstractMatrix{T}}, B::AbstractVecOrMat{T}) where {T <: Real}
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

# default to LAPACKSymCache for BlasReals, otherwise generic LUSymCache
DenseSymCache{T}() where {T <: BlasReal} = LAPACKSymCache{T}()
DenseSymCache{T}() where {T <: Real} = LUSymCache{T}()

#=
symmetric positive definite: Cholesky
=#

abstract type DensePosDefCache{T <: Real} end

mutable struct LAPACKPosDefCache{T <: BlasReal} <: DensePosDefCache{T}
    uplo
    AF
    info
    LAPACKPosDefCache{T}() where {T <: BlasReal} = new{T}()
end

function load_matrix(cache::LAPACKPosDefCache{T}, A::Symmetric{T, Matrix{T}}) where {T <: BlasReal}
    LinearAlgebra.require_one_based_indexing(A.data)
    LinearAlgebra.chkstride1(A.data)
    n = LinearAlgebra.checksquare(A.data)
    cache.uplo = A.uplo
    cache.AF = Matrix{T}(undef, n, n)
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK functions
for (potrf, elty) in [(:dpotrf_, :Float64), (:spotrf_, :Float32)]
    @eval begin
        function update_fact(cache::LAPACKPosDefCache{$elty}, A::Symmetric{$elty, <:AbstractMatrix{$elty}})
            copyto!(cache.AF, A.data)

            # call dpotrf( uplo, n, a, lda, info )
            ccall((@blasfunc($potrf), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}),
                cache.uplo, size(cache.AF, 1), cache.AF, stride(cache.AF, 2),
                cache.info)

            if cache.info[] < 0
                throw(ArgumentError("invalid argument #$(-cache.info[]) to LAPACK call"))
            elseif 0 < cache.info[] <= size(cache.AF, 1)
                @warn("factorization failed: #$(cache.info[])")
                return false
            # elseif cache.info[] > size(cache.AF, 1)
            #     @warn("condition number is small: $(cache.rcond[])")
            end

            return true
        end
    end
end

for (potrs, elty) in [(:dpotrs_, :Float64), (:spotrs_, :Float32)]
    @eval begin
        function solve_system(cache::LAPACKPosDefCache{$elty}, X::AbstractVecOrMat{$elty})
            LinearAlgebra.require_one_based_indexing(X)
            LinearAlgebra.chkstride1(X)

            # call dpotrs( uplo, n, nrhs, a, lda, b, ldb, info )
            ccall((@blasfunc($potrs), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}),
                cache.uplo, size(cache.AF, 1), size(X, 2), cache.AF,
                stride(cache.AF, 2), X, stride(X, 2), cache.info)

            return X
        end
    end
end

for (potri, elty) in [(:dpotri_, :Float64), (:spotri_, :Float32)]
    @eval begin
        function invert(cache::LAPACKPosDefCache{$elty}, X::Symmetric{$elty, <:Matrix{$elty}})
            copyto!(X.data, cache.AF)

            # call dpotri( uplo, n, a, lda, info )
            ccall((@blasfunc($potri), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}),
                X.uplo, size(X.data, 1), X.data, stride(X.data, 2),
                cache.info)

            return X
        end
    end
end

mutable struct CholPosDefCache{T <: Real} <: DensePosDefCache{T}
    is_factorized::Bool
    fact
    CholPosDefCache{T}() where {T <: Real} = new{T}()
end

function load_matrix(cache::CholPosDefCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    cache.is_factorized = false
    return cache
end

function solve_system(cache::CholPosDefCache{T}, X::AbstractVecOrMat{T}, A::Symmetric{T, <:AbstractMatrix{T}}, B::AbstractVecOrMat{T}) where {T <: Real}
    if !cache.is_factorized
        cache.is_factorized = true
        cache.fact = cholesky!(A, check = false)
        if !issuccess(cache.fact)
            @warn("numerical failure: Cholesky factorization failed")
            return false
        end
    end

    ldiv!(X, cache.fact, B)

    return true
end

# default to LAPACKPosDefCache for BlasReals, otherwise generic CholPosDefCache
DensePosDefCache{T}() where {T <: BlasReal} = LAPACKPosDefCache{T}()
DensePosDefCache{T}() where {T <: Real} = CholPosDefCache{T}()
