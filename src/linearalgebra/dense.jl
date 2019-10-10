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

mutable struct LAPACKNonSymCache{T <: BlasReal} <: DenseNonSymCache{T}
    AF
    ipiv
    info
    LAPACKNonSymCache{T}() where {T <: BlasReal} = new{T}()
end

function load_matrix(cache::LAPACKNonSymCache{T}, A::Matrix{T}) where {T <: BlasReal}
    LinearAlgebra.require_one_based_indexing(A)
    LinearAlgebra.chkstride1(A)
    n = LinearAlgebra.checksquare(A)
    cache.AF = Matrix{T}(undef, n, n)
    cache.ipiv = Vector{Int}(undef, n)
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK functions
for (getrf, elty) in [(:dgetrf_, :Float64), (:spotrf_, :Float32)]
    @eval begin
        function update_fact(cache::LAPACKNonSymCache{$elty}, A::AbstractMatrix{$elty})
            n = LinearAlgebra.checksquare(A)
            copyto!(cache.AF, A)

            # call dgetrf( m, n, a, lda, ipiv, info )
            ccall((@blasfunc($getrf), liblapack), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{BlasInt}),
                n, n, cache.AF, stride(cache.AF, 2),
                cache.ipiv, cache.info)

            if cache.info[] < 0
                throw(ArgumentError("invalid argument #$(-cache.info[]) to LAPACK call"))
            elseif 0 < cache.info[] <= n
                @warn("factorization failed: #$(cache.info[])")
                return false
            # elseif cache.info[] > n
            #     @warn("condition number is small: $(cache.rcond[])")
            end

            return true
        end
    end
end

for (getrs, elty) in [(:dgetrs_, :Float64), (:sgetrs_, :Float32)]
    @eval begin
        function solve_system(cache::LAPACKNonSymCache{$elty}, X::AbstractVecOrMat{$elty})
            LinearAlgebra.require_one_based_indexing(X)
            LinearAlgebra.chkstride1(X)

            # call dgetrs( trans, n, nrhs, a, lda, ipiv, b, ldb, info )
            ccall((@blasfunc($getrs), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}),
                'N', size(cache.AF, 1), size(X, 2), cache.AF,
                stride(cache.AF, 2), cache.ipiv, X, stride(X, 2),
                cache.info)

            return X
        end
    end
end

mutable struct LUNonSymCache{T <: Real} <: DenseNonSymCache{T}
    AF
    fact
    LUNonSymCache{T}() where {T <: Real} = new{T}()
end

function load_matrix(cache::LUNonSymCache{T}, A::AbstractMatrix{T}) where {T <: Real}
    n = LinearAlgebra.checksquare(A)
    cache.AF = zeros(T, n, n)
    return cache
end

function update_fact(cache::LUNonSymCache{T}, A::AbstractMatrix{T}) where {T <: Real}
    copyto!(cache.AF, A)
    cache.fact = lu!(A)
    return issuccess(cache.fact)
end

solve_system(cache::LUNonSymCache{T}, X::AbstractVecOrMat{T}) where {T <: Real} = ldiv!(cache.fact, X)

# default to LAPACKNonSymCache for BlasReals, otherwise generic LUNonSymCache
DenseNonSymCache{T}() where {T <: BlasReal} = LAPACKNonSymCache{T}()
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

function load_matrix(cache::LAPACKSymCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: BlasReal}
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
        function invert(cache::LAPACKSymCache{$elty}, X::Symmetric{$elty, Matrix{$elty}})
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
    AF
    fact
    LUSymCache{T}() where {T <: Real} = new{T}()
end

function load_matrix(cache::LUSymCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    n = LinearAlgebra.checksquare(A)
    cache.AF = copy(A)
    return cache
end

function update_fact(cache::LUSymCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    copyto!(cache.AF, A)
    cache.fact = lu!(A)
    return issuccess(cache.fact)
end

solve_system(cache::LUSymCache{T}, X::AbstractVecOrMat{T}) where {T <: Real} = ldiv!(cache.fact, X)

function invert(cache::LUSymCache{T}, X::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    copyto!(X.data, inv(cache.fact))
    return X
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

function load_matrix(cache::LAPACKPosDefCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: BlasReal}
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
        function invert(cache::LAPACKPosDefCache{$elty}, X::Symmetric{$elty, Matrix{$elty}})
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
    AF
    fact
    CholPosDefCache{T}() where {T <: Real} = new{T}()
end

function load_matrix(cache::CholPosDefCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    n = LinearAlgebra.checksquare(A)
    cache.AF = copy(A)
    return cache
end

function update_fact(cache::CholPosDefCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    copyto!(cache.AF, A)
    cache.fact = cholesky!(A)
    return issuccess(cache.fact)
end

solve_system(cache::CholPosDefCache{T}, X::AbstractVecOrMat{T}) where {T <: Real} = ldiv!(cache.fact, X)

function invert(cache::CholPosDefCache{T}, X::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    copyto!(X.data, inv(cache.fact))
    return X
end

# default to LAPACKPosDefCache for BlasReals, otherwise generic CholPosDefCache
DensePosDefCache{T}() where {T <: BlasReal} = LAPACKPosDefCache{T}()
DensePosDefCache{T}() where {T <: Real} = CholPosDefCache{T}()
