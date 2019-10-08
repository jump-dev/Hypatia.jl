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
        function solve_dense_system(cache::GESVXNonSymCache{$elty}, X::AbstractVecOrMat{$elty}, A::Matrix{$elty}, B::AbstractVecOrMat{$elty})
            LinearAlgebra.require_one_based_indexing(X, B)
            LinearAlgebra.chkstride1(X, B)

            nrhs = size(B, 2)
            @assert size(X, 2) == nrhs
            if length(cache.ferr) < nrhs
                resize!(cache.ferr, nrhs)
                resize!(cache.berr, nrhs)
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

function solve_dense_system(cache::LUNonSymCache{T}, X::AbstractVecOrMat{T}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {T <: Real}
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
symmetric indefinite: BunchKaufman / SYSVX (and LU fallback)
TODO add a generic BunchKaufman implementation to Julia and use that instead of LU for generic case
TODO try Aasen's version (http://www.netlib.org/lapack/lawnspdf/lawn294.pdf) and others
=#

abstract type DenseSymCache{T <: Real} end

reset_fact(cache::DenseSymCache) = (cache.is_factorized = false)

mutable struct SYSVXSymCache{T <: BlasReal} <: DenseSymCache{T}
    is_factorized::Bool
    n
    lda
    AF
    ipiv
    rcond
    ferr
    berr
    work
    lwork
    iwork
    info
    SYSVXSymCache{T}() where {T <: BlasReal} = new{T}()
end

function load_dense_matrix(cache::SYSVXSymCache{T}, A::Symmetric{T, Matrix{T}}) where {T <: BlasReal}
    cache.is_factorized = false
    LinearAlgebra.require_one_based_indexing(A.data)
    LinearAlgebra.chkstride1(A.data)
    n = cache.n = LinearAlgebra.checksquare(A.data)
    cache.lda = stride(A.data, 2)
    cache.AF = Matrix{T}(undef, n, n)
    cache.ipiv = Vector{Int}(undef, n)
    cache.rcond = Ref{T}()
    cache.ferr = Vector{T}(undef, 0) # NOTE ferr and berr are resized if too small
    cache.berr = Vector{T}(undef, 0)
    cache.work = Vector{T}(undef, 1) # NOTE this will be resized according to query
    cache.lwork = BlasInt(-1) # NOTE this initiates a query for optimal size of work
    cache.iwork = Vector{Int}(undef, n)
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK function
for (sysvx, elty) in [(:dsysvx_, :Float64), (:ssysvx_, :Float32)]
    @eval begin
        function solve_dense_system(cache::SYSVXSymCache{$elty}, X::AbstractVecOrMat{$elty}, A::Symmetric{$elty, Matrix{$elty}}, B::AbstractVecOrMat{$elty})
            LinearAlgebra.require_one_based_indexing(X, B)
            LinearAlgebra.chkstride1(X, B)

            nrhs = size(B, 2)
            @assert size(X, 2) == nrhs
            if length(cache.ferr) < nrhs
                resize!(cache.ferr, nrhs)
                resize!(cache.berr, nrhs)
            end

            if cache.is_factorized
                cache.is_factorized = true
                do_fact = 'F'
            else
                do_fact = 'N'
            end

            # call dsysvx( fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, lwork, iwork, info )
            ccall((@blasfunc($sysvx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                do_fact, A.uplo, cache.n, nrhs,
                A.data, cache.lda, cache.AF, cache.n,
                cache.ipiv, B, stride(B, 2), X,
                stride(X, 2), cache.rcond, cache.ferr, cache.berr,
                cache.work, cache.lwork, cache.iwork, cache.info)

            if cache.lwork < 0 # query for optimal work size, and resize work before solving
                cache.lwork = BlasInt(cache.work[1])
                resize!(cache.work, cache.lwork)
                return solve_dense_system(cache, X, A, B)
            end

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

mutable struct LUSymCache{T <: Real} <: DenseSymCache{T}
    is_factorized::Bool
    fact
    LUSymCache{T}() where {T <: Real} = new{T}()
end

function load_dense_matrix(cache::LUSymCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    cache.is_factorized = false
    return cache
end

function solve_dense_system(cache::LUSymCache{T}, X::AbstractVecOrMat{T}, A::Symmetric{T, <:AbstractMatrix{T}}, B::AbstractVecOrMat{T}) where {T <: Real}
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

# default to SYSVXSymCache for BlasReals, otherwise generic LUSymCache
DenseSymCache{T}() where {T <: BlasReal} = SYSVXSymCache{T}()
DenseSymCache{T}() where {T <: Real} = LUSymCache{T}()

#=
symmetric positive definite: Cholesky / POSVX
=#

abstract type DensePosDefCache{T <: Real} end

reset_fact(cache::DensePosDefCache) = (cache.is_factorized = false)

mutable struct POSVXSymCache{T <: BlasReal} <: DensePosDefCache{T}
    is_factorized::Bool
    n
    lda
    AF
    equed
    s
    rcond
    ferr
    berr
    work
    iwork
    info
    POSVXSymCache{T}() where {T <: BlasReal} = new{T}()
end

function load_dense_matrix(cache::POSVXSymCache{T}, A::Symmetric{T, Matrix{T}}) where {T <: BlasReal}
    cache.is_factorized = false
    LinearAlgebra.require_one_based_indexing(A.data)
    LinearAlgebra.chkstride1(A.data)
    n = cache.n = LinearAlgebra.checksquare(A.data)
    cache.lda = stride(A.data, 2)
    cache.AF = Matrix{T}(undef, n, n)
    cache.equed = Ref{UInt8}('E')
    cache.s = Vector{T}(undef, n)
    cache.rcond = Ref{T}()
    cache.ferr = Vector{T}(undef, 0) # NOTE ferr and berr are resized if too small
    cache.berr = Vector{T}(undef, 0)
    cache.work = Vector{T}(undef, 3 * n)
    cache.iwork = Vector{Int}(undef, n)
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK function
for (posvx, elty) in [(:dposvx_, :Float64), (:sposvx_, :Float32)]
    @eval begin
        function solve_dense_system(cache::POSVXSymCache{$elty}, X::AbstractVecOrMat{$elty}, A::Symmetric{$elty, Matrix{$elty}}, B::AbstractVecOrMat{$elty})
            LinearAlgebra.require_one_based_indexing(X, B)
            LinearAlgebra.chkstride1(X, B)

            nrhs = size(B, 2)
            @assert size(X, 2) == nrhs
            if length(cache.ferr) < nrhs
                resize!(cache.ferr, nrhs)
                resize!(cache.berr, nrhs)
            end

            set_min_diag!(A.data, sqrt(eps($elty)))

            if cache.is_factorized
                cache.is_factorized = true
                do_fact = 'F'
            else
                do_fact = 'E'
            end

            # call dposvx( fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info )
            ccall((@blasfunc($posvx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{UInt8}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                do_fact, A.uplo, cache.n, nrhs,
                A.data, cache.lda, cache.AF, cache.n,
                cache.equed, cache.s, B, stride(B, 2),
                X, stride(X, 2), cache.rcond, cache.ferr,
                cache.berr, cache.work, cache.iwork, cache.info)

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

mutable struct CholSymCache{T <: Real} <: DensePosDefCache{T}
    is_factorized::Bool
    fact
    CholSymCache{T}() where {T <: Real} = new{T}()
end

function load_dense_matrix(cache::CholSymCache{T}, A::Symmetric{T, <:AbstractMatrix{T}}) where {T <: Real}
    cache.is_factorized = false
    return cache
end

function solve_dense_system(cache::CholSymCache{T}, X::AbstractVecOrMat{T}, A::Symmetric{T, <:AbstractMatrix{T}}, B::AbstractVecOrMat{T}) where {T <: Real}
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

# default to POSVXSymCache for BlasReals, otherwise generic CholSymCache
DensePosDefCache{T}() where {T <: BlasReal} = POSVXSymCache{T}()
DensePosDefCache{T}() where {T <: Real} = CholSymCache{T}()
