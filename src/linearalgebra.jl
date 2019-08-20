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
import LinearAlgebra.issuccess

LinearAlgebra.issuccess(F::Union{Cholesky, CholeskyPivoted}) = isposdef(F)


# TODO delete later
hyp_chol!(A::HermOrSym{T, Matrix{T}}) where {T <: BlasFloat} = cholesky!(A, Val(true), check = false)
hyp_chol!(A::HermOrSym{T, Matrix{T}}) where {T <: RealOrComplex{<:Real}} = cholesky!(A, check = false)



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

# cache for LAPACK Bunch-Kaufman (like SYTRF_ROOK)
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
    c.lwork = hyp_bk!(c, A)
    c.work = Vector{R}(undef, c.lwork)
    return c
end

for (sytrf, elty, rtyp) in (
    (:dsytrf_rook_, :Float64, :Float64),
    (:ssytrf_rook_, :Float32, :Float32),
    # (:zsytrf_rook_, :ComplexF64, :Float64),
    # (:csytrf_rook_, :ComplexF32, :Float32),
    )
    @eval begin
        function hyp_bk!(c::HypBKCache{$elty}, A::AbstractMatrix{$elty})
            ccall((@blasfunc($sytrf), liblapack), Cvoid, (
                Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}
                ), c.uplo, c.n, A, c.lda, c.ipiv, c.work, c.lwork, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif c.lwork == -1
                return BlasInt(real(c.work[1]))
            elseif c.info[] == c.n
                println("RCOND is small: $(c.rcond[])")
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

# fall back to Cholesky for eltype not BlasReal
hyp_bk!(c::HypBKCache{R}, A::AbstractMatrix{R}) where {R <: Real} = cholesky!(Symmetric(A, Symbol(c.uplo)), check = false)

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

# NOTE X (solution) argument needs to be a Matrix type or else silent failures occur
function HypCholSolveCache(use_upper::Bool, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: BlasReal}
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
        function hyp_chol_solve!(c::HypCholSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
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
                println("factorization failed: #$(c.info[])")
                return false
            elseif c.info[] == c.n
                println("RCOND is small: $(c.rcond[])")
            end
            return true
        end
    end
end

function HypCholSolveCache(use_upper::Bool, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    c = HypCholSolveCache{R}()
    c.uplo = (use_upper ? :U : :L)
    return c
end

function hyp_chol_solve!(c::HypCholSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    F = cholesky!(Symmetric(A, c.uplo), check = false)
    if !isposdef(F)
        return false
    end
    ldiv!(X, F, B)
    return true
end

# cache for LAPACK Bunch-Kaufman with linear solve (like SYSVX)
# TODO try Aasen's version (http://www.netlib.org/lapack/lawnspdf/lawn294.pdf) and other
mutable struct HypBKSolveCache{R <: Real}
    tol_diag
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

# NOTE X (solution) argument needs to be a Matrix type or else silent failures occur
function HypBKSolveCache(use_upper::Bool, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypBKSolveCache{R}()
    c.tol_diag = sqrt(eps(R)) # TODO tune
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
        function hyp_bk_solve!(c::HypBKSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            # ensure diagonal terms in symmetric (and ideally PSD) matrix are not too small
            @inbounds for j in 1:c.n
                if A[j, j] < c.tol_diag
                    A[j, j] = c.tol_diag
                end
            end

            ccall((@blasfunc($sysvx), Base.liblapack_name), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                'N', c.uplo, c.n, c.nrhs, A, c.lda, c.AF, c.ldaf, c.ipiv, B,
                c.ldb, X, c.n, c.rcond, c.ferr, c.berr, c.work, c.lwork, c.iwork, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif 0 < c.info[] <= c.n
                println("factorization failed: #$(c.info[])")
                return false
            elseif c.info[] == c.n
                println("RCOND is small: $(c.rcond[])")
            end
            return true
        end
    end
end

function HypBKSolveCache(use_upper::Bool, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    c = HypBKSolveCache{R}()
    c.tol_diag = sqrt(eps(R)) # TODO tune
    c.uplo = (use_upper ? :U : :L)
    return c
end

# fall back to Cholesky solve for eltype not BlasReal
function hyp_bk_solve!(c::HypBKSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    # ensure diagonal terms in symmetric (and ideally PSD) matrix are not too small
    @inbounds for j in 1:size(A, 1)
        if A[j, j] < c.tol_diag
            A[j, j] = c.tol_diag
        end
    end
    F = cholesky!(Symmetric(A, c.uplo), check = false)
    if !isposdef(F)
        return false
    end
    ldiv!(X, F, B)
    return true
end
