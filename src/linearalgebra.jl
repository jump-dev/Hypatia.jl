#=
Copyright 2019, Chris Coey and contributors
=#

# ensure diagonal terms in symm/herm (and ideally PSD) matrix are not too small
function set_min_diag!(A::Matrix{<:RealOrComplex{T}}, tol::T) where {T <: Real}
    if tol <= 0
        return A
    end
    @inbounds for j in 1:size(A, 1)
        if A[j, j] < tol
            A[j, j] = tol
        end
    end
    return A
end


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


# TODO equilibrate for cholesky and bk - use DPOEQUB/DSYEQUB


# cache for LAPACK cholesky (like POTRF)
mutable struct HypCholCache{R <: Real, T <: RealOrComplex{R}}
    tol_diag
    uplo
    n
    lda
    info
    HypCholCache{R, T}() where {T <: RealOrComplex{R}} where {R <: Real} = new{R, T}()
end

function HypCholCache(uplo::Char, A::StridedMatrix{T}; tol_diag = zero(R)) where {T <: RealOrComplex{R}} where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    Base.require_one_based_indexing(A)
    c = HypCholCache{R, T}()
    c.tol_diag = tol_diag
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
            set_min_diag!(A, c.tol_diag)

            ccall((@blasfunc($potrf), liblapack), Cvoid, (
                Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                ), c.uplo, c.n, A, c.lda, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            end

            return Cholesky{$elty, typeof(A)}(A, c.uplo, c.info[])
        end
    end

    @eval begin
        function hyp_chol_inv!(c::HypCholCache{$rtyp, $elty}, fact_A::Cholesky{$elty, <:StridedMatrix{$elty}})
            ccall((@blasfunc($potri), liblapack), Cvoid, (
                Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                ), c.uplo, c.n, fact_A.factors, c.lda, c.info)

            if c.info[] < 0
                throw(ArgumentError("invalid argument #$(-c.info[]) to LAPACK call"))
            elseif c.info[] > 0
                error("failed to calculate matrix inverse from cholesky")
            end

            return fact_A.factors
        end
    end

end

function HypCholCache(uplo::Char, A::AbstractMatrix{T}; tol_diag = zero(R)) where {T <: RealOrComplex{R}} where {R <: Real}
    c = HypCholCache{R, T}()
    c.tol_diag = tol_diag
    c.uplo = uplo
    return c
end

function hyp_chol!(c::HypCholCache{R, T}, A::AbstractMatrix{T}) where {T <: RealOrComplex{R}} where {R <: Real}
    set_min_diag!(A, c.tol_diag)
    return cholesky!(Hermitian(A, Symbol(c.uplo)), check = false)
end

function hyp_chol_inv!(c::HypCholCache{R, T}, fact_A::Cholesky{T, <:AbstractMatrix{T}}) where {T <: RealOrComplex{R}} where {R <: Real}
    return inv(fact_A)
end

# cache for LAPACK Bunch-Kaufman (like SYTRF_ROOK)
mutable struct HypBKCache{R <: Real}
    tol_diag
    uplo
    n
    lda
    ipiv
    work
    lwork
    info
    HypBKCache{R}() where {R <: Real} = new{R}()
end

function HypBKCache(uplo::Char, A::AbstractMatrix{R}; tol_diag = zero(R)) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypBKCache{R}()
    c.tol_diag = tol_diag
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
    )
    @eval begin
        function hyp_bk!(c::HypBKCache{$elty}, A::AbstractMatrix{$elty})
            set_min_diag!(A, c.tol_diag)

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

function HypBKCache(uplo::Char, A::AbstractMatrix{R}; tol_diag = zero(R)) where {R <: Real}
    c = HypBKCache{R}()
    c.tol_diag = tol_diag
    c.uplo = uplo
    return c
end

# fall back to Cholesky for eltype not BlasReal
function hyp_bk!(c::HypBKCache{R}, A::AbstractMatrix{R}) where {R <: Real}
    set_min_diag!(A, c.tol_diag)
    return cholesky!(Symmetric(A, Symbol(c.uplo)), check = false)
end

# cache for LAPACK cholesky with linear solve (like POSVX)
mutable struct HypCholSolveCache{R <: Real}
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
    S
    info
    HypCholSolveCache{R}() where {R <: Real} = new{R}()
end

# NOTE X (solution) argument needs to be a Matrix type or else silent failures occur
function HypCholSolveCache(uplo::Char, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}; tol_diag = zero(R)) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypCholSolveCache{R}()
    c.tol_diag = tol_diag
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

# access LAPACK functions
for (posvx, elty, rtyp) in (
    (:dposvx_, :Float64, :Float64),
    (:sposvx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_chol_solve!(c::HypCholSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            set_min_diag!(A, c.tol_diag)

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

function HypCholSolveCache(uplo::Char, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}; tol_diag = zero(R)) where {R <: Real}
    c = HypCholSolveCache{R}()
    c.tol_diag = tol_diag
    c.uplo = uplo
    return c
end

function hyp_chol_solve!(c::HypCholSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    set_min_diag!(A, c.tol_diag)
    F = cholesky!(Symmetric(A, Symbol(c.uplo)), check = false)
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
function HypBKSolveCache(uplo::Char, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}; tol_diag = zero(R)) where {R <: BlasReal}
    LinearAlgebra.chkstride1(A)
    c = HypBKSolveCache{R}()
    c.tol_diag = tol_diag
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

# access LAPACK functions
for (sysvx, elty, rtyp) in (
    (:dsysvx_, :Float64, :Float64),
    (:ssysvx_, :Float32, :Float32),
    )
    @eval begin
        function hyp_bk_solve!(c::HypBKSolveCache{$elty}, X::Matrix{$elty}, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            set_min_diag!(A, c.tol_diag)

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

function HypBKSolveCache(uplo::Char, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}; tol_diag = zero(R)) where {R <: Real}
    c = HypBKSolveCache{R}()
    c.tol_diag = tol_diag
    c.uplo = uplo
    return c
end

# fall back to Cholesky solve for eltype not BlasReal
function hyp_bk_solve!(c::HypBKSolveCache{R}, X::Matrix{R}, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    set_min_diag!(A, c.tol_diag)
    F = cholesky!(Symmetric(A, Symbol(c.uplo)), check = false)
    if !isposdef(F)
        return false
    end
    ldiv!(X, F, B)
    return true
end
