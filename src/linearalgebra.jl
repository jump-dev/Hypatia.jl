#=
Copyright 2019, Chris Coey and contributors
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.BlasFloat
import LinearAlgebra.HermOrSym
import LinearAlgebra.mul!
import LinearAlgebra.gemv!
using LinearAlgebra.LAPACK: BlasInt, chklapackerror, @blasfunc, liblapack
using LinearAlgebra.LAPACK: checksquare
import Base.adjoint
import Base.eltype
import Base.size
import Base.*
import Base.-

# for (syequb_, elty, relty) in
#     ((:dsyequb_, :Float64, :Float64),
#      # (:syequb_, :ComplexF64, :Float64),
#      # (:syequb_, :ComplexF32, :Float32),
#      # (:syequb_, :Float32, :Float32),
#      )
#     @eval begin
#         function sysvx(A::AbstractMatrix{$elty}, b::AbstractMatrix{$elty})
#             m,n = size(A)
#             lda = max(1, stride(A,2))
#             S = Vector{$relty}(undef, n)
#             info = Ref{BlasInt}()
#             scond = Ref{$relty}()
#             amax = Ref{$relty}()
#             work = Vector{$relty}(undef, 3 * n)
#             ccall((@blasfunc($syequb_), liblapack), Cvoid,
#                   (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
#                    Ptr{$relty},
#                    Ptr{$relty}, Ptr{$relty}, Ptr{$relty},
#                    Ptr{BlasInt}),
#                   'L', n, A, lda, S, scond, amax, work, info)
#             chklapackerror(info[])
#             S, scond, amax
#         end
#     end
# end
#
# function hyp_sysvx!(A::AbstractMatrix{$elty}, B::AbstractVecOrMat{$elty})
#     n = size(A, 1)
#     @assert n == size(A, 2) == size(B, 1)
#
#     lda = stride(A, 2)
#     nrhs = size(B, 2)
#     ldb = stride(B, 2)
#     rcond = Ref{Float64}()
#
#     fact = 'N'
#     uplo = 'U'
#     equed = 'Y'
#
#     info = Ref{BlasInt}()
#
#     ccall((@blasfunc(dposvx_), Base.liblapack_name), Cvoid,
#         (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
#         Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
#         Ref{UInt8}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt},
#         Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
#         Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}),
#         fact, uplo, n, nrhs, A, lda, AF, lda, equed, S, B,
#         ldb, X, n, rcond, ferr, berr, work, iwork, info)
#     B, A, ipiv
# end
#
# function sysv!(uplo::AbstractChar, A::AbstractMatrix{$elty}, B::AbstractVecOrMat{$elty})
#     @assert !has_offset_axes(A, B)
#     chkstride1(A,B)
#     n = checksquare(A)
#     chkuplo(uplo)
#     if n != size(B,1)
#         throw(DimensionMismatch("B has first dimension $(size(B,1)), but needs $n"))
#     end
#     ipiv  = similar(A, BlasInt, n)
#     work  = Vector{$elty}(undef, 1)
#     lwork = BlasInt(-1)
#     info  = Ref{BlasInt}()
    # for i = 1:2  # first call returns lwork as work[1]
    #     ccall((@blasfunc($sysv), liblapack), Cvoid,
    #           (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
    #            Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
    #           uplo, n, size(B,2), A, max(1,stride(A,2)), ipiv, B, max(1,stride(B,2)),
    #           work, lwork, info)
    #     chkargsok(info[])
    #     chknonsingular(info[])
    #     if i == 1
    #         lwork = BlasInt(real(work[1]))
    #         resize!(work, lwork)
    #     end
#     end
#     B, A, ipiv
# end


function hyp_sysvx!(
    X::Matrix{Float64},
    A::Matrix{Float64},
    B::Matrix{Float64},
    )

    n = size(A, 1)
    @assert n == size(A, 2) == size(B, 1)

    lda = stride(A, 2)
    ldaf = n
    nrhs = size(B, 2)
    ldb = stride(B, 2)
    rcond = Ref{Float64}()
    lwork = Ref{BlasInt}(5n)
    fact = 'N'
    uplo = 'U'

    ferr = Vector{Float64}(undef, nrhs)
    berr = Vector{Float64}(undef, nrhs)
    work = Vector{Float64}(undef, 5n)
    iwork = Vector{Int}(undef, n)
    AF = Matrix{Float64}(undef, ldaf, n)
    ipiv = Vector{Int}(undef, n)

    info = Ref{BlasInt}()

    # for i in 1:2
        ccall((@blasfunc(dsysvx_), Base.liblapack_name), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
            Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
            fact, uplo, n, nrhs, A, lda, AF, ldaf, ipiv, B,
            ldb, X, n, rcond, ferr, berr, work, lwork, iwork, info)

    #     if i == 1
    #         lwork = BlasInt(real(work[1]))
    #         @show n, lwork
    #         resize!(work, lwork)
    #     end
    # end

    if lwork[] > 5n
        println("in sysvx, lwork increased from $(5n) to $(lwork[])")
    end
    if info[] != 0 && info[] != n+1
        println("failure to solve linear system (sysvx status $(info[]))")
        return false
    end
    return true
end

function hyp_posvx!(
    X::Matrix{Float64},
    A::Matrix{Float64},
    B::Matrix{Float64},
    )

    n = size(A, 1)
    @assert n == size(A, 2) == size(B, 1)

    lda = stride(A, 2)
    ldaf = n
    nrhs = size(B, 2)
    ldb = stride(B, 2)
    rcond = Ref{Float64}()
    lwork = Ref{BlasInt}(5n)
    fact = 'E'
    uplo = 'U'
    equed = 'Y'

    ferr = Vector{Float64}(undef, nrhs)
    berr = Vector{Float64}(undef, nrhs)
    work = Vector{Float64}(undef, 5n)
    iwork = Vector{Int}(undef, n)
    AF = Matrix{Float64}(undef, ldaf, n)
    ipiv = Vector{Int}(undef, n)
    S = Vector{Float64}(undef, n)

    info = Ref{BlasInt}()

    # for i in 1:2
        ccall((@blasfunc(dposvx_), Base.liblapack_name), Cvoid,
            (Ref{UInt8},
            Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
            Ref{UInt8}, Ptr{Float64},
            Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
            Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}),
            fact, uplo, n, nrhs, A, lda, AF, ldaf, equed, S, B,
            ldb, X, n, rcond, ferr, berr, work, iwork, info)

            @show n, lwork[]

    #     if i == 1
    #         lwork = BlasInt(real(work[1]))
    #         @show n, lwork
    #         resize!(work, lwork)
    #     end
    # end

    # @show S, equed

    if lwork[] > 5n
        println("in sysvx, lwork increased from $(5n) to $(lwork[])")
    end
    if info[] != 0 && info[] != n+1
        println("failure to solve linear system (sysvx status $(info[]))")
        return false
    end
    return (true, S)
end

function hyp_posvxx!(
    X::Matrix{Float64},
    A::Matrix{Float64},
    B::Matrix{Float64},
    )

    n = size(A, 1)
    @assert n == size(A, 2) == size(B, 1)

    lda = stride(A, 2)
    ldaf = n
    nrhs = size(B, 2)
    ldb = stride(B, 2)
    rcond = Ref{Float64}()
    lwork = Ref{BlasInt}(5n)
    fact = 'E'
    uplo = 'U'
    equed = 'Y'

    rpvgrw = Ref{Float64}()
    berr = Vector{Float64}(undef, nrhs)
    n_err_bnds = 1 # TODO understand how to use
    err_bnds_norm = Matrix{Float64}(undef, nrhs, n_err_bnds)
    err_bnds_comp = Matrix{Float64}(undef, nrhs, n_err_bnds)
    nparams = 0 # TODO understand how to use
    params = Vector{Float64}(undef, nparams)
    work = Vector{Float64}(undef, 5n)
    iwork = Vector{Int}(undef, n)
    S = Vector{Float64}(undef, n)

    info = Ref{BlasInt}()

    # for i in 1:2
        ccall((@blasfunc(dposvxx_), Base.liblapack_name), Cvoid,
            (Ref{UInt8}, # fact
            Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, #  uplo, n, nrhs
            Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt}, # A, lda, ldaf
            Ref{UInt8}, Ptr{Float64}, # equed, S
            Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, # B, ldb, X,
            Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, # ldx/n, rcond, rpvgrw, berr
            Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, # n_err_bnds, err_bnds_norm, err_bnds_comp,
            Ref{BlasInt}, Ptr{BlasInt}, # nparams, params
            Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}),
            fact, uplo, n, nrhs, A, lda, ldaf, equed, S, B,
            ldb, X, n, rcond, rpvgrw, berr,
            n_err_bnds, err_bnds_norm, err_bnds_comp,
            nparams, params,
            work, iwork, info)

    #     if i == 1
    #         lwork = BlasInt(real(work[1]))
    #         @show n, lwork
    #         resize!(work, lwork)
    #     end
    # end

    # @show S, equed
    @show rcond

    if lwork[] > 5n
        println("in sysvx, lwork increased from $(5n) to $(lwork[])")
    end
    if info[] != 0 && info[] != n+1
        println("failure to solve linear system (sysvx status $(info[]))")
        return false
    end
    return true
end

function equilibrators(A::AbstractMatrix{T}) where {T}
    m,n = size(A)
    R = zeros(T,m)
    C = zeros(T,n)
    @inbounds for j=1:n
        for i=1:m
            R[i] = max(R[i],abs(A[i,j]))
        end
    end
    @inbounds for i=1:m
        if R[i] > 0
            R[i] = T(2)^floor(Int,log2(R[i]))
        end
    end
    R .= 1 ./ R
    @inbounds for i=1:m
        for j=1:n
            C[j] = max(C[j],R[i] * abs(A[i,j]))
        end
    end
    @inbounds for j=1:n
        if C[j] > 0
            C[j] = T(2)^floor(Int,log2(C[j]))
        end
    end
    C .= 1 ./ C
    R,C
end


hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: BlasReal} = BLAS.syrk!('U', 'T', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{Complex{T}}, A::Matrix{Complex{T}}) where {T <: BlasReal} = BLAS.herk!('U', 'C', one(T), A, zero(T), U)
hyp_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: RealOrComplex{<:Real}} = mul!(U, A', A)

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

struct BlockMatrix{T <: Real}
    nrows::Int
    ncols::Int
    blocks::Vector
    rows::Vector{UnitRange{Int}}
    cols::Vector{UnitRange{Int}}

    function BlockMatrix{T}(nrows::Int, ncols::Int, blocks::Vector, rows::Vector{UnitRange{Int}}, cols::Vector{UnitRange{Int}}) where {T <: Real}
        @assert length(blocks) == length(rows) == length(cols)
        return new{T}(nrows, ncols, blocks, rows, cols)
    end
end

eltype(A::BlockMatrix{T}) where {T <: Real} = T

size(A::BlockMatrix) = (A.nrows, A.ncols)
size(A::BlockMatrix, d) = (d == 1 ? A.nrows : A.ncols)

adjoint(A::BlockMatrix{T}) where {T <: Real} = BlockMatrix{T}(A.ncols, A.nrows, adjoint.(A.blocks), A.cols, A.rows)

# TODO try to speed up by using better logic for alpha and beta (see Julia's 5-arg mul! code)
# TODO check that this eliminates allocs when using IterativeSolvers methods, and that it is as fast as possible
function mul!(y::AbstractVector{T}, A::BlockMatrix{T}, x::AbstractVector{T}, alpha::Number, beta::Number) where {T <: Real}
    @assert size(x, 1) == A.ncols
    @assert size(y, 1) == A.nrows
    @assert size(x, 2) == size(y, 2)

    @. y *= beta
    for (b, r, c) in zip(A.blocks, A.rows, A.cols)
        if isempty(r) || isempty(c)
            continue
        end
        xk = view(x, c)
        yk = view(y, r)
        mul!(yk, b, xk, alpha, true)
    end
    return y
end

*(A::BlockMatrix{T}, x::AbstractVector{T}) where {T <: Real} = mul!(similar(x, size(A, 1)), A, x)

-(A::BlockMatrix{T}) where {T <: Real} = BlockMatrix{T}(A.nrows, A.ncols, -A.blocks, A.rows, A.cols)
