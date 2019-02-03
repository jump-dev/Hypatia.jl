#=
Copyright 2018, Chris Coey and contributors

LAPACK helper functions for linear systems
TODO contribute to Julia LinearAlgebra
=#

mutable struct POSVXData
    n::Int
    nrhs::Int
    lda::Int
    ldb::Int
    AF::Matrix{Float64}
    S::Vector{Float64}
    ferr::Vector{Float64}
    berr::Vector{Float64}
    work::Vector{Float64}
    iwork::Vector{BlasInt}
    rcond::Ref{Float64}
    info::Ref{BlasInt}
    fact::Char
    uplo::Char
    equed::Char

    function POSVXData(A::Matrix{Float64}, B::Matrix{Float64})
        d = new()

        d.n = size(B, 1)
        @assert d.n == size(A, 1) == size(A, 2)
        d.nrhs = size(B, 2)
        d.lda = stride(A, 2)
        d.ldb = stride(B, 2)

        d.AF = Matrix{Float64}(undef, d.n, d.n)
        d.S = Vector{Float64}(undef, d.n)
        d.ferr = Vector{Float64}(undef, d.nrhs)
        d.berr = Vector{Float64}(undef, d.nrhs)
        d.work = Vector{Float64}(undef, 3 * d.n)
        d.iwork = Vector{BlasInt}(undef, d.n)
        d.rcond = Ref{Float64}()
        d.info = Ref{BlasInt}()

        d.fact = 'E'
        d.uplo = 'U'
        d.equed = 'Y'

        return d
    end
end

# call LAPACK dposvx function (compare to dposv and dposvxx)
# performs equilibration and iterative refinement
function hypatia_posvx!(
    X::Matrix{Float64},
    A::Matrix{Float64},
    B::Matrix{Float64},
    d::POSVXData,
    )
    @assert d.nrhs == size(X, 2) == size(B, 2)
    @assert d.n == size(A, 1) == size(A, 2) == size(X, 1) == size(B, 1)

    ccall((LinearAlgebra.BLAS.@blasfunc(dposvx_), Base.liblapack_name), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
        Ref{UInt8}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
        Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}),
        d.fact, d.uplo, d.n, d.nrhs, A, d.lda, d.AF, d.lda, d.equed, d.S, B,
        d.ldb, X, d.n, d.rcond, d.ferr, d.berr, d.work, d.iwork, d.info)

    if d.info[] != 0 && d.info[] != d.n+1
        println("failure to solve linear system (posvx status $(d.info[]))")
        return false
    end
    return true
end

# call LAPACK dsysvx function (compare to dsysv and dsysvxx)
# performs equilibration and iterative refinement
function hypatia_sysvx!(
    X::Matrix{Float64},
    A::Matrix{Float64},
    B::Matrix{Float64},
    ferr,
    berr,
    work,
    iwork,
    AF,
    ipiv,
    )
    n = size(A, 1)
    @assert n == size(A, 2) == size(B, 1)

    lda = stride(A, 2)
    nrhs = size(B, 2)
    ldb = stride(B, 2)
    rcond = Ref{Float64}()
    lwork = Ref{BlasInt}(5n)
    fact = 'N'
    uplo = 'U'

    info = Ref{BlasInt}()

    ccall((LinearAlgebra.BLAS.@blasfunc(dsysvx_), Base.liblapack_name), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
        Ptr{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
        Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
        Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        fact, uplo, n, nrhs, A, lda, AF, lda, ipiv, B,
        ldb, X, n, rcond, ferr, berr, work, lwork, iwork, info)

    if lwork[] > 5n
        println("in sysvx, lwork increased from $(5n) to $(lwork[])")
    end
    if info[] != 0 && info[] != n+1
        println("failure to solve linear system (sysvx status $(info[]))")
        return false
    end
    return true
end
