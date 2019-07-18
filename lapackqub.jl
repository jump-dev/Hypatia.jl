using LinearAlgebra.LAPACK: BlasInt, chklapackerror, @blasfunc, liblapack
using LinearAlgebra.LAPACK: checksquare
for (syequb_, elty, relty) in
    ((:dsyequb_, :Float64, :Float64),
     # (:syequb_, :ComplexF64, :Float64),
     # (:syequb_, :ComplexF32, :Float32),
     # (:syequb_, :Float32, :Float32),
     )
    @eval begin
        function syequb(A::AbstractMatrix{$elty})
            m,n = size(A)
            lda = max(1, stride(A,2))
            C = Vector{$relty}(undef, n)
            R = Vector{$relty}(undef, m)
            info = Ref{BlasInt}()
            rowcond = Ref{$relty}()
            colcond = Ref{$relty}()
            amax = Ref{$relty}()
            ccall((@blasfunc($syequb_), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$relty}, Ptr{$relty},
                   Ptr{$relty}, Ptr{$relty}, Ptr{$relty},
                   Ptr{BlasInt}),
                  'L', n, A, lda, R, C, rowcond, colcond, amax, info)
            chklapackerror(info[])
            R, C, rowcond, colcond, amax
        end
    end
end

using LinearAlgebra.LAPACK: BlasInt, chklapackerror, @blasfunc, liblapack
using LinearAlgebra.LAPACK: checksquare
for (geequb, elty, relty) in
    ((:dgeequb_, :Float64, :Float64),
     (:zgeequb_, :ComplexF64, :Float64),
     (:cgeequb_, :ComplexF32, :Float32),
     (:sgeequb_, :Float32, :Float32))
    @eval begin
#=
*       SUBROUTINE DGEEQUB( M, N, A, LDA, R, C, ROWCND, COLCND, AMAX,
*                           INFO )
*
*       .. Scalar Arguments ..
*       INTEGER            INFO, LDA, M, N
*       DOUBLE PRECISION   AMAX, COLCND, ROWCND
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION   A( LDA, * ), C( * ), R( * )
=#
        function geequb(A::AbstractMatrix{$elty})
            m,n = size(A)
            lda = max(1, stride(A,2))
            C = Vector{$relty}(undef, n)
            R = Vector{$relty}(undef, m)
            info = Ref{BlasInt}()
            rowcond = Ref{$relty}()
            colcond = Ref{$relty}()
            amax = Ref{$relty}()
            ccall((@blasfunc($geequb), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$relty}, Ptr{$relty},
                   Ptr{$relty}, Ptr{$relty}, Ptr{$relty},
                   Ptr{BlasInt}),
                  m, n, A, lda, R, C, rowcond, colcond, amax, info)
            chklapackerror(info[])
            R, C, rowcond, colcond, amax
        end
    end
end
