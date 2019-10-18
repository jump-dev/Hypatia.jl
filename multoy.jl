function LinearAlgebra.rmul!(A::StridedMatrix, Q::QRSparseQ)
    if size(A, 2) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(A))"))
    end
    tmp = similar(A, size(A, 1))
    for l in 1:size(Q.factors, 2)
        τl = -Q.τ[l]
        h = view(Q.factors, :, l)
        LinearAlgebra.mul!(tmp, A, h)
        LinearAlgebra.lowrankupdate!(A, tmp, h, τl)
    end
    return A
end


function LinearAlgebra.rmul!(B::StridedMatrix, A::SparseMatrixCSC{Float64, <:Integer}, Q::SuiteSparse.SPQR.QRSparseQ)
    (m, n) = size(A)
    B = zeros(m, n)
    if n != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $((size(A)))"))
    end
    row_start = 1
    while row_start <= n
        row_end = min(m, row_start + 3)
        B_block = view(B, row_start:row_end, :)
        B_block .= view(A, row_start:row_end, :)
        tmp = similar(A, size(A_block, 1))
        for l in 1:size(Q.factors, 2)
            τl = -Q.τ[l]
            h = view(Q.factors, :, l)
            LinearAlgebra.mul!(tmp, B_block, h)
            LinearAlgebra.lowrankupdate!(B_block, tmp, h, τl)
        end
        row_start += 4
    end
    return B
end

function LinearAlgebra.rmul!(A::SparseMatrixCSC{Float64, <:Integer}, adjQ::Adjoint{<:Any, <:QSuiteSparse.SPQR.RSparseQ})
    (m, n) = size(A)
    Q = adjQ.parent
    if size(A, 2) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(A))"))
    end
    tmp = similar(A, m)
    col_start = 1
    while col_start < n
        col_end = min(n, col_start + 3)
        A_block = view(A, :, col_start:col_end)
        for l in size(Q.factors, 2):-1:1
            τl = -Q.τ[l]
            h = view(Q.factors, :, l)
            LinearAlgebra.mul!(tmp, A_block, h)
            LinearAlgebra.lowrankupdate!(A_block, tmp, h, τl')
        end
        col_start += 4
    end
    return A
end



for n in [100, 300, 500]
    A = sparse(randn(n, n))
    Q = qr(sparse(randn(n, n))).Q
    A_old = copy(A)
    B = Matrix(copy(A))
    @show n
    @time rmul!(A, Q)
    @time rmul!(B, Q)
    @test isapprox(A, B)
end


function LinearAlgebra.lowrankupdate!(A::AbstractMatrix, x::AbstractVector, y::SparseArrays.SparseVectorUnion, α::Number = 1)
    nzi = SparseArrays.nonzeroinds(y)
    nzv = SparseArrays.nonzeros(y)
    @inbounds for (j, v) in zip(nzi, nzv)
        αv = α * conj(v)
        for i in axes(x, 1)
            A[i, j] += x[i] * αv
        end
    end
    return A
end





using SparseARrays, LinearAlgebra
A = sparse(randn(4, 4))
f = qr(A)
t = -f.Q.τ[1]
h = f.Q.factors[:, 1]
tmp = A * h
LinearAlgebra.lowrankupdate!(A, tmp, h, t) #



method = 0
A = sparse(randn(4, 4))
f = qr(A)
H = SuiteSparse.SPQR.Sparse(f.Q.factors)
Htau = Ref{Ptr{SuiteSparse.CHOLMOD.C_Dense{Float64}}}(C_NULL)
HPinv = Ref{Ptr{SuiteSparse.CHOLMOD.SuiteSparse_long}}()
# HH = unsafe_load(pointer(H))
X = SuiteSparse.SPQR.Dense(randn(4, 4))
X_old = copy(X)
ccall((:SuiteSparseQR_C_qmult, :libspqr), SuiteSparse.CHOLMOD.SuiteSparse_long,
    (
        Cint, # method
        Ptr{SuiteSparse.CHOLMOD.C_Sparse{Float64}}, # H
        Ptr{Ptr{SuiteSparse.CHOLMOD.C_Dense{Float64}}}, # Htau
        Ptr{Ptr{SuiteSparse.CHOLMOD.SuiteSparse_long}}, # HPinv
        # Ptr{SuiteSparse.CHOLMOD.C_Sparse{Float64}}, # X
        Ptr{SuiteSparse.CHOLMOD.C_Dense{Float64}}, # X
        Ptr{Cvoid}, # cc
        ),
    method, H, Htau, HPinv, X, SuiteSparse.CHOLMOD.common_struct)

f.Q' * X_old

Dense(ccall((:SuiteSparseQR_C_qmult, :libspqr), Ptr{C_Dense{Tv}},
        (Cint, Ptr{C_Factorization{Tv}}, Ptr{C_Dense{Tv}}, Ptr{Void}),
            method, H, Htau, HPinv, X.p, SuiteSparse.CHOLMOD.common_struct))


VTypes = Float64
function qmult(method::Integer, QR::Factorization{Tv}, X::Dense{Tv}) where Tv<:VTypes
    mQR, nQR = size(QR)
    mX, nX = size(X)
    if (method == QTX || method == QX) && mQR != mX
        throw(DimensionMismatch("Q matrix size $mQR and dense matrix has $mX rows"))
    elseif (method == XQT || method == XQ) && mQR != nX
        throw(DimensionMismatch("Q matrix size $mQR and dense matrix has $nX columns"))
    end
    d = Dense(ccall((:SuiteSparseQR_C_qmult, :libspqr), Ptr{C_Dense{Tv}},
            (Cint, Ptr{C_Factorization{Tv}}, Ptr{C_Dense{Tv}}, Ptr{Void}),
                method, get(QR.p), get(X.p), common()))
    finalizer(d, free!)
    d
end



ccall((:SuiteSparseQR_C, :libspqr), CHOLMOD.SuiteSparse_long,
(),
)
