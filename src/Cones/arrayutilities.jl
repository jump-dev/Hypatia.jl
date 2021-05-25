#=
utilities for arrays
=#

"""
$(SIGNATURES)

The dimension of the real vectorization of a real or complex vector of length
`len::Int`.
"""
vec_length(::Type{<:Real}, len::Int) = len

vec_length(::Type{<:Complex}, len::Int) = len + len

"""
$(SIGNATURES)

Copy a vector in-place.
"""
vec_copyto!(
    v1::AbstractVecOrMat{T},
    v2::AbstractVecOrMat{T},
    ) where T = copyto!(v1, v2)

"""
$(SIGNATURES)

Copy a complex vector to a real vector in-place.
"""
function vec_copyto!(
    rvec::AbstractVecOrMat{T},
    cvec::AbstractVecOrMat{Complex{T}},
    ) where T
    @assert length(rvec) == 2 * length(cvec)
    k = 1
    @inbounds for i in eachindex(cvec)
        ci = cvec[i]
        rvec[k] = real(ci)
        rvec[k + 1] = imag(ci)
        k += 2
    end
    return rvec
end

"""
$(SIGNATURES)

Copy a real vector to a complex vector in-place.
"""
function vec_copyto!(
    cvec::AbstractVecOrMat{Complex{T}},
    rvec::AbstractVecOrMat{T},
    ) where T
    @assert length(rvec) == 2 * length(cvec)
    k = 1
    @inbounds for i in eachindex(cvec)
        cvec[i] = Complex(rvec[k], rvec[k + 1])
        k += 2
    end
    return cvec
end

# symmetric/Hermitian matrices and svec rescalings

"""
$(SIGNATURES)

The dimension of the vectorized triangle of a real symmetric matrix with side
dimension `side::Int`.
"""
svec_length(side::Int) = div(side * (side + 1), 2)

"""
$(SIGNATURES)

The dimension of the real vectorized triangle of a real symmetric or complex
Hermitian matrix with side dimension `side::Int`.
"""
svec_length(::Type{<:Real}, side::Int) = svec_length(side)

svec_length(::Type{<:Complex}, side::Int) = side^2

"""
$(SIGNATURES)

The side dimension of a real symmetric matrix with vectorized triangle length
`len::Int`.
"""
function svec_side(len::Int)
    side = div(isqrt(1 + 8 * len), 2)
    @assert side * (side + 1) == 2 * len
    return side
end

"""
$(SIGNATURES)

The side dimension of a real symmetric or complex Hermitian matrix with real
vectorized triangle length  `len::Int`.
"""
svec_side(::Type{<:Real}, len::Int) = svec_side(len)

function svec_side(::Type{<:Complex}, len::Int)
    side = isqrt(len)
    @assert side^2 == len
    return side
end

"""
$(SIGNATURES)

The index in the vectorized triangle of a symmetric matrix for element
(`row::Int`, `col::Int`).
"""
function svec_idx(row::Int, col::Int)
    if row < col
        (row, col) = (col, row)
    end
    return div((row - 1) * row, 2) + col
end

"""
$(SIGNATURES)

The indices corresponding to block `block::Int` in a vector of blocks with equal
length `incr::Int`.
"""
block_idxs(incr::Int, block::Int) = (incr * (block - 1) .+ (1:incr))

"""
$(SIGNATURES)

Rescale the elements corresponding to off-diagonals in `arr::AbstractVecOrMat`,
with scaling `scal::Real` and default block increment `incr::Int = 1`.
"""
function scale_svec!(
    arr::AbstractVecOrMat,
    scal::Real;
    incr::Int = 1,
    )
    @assert incr > 0
    n = size(arr, 1)
    (d, r) = divrem(n, incr)
    @assert iszero(r)
    side = svec_side(d)
    k = 1
    for i in 1:side
        for j in 1:(i - 1)
            @inbounds @views @. arr[k:(k + incr - 1), :] *= scal
            k += incr
        end
        k += incr
    end
    @assert k == 1 + n
    return arr
end

"""
$(SIGNATURES)

Copy a real symmetric matrix upper triangle to a svec-scaled vector in-place.
"""
function smat_to_svec!(
    vec::AbstractVector{T},
    mat::AbstractMatrix{T},
    rt2::Real,
    ) where T
    k = 1
    m = size(mat, 1)
    @assert m == size(mat, 2)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            vec[k] = mat[i, j]
        else
            vec[k] = mat[i, j] * rt2
        end
        k += 1
    end
    @assert k == length(vec) + 1
    return vec
end

"""
$(SIGNATURES)

Copy a complex Hermitian matrix upper triangle to a svec-scaled real vector
in-place.
"""
function smat_to_svec!(
    vec::AbstractVector{T},
    mat::AbstractMatrix{Complex{T}},
    rt2::Real,
    ) where T
    k = 1
    m = size(mat, 1)
    @assert m == size(mat, 2)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            vec[k] = real(mat[i, j])
            k += 1
        else
            ck = mat[i, j] * rt2
            vec[k] = real(ck)
            k += 1
            vec[k] = -imag(ck)
            k += 1
        end
    end
    @assert k == length(vec) + 1
    return vec
end

"""
$(SIGNATURES)

Copy a svec-scaled vector to a real symmetric matrix upper triangle in-place.
"""
function svec_to_smat!(
    mat::AbstractMatrix{T},
    vec::AbstractVector{T},
    rt2::Real,
    ) where T
    k = 1
    m = size(mat, 1)
    @assert m == size(mat, 2)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            mat[i, j] = vec[k]
        else
            mat[i, j] = vec[k] / rt2
        end
        k += 1
    end
    @assert k == length(vec) + 1
    return mat
end

"""
$(SIGNATURES)

Copy a svec-scaled real vector to a complex Hermitian matrix upper triangle
in-place.
"""
function svec_to_smat!(
    mat::AbstractMatrix{Complex{T}},
    vec::AbstractVector{T},
    rt2::Real,
    ) where T
    k = 1
    m = size(mat, 1)
    @assert m == size(mat, 2)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = Complex(vec[k], -vec[k + 1]) / rt2
            k += 2
        end
    end
    @assert k == length(vec) + 1
    return mat
end

# Kronecker utilities

# compute the real symmetric Kronecker product of a matrix in-place
function symm_kron!(
    skr::AbstractMatrix{T},
    mat::AbstractMatrix{T},
    rt2::T,
    ) where {T <: Real}
    side = size(mat, 1)

    col_idx = 1
    @inbounds for l in 1:side
        for k in 1:(l - 1)
            row_idx = 1
            for j in 1:side
                for i in 1:(j - 1)
                    skr[row_idx, col_idx] =
                        mat[i, k] * mat[j, l] + mat[i, l] * mat[j, k]
                    row_idx += 1
                end
                skr[row_idx, col_idx] = rt2 * mat[j, k] * mat[j, l]
                row_idx += 1
                (row_idx > col_idx) && break
            end
            col_idx += 1
        end

        row_idx = 1
        for j in 1:side
            for i in 1:(j - 1)
                skr[row_idx, col_idx] = rt2 * mat[i, l] * mat[j, l]
                row_idx += 1
            end
            skr[row_idx, col_idx] = abs2(mat[j, l])
            row_idx += 1
            (row_idx > col_idx) && break
        end
        col_idx += 1
    end

    return skr
end

# compute the complex Hermitian Kronecker product of a matrix in-place
function symm_kron!(
    skr::AbstractMatrix{T},
    mat::AbstractMatrix{Complex{T}},
    rt2::T,
    ) where {T <: Real}
    side = size(mat, 1)

    col_idx = 1
    @inbounds for l in 1:side
        for k in 1:(l - 1)
            row_idx = 1
            for j in 1:side
                for i in 1:(j - 1)
                    a = mat[i, k] * mat[l, j]
                    b = mat[j, k] * mat[l, i]
                    spectral_kron_element!(skr, row_idx, col_idx, a, b)
                    row_idx += 2
                end
                c = rt2 * mat[j, k] * mat[l, j]
                skr[row_idx, col_idx] = real(c)
                skr[row_idx, col_idx + 1] = imag(c)
                row_idx += 1
                (row_idx > col_idx) && break
            end
            col_idx += 2
        end

        row_idx = 1
        for j in 1:side
            for i in 1:(j - 1)
                c = rt2 * mat[i, l] * mat[l, j]
                skr[row_idx, col_idx] = real(c)
                skr[row_idx + 1, col_idx] = -imag(c)
                row_idx += 2
            end
            skr[row_idx, col_idx] = abs2(mat[j, l])
            row_idx += 1
            (row_idx > col_idx) && break
        end
        col_idx += 1
    end

    return skr
end

# compute an element of the real spectral Kronecker in-place
function spectral_kron_element!(
    skr::AbstractMatrix{T},
    i::Int,
    j::Int,
    a::T,
    b::T,
    ) where {T <: Real}
    @inbounds skr[i, j] = a + b
    return skr
end

# compute an element of the complex spectral Kronecker in-place
function spectral_kron_element!(
    skr::AbstractMatrix{T},
    i::Int,
    j::Int,
    a::Complex{T},
    b::Complex{T},
    ) where {T <: Real}
    apb = a + b
    amb = a - b
    @inbounds begin
        skr[i, j] = real(apb)
        skr[i + 1, j] = -imag(amb)
        skr[i, j + 1] = imag(apb)
        skr[i + 1, j + 1] = real(amb)
    end
    return skr
end

# compute a real symmetric Kronecker-like outer product of a real or complex
# matrix of eigenvectors and a real symmetric matrix
function eig_dot_kron!(
    skr::AbstractMatrix{T},
    inner::Matrix{T},
    vecs::Matrix{R},
    temp1::Matrix{R},
    temp2::Matrix{R},
    temp3::Matrix{R},
    V::Matrix{R},
    rt2::T,
    ) where {T <: Real, R <: RealOrComplex{T}}
    @assert issymmetric(inner) # must be symmetric (wrapper is less efficient)
    rt2i = inv(rt2)
    d = size(inner, 1)
    copyto!(V, vecs') # allows fast column slices
    V_views = [view(V, :, i) for i in 1:size(inner, 1)]
    scals = (R <: Complex{T} ? (rt2i, rt2i * im) : (rt2i,)) # real and imag parts

    col_idx = 1
    @inbounds for (j, V_j) in enumerate(V_views)
        for i in 1:(j - 1), scal in scals
            mul!(temp3, V_j, V_views[i]', scal, false)
            @. temp2 = inner * (temp3 + temp3')
            mul!(temp1, Hermitian(temp2, :U), V)
            mul!(temp2, V', temp1)
            @views smat_to_svec!(skr[:, col_idx], temp2, rt2)
            col_idx += 1
        end

        mul!(temp2, V_j, V_j')
        temp2 .*= inner
        mul!(temp1, Hermitian(temp2, :U), V)
        mul!(temp2, V', temp1)
        @views smat_to_svec!(skr[:, col_idx], temp2, rt2)
        col_idx += 1
    end

    return skr
end
