#=
utilities for arrays
=#

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

# symmetric/svec rescalings

"""
$(SIGNATURES)

Compute the number of elements in the triangle of a real symmetric matrix with
side dimension `side::Int`.
"""
function svec_length(side::Int)
    (len, r) = divrem(side * (side + 1), 2)
    @assert iszero(r)
    return len
end

"""
$(SIGNATURES)

Compute the side dimension of a real symmetric matrix from the length `len::Int`
of its vectorized triangle.
"""
function svec_side(len::Int)
    side = round(Int, sqrt(0.25 + 2 * len) - 0.5)
    @assert side * (side + 1) == 2 * len
    return side
end

"""
$(SIGNATURES)

Compute the index in the vectorized triangle of a symmetric matrix for element
(`row::Int`, `col::Int`).
"""
svec_idx(row::Int, col::Int) = (svec_length(row - 1) + col)

"""
$(SIGNATURES)

Compute the indices corresponding to block `block::Int` in a vector of blocks
with equal length `incr::Int`.
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

"""
$(SIGNATURES)

Compute the real symmetric Kronecker product of a matrix in-place.
"""
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

"""
$(SIGNATURES)

Compute the complex Hermitian Kronecker product of a matrix in-place.
"""
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

"""
$(SIGNATURES)

Compute an element of the real spectral Kronecker in-place.
"""
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

"""
$(SIGNATURES)

Compute an element of the complex spectral Kronecker in-place.
"""
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
