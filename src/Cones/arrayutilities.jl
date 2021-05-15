#=
utilities for arrays
=#

# real and complex vectors

vec_copy_to!(
    v1::AbstractVecOrMat{T},
    v2::AbstractVecOrMat{T},
    ) where {T <: Real} = copyto!(v1, v2)

vec_copy_to!(
    v1::AbstractVecOrMat{T},
    v2::AbstractVecOrMat{Complex{T}},
    ) where {T <: Real} = cvec_to_rvec!(v1, v2)

vec_copy_to!(
    v1::AbstractVecOrMat{Complex{T}},
    v2::AbstractVecOrMat{T},
    ) where {T <: Real} = rvec_to_cvec!(v1, v2)

function rvec_to_cvec!(
    cvec::AbstractVecOrMat{Complex{T}},
    rvec::AbstractVecOrMat{T},
    ) where T
    k = 1
    @inbounds for i in eachindex(cvec)
        cvec[i] = Complex(rvec[k], rvec[k + 1])
        k += 2
    end
    return cvec
end

function cvec_to_rvec!(
    rvec::AbstractVecOrMat{T},
    cvec::AbstractVecOrMat{Complex{T}},
    ) where T
    k = 1
    @inbounds for i in eachindex(cvec)
        ci = cvec[i]
        rvec[k] = real(ci)
        rvec[k + 1] = imag(ci)
        k += 2
    end
    return rvec
end


# symmetric/svec rescalings

svec_length(side::Int) = div(side * (side + 1), 2)

svec(row::Int, col::Int) = (div((row - 1) * row, 2) + col)

block_idxs(incr::Int, block::Int) = (incr * (block - 1) .+ (1:incr))

function vec_to_svec!(
    arr::AbstractVecOrMat{T};
    scal::T = sqrt(T(2)),
    incr::Int = 1,
    ) where T
    n = size(arr, 1)
    @assert iszero(rem(n, incr))
    side = round(Int, sqrt(0.25 + 2 * div(n, incr)) - 0.5)
    k = 1
    for i in 1:side
        @inbounds @views for j in 1:(i - 1)
            @. arr[k:(k + incr - 1), :] *= scal
            k += incr
        end
        k += incr
    end
    @assert k == 1 + n
    return arr
end

svec_to_vec!(arr::AbstractVecOrMat{T}; incr::Int = 1) where T =
    vec_to_svec!(arr, scal = inv(sqrt(T(2))), incr = incr)

function smat_to_svec!(
    vec::AbstractVector{T},
    mat::AbstractMatrix{T},
    rt2::Number,
    ) where T
    k = 1
    m = size(mat, 1)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            vec[k] = mat[i, j]
        else
            vec[k] = mat[i, j] * rt2
        end
        k += 1
    end
    return vec
end

function smat_to_svec!(
    vec::AbstractVector{T},
    mat::AbstractMatrix{Complex{T}},
    rt2::Number,
    ) where T
    k = 1
    m = size(mat, 1)
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
    return vec
end

function svec_to_smat!(
    mat::AbstractMatrix{T},
    vec::AbstractVector{T},
    rt2::Number,
    ) where T
    k = 1
    m = size(mat, 1)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            mat[i, j] = vec[k]
        else
            mat[i, j] = vec[k] / rt2
        end
        k += 1
    end
    return mat
end

function svec_to_smat!(
    mat::AbstractMatrix{Complex{T}},
    vec::AbstractVector{T},
    rt2::Number,
    ) where T
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = Complex(vec[k], -vec[k + 1]) / rt2
            k += 2
        end
    end
    return mat
end

# kronecker utilities

function symm_kron!(
    H::AbstractMatrix{T},
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
                    H[row_idx, col_idx] = mat[i, k] * mat[j, l] +
                        mat[i, l] * mat[j, k]
                    row_idx += 1
                end
                H[row_idx, col_idx] = rt2 * mat[j, k] * mat[j, l]
                row_idx += 1
                (row_idx > col_idx) && break
            end
            col_idx += 1
        end

        row_idx = 1
        for j in 1:side
            for i in 1:(j - 1)
                H[row_idx, col_idx] = rt2 * mat[i, l] * mat[j, l]
                row_idx += 1
            end
            H[row_idx, col_idx] = abs2(mat[j, l])
            row_idx += 1
            (row_idx > col_idx) && break
        end
        col_idx += 1
    end

    return H
end

function symm_kron!(
    H::Matrix{T},
    mat::AbstractMatrix{Complex{T}},
    rt2::T,
    ) where {T <: Real}
    side = size(mat, 1)

    col_idx = 1
    for i in 1:side, j in 1:i
        row_idx = 1
        if i == j
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    H[row_idx, col_idx] = abs2(mat[i2, i])
                    row_idx += 1
                else
                    c = rt2 * mat[i, i2] * mat[j2, j]
                    H[row_idx, col_idx] = real(c)
                    row_idx += 1
                    H[row_idx, col_idx] = -imag(c)
                    row_idx += 1
                end
                (row_idx > col_idx) && break
            end
            col_idx += 1
        else
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    c = rt2 * mat[i2, i] * mat[j, j2]
                    H[row_idx, col_idx] = real(c)
                    H[row_idx, col_idx + 1] = -imag(c)
                    row_idx += 1
                else
                    b1 = mat[i2, i] * mat[j, j2]
                    b2 = mat[j2, i] * mat[j, i2]
                    c1 = b1 + b2
                    H[row_idx, col_idx] = real(c1)
                    H[row_idx, col_idx + 1] = -imag(c1)
                    row_idx += 1
                    c2 = b1 - b2
                    H[row_idx, col_idx] = imag(c2)
                    H[row_idx, col_idx + 1] = real(c2)
                    row_idx += 1
                end
                (row_idx > col_idx) && break
            end
            col_idx += 2
        end
    end

    return H
end

function spectral_hess_element!(
    H::Matrix{T},
    i::Int,
    j::Int,
    a::T,
    b::T,
    ) where {T <: Real}
    @inbounds H[i, j] = a + b
    return H
end

function spectral_hess_element!(
    H::Matrix{T},
    i::Int,
    j::Int,
    a::Complex{T},
    b::Complex{T},
    ) where {T <: Real}
    @inbounds begin
        H[i, j] = real(a) + real(b)
        H[i + 1, j] = imag(b) - imag(a)
        H[i, j + 1] = imag(a) + imag(b)
        H[i + 1, j + 1] = real(a) - real(b)
    end
    return H
end
