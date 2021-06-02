"""
$(TYPEDEF)

Dense Cholesky-based implementation for the sparse positive semidefinite
cone [`PosSemidefTriSparse`](@ref).
"""
struct PSDSparseDense <: PSDSparseImpl end

mutable struct PSDSparseDenseCache{T <: Real, R <: RealOrComplex{T}} <:
    PSDSparseCache{T, R}
    mat::Matrix{R}
    mat2::Matrix{R}
    inv_mat::Matrix{R}
    fact_mat
    PSDSparseDenseCache{T, R}() where {T <: Real, R <: RealOrComplex{T}} =
        new{T, R}()
end

function setup_extra_data!(
    cone::PosSemidefTriSparse{PSDSparseDense, T, R},
    ) where {T, R}
    cone.cache = cache = PSDSparseDenseCache{T, R}()
    cache.mat = zeros(R, cone.side, cone.side)
    cache.mat2 = zero(cache.mat)
    cache.inv_mat = zero(cache.mat)
    return
end

function update_feas(cone::PosSemidefTriSparse{PSDSparseDense})
    @assert !cone.feas_updated
    cache = cone.cache
    Λ = cache.mat

    svec_to_smat_sparse!(Λ, cone.point, cone)
    cache.fact_mat = cholesky!(Hermitian(Λ, :L), check = false)
    cone.is_feas = isposdef(cache.fact_mat)

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidefTriSparse{PSDSparseDense})
    @assert !cone.grad_updated && cone.is_feas
    cache = cone.cache
    inv_mat = cache.inv_mat

    inv_fact!(inv_mat, cache.fact_mat)
    copytri!(inv_mat, 'L', true)
    smat_to_svec_sparse!(cone.grad, inv_mat, cone)
    cone.grad .*= -1

    cone.grad_updated = true
    return cone.grad
end

function update_hess(
    cone::PosSemidefTriSparse{PSDSparseDense, T, T},
    ) where {T <: Real}
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    rt2 = cone.rt2
    H = cone.hess.data
    fill!(H, 0)
    Λi = cone.cache.inv_mat

    @inbounds for (idx2, (i2, j2)) in enumerate(zip(cone.row_idxs, cone.col_idxs))
        for idx1 in 1:idx2
            (i1, j1) = (cone.row_idxs[idx1], cone.col_idxs[idx1])
            H[idx1, idx2] = begin
                if (i1 == j1) && (i2 == j2)
                    abs2(Λi[i1, i2])
                elseif xor(i1 == j1, i2 == j2)
                    rt2 * Λi[i1, i2] * Λi[j1, j2]
                else
                    Λi[i1, i2] * Λi[j1, j2] + Λi[i1, j2] * Λi[j1, i2]
                end
            end
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function update_hess(
    cone::PosSemidefTriSparse{PSDSparseDense, T, Complex{T}},
    ) where {T <: Real}
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    rt2 = cone.rt2
    H = cone.hess.data
    fill!(H, 0)
    Λi = cone.cache.inv_mat

    idx2 = 1
    @inbounds for (i2, j2) in zip(cone.row_idxs, cone.col_idxs)
        idx1 = 1
        if i2 == j2
            for (i1, j1) in zip(cone.row_idxs, cone.col_idxs)
                if i1 == j1
                    H[idx1, idx2] = abs2(Λi[i1, i2])
                    idx1 += 1
                else
                    c = rt2 * Λi[i2, i1] * Λi[j1, j2]
                    H[idx1, idx2] = real(c)
                    idx1 += 1
                    H[idx1, idx2] = -imag(c)
                    idx1 += 1
                end
                (idx1 > idx2) && break
            end
            idx2 += 1
        else
            for (i1, j1) in zip(cone.row_idxs, cone.col_idxs)
                if i1 == j1
                    c = rt2 * Λi[i1, i2] * Λi[j2, j1]
                    H[idx1, idx2] = real(c)
                    H[idx1, idx2 + 1] = -imag(c)
                    idx1 += 1
                else
                    b1 = Λi[i1, i2] * Λi[j2, j1]
                    b2 = Λi[j1, i2] * Λi[j2, i1]
                    c1 = b1 + b2
                    H[idx1, idx2] = real(c1)
                    H[idx1, idx2 + 1] = -imag(c1)
                    idx1 += 1
                    c2 = b1 - b2
                    H[idx1, idx2] = imag(c2)
                    H[idx1, idx2 + 1] = real(c2)
                    idx1 += 1
                end
                (idx1 > idx2) && break
            end
            idx2 += 2
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod_slow!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTriSparse{PSDSparseDense},
    )
    cone.use_hess_prod_slow_updated || update_use_hess_prod_slow(cone)
    @assert cone.hess_updated
    cone.use_hess_prod_slow || return hess_prod!(prod, arr, cone)
    @assert is_feas(cone)
    cache = cone.cache

    @inbounds @views for j in 1:size(arr, 2)
        Λ = svec_to_smat_sparse!(cache.mat2, arr[:, j], cone)
        copytri!(Λ, 'L', true)
        ldiv!(cache.fact_mat, Λ)
        rdiv!(Λ, cache.fact_mat)
        smat_to_svec_sparse!(prod[:, j], Λ, cone)
    end

    return prod
end

function dder3(
    cone::PosSemidefTriSparse{PSDSparseDense},
    dir::AbstractVector,
    )
    @assert is_feas(cone)
    cache = cone.cache

    Λ = svec_to_smat_sparse!(cache.mat2, dir, cone)
    copytri!(Λ, 'L', true)
    ldiv!(cache.fact_mat.L, Λ)
    rdiv!(Λ, cache.fact_mat)
    outer_prod_vec_sparse!(cone.dder3, Λ, cone)

    return cone.dder3
end

function outer_prod_vec_sparse!(
    vec::AbstractVector{T},
    mat::AbstractMatrix{T},
    cone::PosSemidefTriSparse{PSDSparseDense, T, T},
    ) where {T <: Real}
    @assert length(vec) == length(cone.row_idxs)
    @inbounds for (idx, (i, j)) in enumerate(zip(cone.row_idxs, cone.col_idxs))
        @views x = dot(mat[:, i], mat[:, j])
        if i != j
            x *= cone.rt2
        end
        vec[idx] = x
    end
    return vec
end

function outer_prod_vec_sparse!(
    vec::AbstractVector{T},
    mat::AbstractMatrix{Complex{T}},
    cone::PosSemidefTriSparse{PSDSparseDense, T, Complex{T}},
    ) where {T <: Real}
    idx = 1
    @inbounds for (i, j) in zip(cone.row_idxs, cone.col_idxs)
        @views x = dot(mat[:, i], mat[:, j])
        if i == j
            vec[idx] = real(x)
            idx += 1
        else
            x *= cone.rt2
            vec[idx] = real(x)
            vec[idx + 1] = imag(x)
            idx += 2
        end
    end
    @assert idx == length(vec) + 1
    return vec
end

function svec_to_smat_sparse!(
    mat::AbstractMatrix{T},
    vec::AbstractVector{T},
    cone::PosSemidefTriSparse{PSDSparseDense, T, T},
    ) where {T <: Real}
    @assert length(vec) == length(cone.row_idxs)
    fill!(mat, 0)
    @inbounds for (idx, (i, j)) in enumerate(zip(cone.row_idxs, cone.col_idxs))
        x = vec[idx]
        if i != j
            x /= cone.rt2
        end
        mat[i, j] = x
    end
    return mat
end

function svec_to_smat_sparse!(
    mat::AbstractMatrix{Complex{T}},
    vec::AbstractVector{T},
    cone::PosSemidefTriSparse{PSDSparseDense, T, Complex{T}},
    ) where {T <: Real}
    fill!(mat, 0)
    idx = 1
    @inbounds for (i, j) in zip(cone.row_idxs, cone.col_idxs)
        if i == j
            mat[i, j] = vec[idx]
            idx += 1
        else
            mat[i, j] = Complex(vec[idx], vec[idx + 1]) / cone.rt2
            idx += 2
        end
    end
    @assert idx == length(vec) + 1
    return mat
end

function smat_to_svec_sparse!(
    vec::AbstractVector{T},
    mat::AbstractMatrix{T},
    cone::PosSemidefTriSparse{PSDSparseDense, T, T},
    ) where {T <: Real}
    @assert length(vec) == length(cone.row_idxs)
    fill!(vec, 0)
    @inbounds for (idx, (i, j)) in enumerate(zip(cone.row_idxs, cone.col_idxs))
        x = mat[i, j]
        if i != j
            x *= cone.rt2
        end
        vec[idx] = x
    end
    return vec
end

function smat_to_svec_sparse!(
    vec::AbstractVector{T},
    mat::AbstractMatrix{Complex{T}},
    cone::PosSemidefTriSparse{PSDSparseDense, T, Complex{T}},
    ) where {T <: Real}
    fill!(vec, 0)
    idx = 1
    @inbounds for (i, j) in zip(cone.row_idxs, cone.col_idxs)
        x = mat[i, j]
        if i == j
            vec[idx] = real(x)
            idx += 1
        else
            x *= cone.rt2
            vec[idx] = real(x)
            vec[idx + 1] = imag(x)
            idx += 2
        end
    end
    @assert idx == length(vec) + 1
    return vec
end
