#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

matrix epigraph of matrix square

TODO describe
=#

mutable struct MatrixEpiPerSquare{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    n::Int
    m::Int
    per_idx::Int
    is_complex::Bool
    point::Vector{T}
    rt2::T
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    U
    W
    Z
    fact_Z

    function MatrixEpiPerSquare{T, R}(
        n::Int,
        m::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert 1 <= n <= m
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.is_complex = (R <: Complex)
        cone.per_idx = (cone.is_complex ? n ^ 2 + 1 : svec_length(n) + 1)
        cone.dim = cone.per_idx + (cone.is_complex ? 2 : 1) * n * m
        cone.n = n
        cone.m = m
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

MatrixEpiPerSquare{T, R}(n::Int, m::Int) where {R <: RealOrComplex{T}} where {T <: Real} = MatrixEpiPerSquare{T, R}(n, m, false)

# TODO only allocate the fields we use
function setup_data(cone::MatrixEpiPerSquare{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    n = cone.n
    m = cone.m
    cone.U = Hermitian(zeros(R, n, n), :U)
    cone.W = zeros(R, n, m)
    cone.Z = Hermitian(zeros(R, n, n), :U)
    return
end

get_nu(cone::MatrixEpiPerSquare) = cone.n + 1

function set_initial_point(arr::AbstractVector, cone::MatrixEpiPerSquare{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    @inbounds for i in 1:cone.n
        arr[k] = 1
        k += incr * i + 1
    end
    arr[cone.per_idx] = 1
    return arr
end

function update_feas(cone::MatrixEpiPerSquare)
    @assert !cone.feas_updated
    v = cone.point[cone.per_idx]

    if v > 0
        U = cone.U
        @views svec_to_smat!(U.data, cone.point[1:(cone.per_idx - 1)], cone.rt2)
        W = cone.W
        @views vec_copy_to!(W[:], cone.point[(cone.per_idx + 1):end])

        # TODO check posdef of U first? not necessary, but if need fact of U then may as well
        copyto!(cone.Z.data, U)
        mul!(cone.Z.data, cone.W, cone.W', -1, 2 * v)
        cone.fact_Z = cholesky!(cone.Z, check = false)
        cone.is_feas = isposdef(cone.fact_Z)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::MatrixEpiPerSquare)
    @assert cone.is_feas
    U = cone.U
    W = cone.W
    v = cone.point[cone.per_idx]

    Zi = inv(cone.fact_Z)
    cone.grad[cone.per_idx] = -2 * sum(Zi .* U) + (cone.n - 1) / v
    @views smat_to_svec!(cone.grad[1:(cone.per_idx - 1)], -2 .* Zi .* v, cone.rt2)
    cone.grad[(cone.per_idx + 1):end] .= 2 * vec(Zi * cone.W)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::MatrixEpiPerSquare)
    @assert cone.grad_updated
    n = cone.n
    m = cone.m
    per_idx = cone.per_idx
    U = cone.U
    W = cone.W
    v = cone.point[cone.per_idx]
    H = cone.hess.data

    Zi = inv(cone.fact_Z)
    ZiW = cone.fact_Z \ W
    tmpmm = zeros(m, m)

    # H_W_W part
    mul!(tmpmm, W', ZiW) # TODO Hermitian? W' * Zi * W
    tmpmm += I # TODO inefficient

    # TODO parallelize loops
    idx_incr = (cone.is_complex ? 2 : 1)
    r_idx = per_idx + 1
    for i in 1:m, j in 1:n
        c_idx = r_idx
        @inbounds for k in i:m
            ZiWjk = ZiW[j, k]
            tmpmmik = tmpmm[i, k]
            lstart = (i == k ? j : 1)
            @inbounds for l in lstart:n
                term1 = Zi[l, j] * tmpmmik
                term2 = ZiW[l, i] * ZiWjk
                _hess_WW_element(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
        r_idx += idx_incr
    end
    H[(per_idx + 1):end, (per_idx + 1):end] .*= 2

    # H_U_U part
    @views _build_hess(H[1:(per_idx - 1), 1:(per_idx - 1)], Zi, cone.rt2)
    H[1:(per_idx - 1), 1:(per_idx - 1)] .*= 4 * v ^ 2

    # H_v_v part
    H[per_idx, per_idx] = sum((Zi * U * Zi) .* U) * 4 - (cone.n - 1) / v / v

    # H_U_W part
    # row_idx = 1
    # Zi2 = Zi^2
    # ZiW = Zi * W
    # for i in 1:n, j in 1:i
    #     col_idx = per_idx + 1
    #     for l in 1:m, k in 1:n
    #         H[row_idx, col_idx] = 0
    #         for q in 1:n
    #             if (i == j) && (k == q)
    #                 H[row_idx, col_idx] += Zi[i, j] * Zi[k, q] * W[q, l]
    #                 # H[row_idx, col_idx] += abs2(Zi[i, j]) * W[q, l]
    #             elseif i != j
    #                 H[row_idx, col_idx] += Zi[i, j] * Zi[k, q] * W[q, l] * cone.rt2
    #             elseif (i != j) && (k != q)
    #                 H[row_idx, col_idx] += Zi[i, j] * Zi[k, q] * W[q, l]
    #                 # H[row_idx, col_idx] += (Zi[i, k] * Zi[q, j] + Zi[j, k] * Zi[q, i]) * W[q, l]
    #             else
    #                 H[row_idx, col_idx] += Zi[i, j] * Zi[k, q] * W[q, l]
    #                 # H[row_idx, col_idx] += cone.rt2 * Zi[k, i] * Zi[j, q] * W[q, l]
    #             end
    #         end
    #         col_idx += 1
    #     end
    #     row_idx += 1
    # end
    # H[1:(per_idx - 1), (per_idx + 1):end] .*= -4 * v

    col_idx = per_idx + 1
    for l in 1:m, k in 1:n
        mat1 = zeros(n, n)
        mat1[k, :] += vec(W[:, l])
        mat1[:, k] += vec(W[:, l])
        mat = Zi * mat1 * Zi
        @views smat_to_svec!(H[1:(per_idx - 1), col_idx], mat, cone.rt2)
        col_idx += 1
    end
    H[1:(per_idx - 1), (per_idx + 1):end] .*= -2 * v


    # if (i == j) && (i2 == j2)
    #     H[k2, k] = abs2(mat[i2, i])
    # elseif (i != j) && (i2 != j2)
    #     H[k2, k] = mat[i2, i] * mat[j, j2] + mat[j2, i] * mat[j, i2]
    # else
    #     H[k2, k] = rt2 * mat[i2, i] * mat[j, j2]
    # end
    # if k2 == k
    #     break
    # end

    # H_U_v part
    mat = (Zi .* v * U .* 2 * Zi .- Zi) .* 2
    @views smat_to_svec!(H[1:(per_idx - 1), per_idx], mat, cone.rt2)

    # H_v_W part
    H[per_idx, (per_idx + 1):end] = -2 * Zi * U * Zi * W * 2

    cone.hess_updated = true
    return cone.hess
end
