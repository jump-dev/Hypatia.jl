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
    Z::Hermitian{R,Matrix{R}}
    fact_Z
    Zi::Hermitian{R, Matrix{R}}
    ZiW::Matrix{R}
    tmpmm::Matrix{R}
    tmpnn::Matrix{R}

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
    cone.ZiW = Matrix{R}(undef, n, m)
    cone.tmpmm = Matrix{R}(undef, m, m)
    cone.tmpnn = Matrix{R}(undef, n, n)
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
        @views U = svec_to_smat!(cone.U.data, cone.point[1:(cone.per_idx - 1)], cone.rt2)
        @views W = vec_copy_to!(cone.W[:], cone.point[(cone.per_idx + 1):end])
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
    dim = cone.dim
    v = cone.point[cone.per_idx]

    Zi = cone.Zi = Hermitian(inv(cone.fact_Z), :U)
    @views smat_to_svec!(cone.grad[1:(cone.per_idx - 1)], Zi, cone.rt2)
    @views cone.grad[1:(cone.per_idx - 1)] .*= -2v
    cone.grad[cone.per_idx] = -2 * dot(Zi, U) + (cone.n - 1) / v
    ldiv!(cone.ZiW, cone.fact_Z, W)
    @views vec_copy_to!(cone.grad[(cone.per_idx + 1):dim], cone.ZiW[:])
    @. @views cone.grad[(cone.per_idx + 1):dim] *= 2

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::MatrixEpiPerSquare)
    @assert cone.grad_updated
    n = cone.n
    m = cone.m
    dim = cone.dim
    per_idx = cone.per_idx
    U = cone.U
    W = cone.W
    v = cone.point[cone.per_idx]
    H = cone.hess.data
    tmpmm = cone.tmpmm
    tmpnn = cone.tmpnn
    Zi = cone.Zi
    ZiW = cone.ZiW

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
    @views H[(per_idx + 1):dim, (per_idx + 1):dim] .*= 2

    # H_U_U part
    @views _symm_kron(H[1:(per_idx - 1), 1:(per_idx - 1)], Zi, cone.rt2)
    @. @views H[1:(per_idx - 1), 1:(per_idx - 1)] *= 4 * abs2(v)

    # H_v_v part
    @views H[per_idx, per_idx] = sum((Zi * U * Zi) .* U) * 4 - (cone.n - 1) / v / v

    # H_U_W part
    row_idx = 1
    for i in 1:n, j in 1:i
        col_idx = per_idx + 1
        for l in 1:m, k in 1:n
            H[row_idx, col_idx] = Zi[i, k] * ZiW[j, l] + Zi[k, j] * ZiW[i, l]
            if i != j
                H[row_idx, col_idx] *= cone.rt2
            end
            col_idx += 1
        end
        row_idx += 1
    end
    @. @views H[1:(per_idx - 1), (per_idx + 1):dim] *= -2v

    # H_U_v part
    mul!(tmpnn, U, Zi)
    ldiv!(cone.fact_Z, tmpnn)
    ZiUZi = tmpnn

    # H_v_W part
    # NOTE ZiW is overwritten
    mul!(ZiW, ZiUZi, W)
    @. ZiW *= -4
    @views vec_copy_to!(H[per_idx, (per_idx + 1):dim], ZiW[:])

    @. ZiUZi *= 2v
    @. ZiUZi -= Zi
    @. ZiUZi *= 2
    @views smat_to_svec!(H[1:(per_idx - 1), per_idx], ZiUZi, cone.rt2)

    cone.hess_updated = true
    return cone.hess
end
