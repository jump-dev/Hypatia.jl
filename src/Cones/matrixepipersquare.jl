#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

matrix epigraph of matrix square

(U, v, W) in (S_+^n, R_+, R^(n, m)) such that 2 * U * v - W * W' in S_+^n
=#

mutable struct MatrixEpiPerSquare{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    max_neighborhood::T
    use_heuristic_neighborhood::Bool
    dim::Int
    n::Int
    m::Int
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
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    U_idxs::UnitRange{Int}
    v_idx::Int
    W_idxs::UnitRange{Int}
    U::Hermitian{R,Matrix{R}}
    W::Matrix{R}
    Z::Hermitian{R,Matrix{R}}
    fact_Z
    Zi::Hermitian{R, Matrix{R}}
    ZiW::Matrix{R}
    tmpmm::Matrix{R}
    tmpnn::Matrix{R}

    function MatrixEpiPerSquare{T, R}(
        n::Int,
        m::Int;
        use_dual::Bool = false,
        max_neighborhood::Real = default_max_neighborhood(),
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert 1 <= n <= m
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.max_neighborhood = max_neighborhood
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.is_complex = (R <: Complex)
        cone.v_idx = (cone.is_complex ? n ^ 2 + 1 : svec_length(n) + 1)
        cone.dim = cone.v_idx + (cone.is_complex ? 2 : 1) * n * m
        cone.n = n
        cone.m = m
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        cone.U_idxs = 1:(cone.v_idx - 1)
        cone.W_idxs = (cone.v_idx + 1):cone.dim
        return cone
    end
end

# TODO only allocate the fields we use
function setup_data(cone::MatrixEpiPerSquare{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
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
    arr[cone.v_idx] = 1
    return arr
end

function update_feas(cone::MatrixEpiPerSquare)
    @assert !cone.feas_updated
    v = cone.point[cone.v_idx]

    if v > 0
        @views U = svec_to_smat!(cone.U.data, cone.point[cone.U_idxs], cone.rt2)
        @views W = vec_copy_to!(cone.W, cone.point[cone.W_idxs])
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
    v = cone.point[cone.v_idx]

    Zi = cone.Zi = Hermitian(inv(cone.fact_Z), :U)
    @views smat_to_svec!(cone.grad[cone.U_idxs], Zi, cone.rt2)
    @views cone.grad[cone.U_idxs] .*= -2v
    cone.grad[cone.v_idx] = -2 * dot(Zi, U) + (cone.n - 1) / v
    ldiv!(cone.ZiW, cone.fact_Z, W)
    @views vec_copy_to!(cone.grad[cone.W_idxs], cone.ZiW)
    @. @views cone.grad[cone.W_idxs] *= 2

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::MatrixEpiPerSquare)
    @assert cone.grad_updated
    n = cone.n
    m = cone.m
    dim = cone.dim
    U_idxs = cone.U_idxs
    v_idx = cone.v_idx
    W_idxs = cone.W_idxs
    U = cone.U
    W = cone.W
    v = cone.point[cone.v_idx]
    H = cone.hess.data
    tmpmm = cone.tmpmm
    tmpnn = cone.tmpnn
    Zi = cone.Zi
    ZiW = cone.ZiW
    idx_incr = (cone.is_complex ? 2 : 1)

    # H_W_W part
    mul!(tmpmm, W', ZiW)
    tmpmm += I # TODO inefficient

    # TODO parallelize loops
    r_idx = v_idx + 1
    for i in 1:m, j in 1:n
        c_idx = r_idx
        @inbounds for k in i:m
            ZiWjk = ZiW[j, k]
            tmpmmik = tmpmm[i, k]
            lstart = (i == k ? j : 1)
            @inbounds for l in lstart:n
                term1 = Zi[l, j] * tmpmmik
                term2 = ZiW[l, i] * ZiWjk
                hess_element(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
        r_idx += idx_incr
    end
    @views H[W_idxs, W_idxs] .*= 2

    # H_U_U part
    @views H_U_U = H[U_idxs, U_idxs]
    symm_kron(H_U_U, Zi, cone.rt2)
    @. @views H_U_U *= 4 * abs2(v)

    # H_v_v part
    ldiv!(tmpnn, cone.fact_Z, U)
    rdiv!(tmpnn, cone.fact_Z)
    ZiUZi = Hermitian(tmpnn)
    @views H[v_idx, v_idx] = 4 * dot(ZiUZi, U) - (cone.n - 1) / v / v

    # H_U_W part
    # TODO parallelize loops
    # TODO use dispatch for complex part and clean up
    row_idx = 1
    for i in 1:n, j in 1:i # U lower tri idxs
        col_idx = v_idx + 1
        for l in 1:m, k in 1:n # W idxs
            @inbounds if cone.is_complex
                term1 = Zi[k, i] * ZiW[j, l]
                term2 = Zi[k, j] * ZiW[i, l]
                if i != j
                    term1 *= cone.rt2
                    term2 *= cone.rt2
                end
                H[row_idx, col_idx] = real(term1) + real(term2)
                H[row_idx, col_idx + 1] = imag(term1) + imag(term2)
                if i != j
                    H[row_idx + 1, col_idx] = imag(term2) - imag(term1)
                    H[row_idx + 1, col_idx + 1] = real(term1) - real(term2)
                end
            else
                term = Zi[i, k] * ZiW[j, l] + Zi[k, j] * ZiW[i, l]
                if i != j
                    term *= cone.rt2
                end
                H[row_idx, col_idx] = term
            end
            col_idx += idx_incr
        end
        if i != j
            row_idx += idx_incr
        else
            row_idx += 1
        end
    end
    @. @views H[U_idxs, W_idxs] *= -2v

    # H_v_W part
    # NOTE overwrites ZiW
    # TODO better to do ZiU * ZiW?
    mul!(ZiW, ZiUZi, W, -4, false)
    @views vec_copy_to!(H[v_idx, W_idxs], ZiW)

    # H_U_v part
    # NOTE overwrites ZiUZi
    axpby!(-2, Zi, 4v, tmpnn)
    @views smat_to_svec!(H[U_idxs, v_idx], tmpnn, cone.rt2)

    cone.hess_updated = true
    return cone.hess
end
