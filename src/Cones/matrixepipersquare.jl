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


    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::MatrixEpiPerSquare)
    @assert cone.grad_updated


    cone.hess_updated = true
    return cone.hess
end
