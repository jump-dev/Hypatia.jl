#=
smat(w) in S_+^d intersected with w in R_+^(sdim(d))

barrier -logdet(W) - sum(log(W_ij) for i in 1:n, j in 1:(i-1))
where W = smat(w)

TODO
- improve description
=#

mutable struct DoublyNonnegativeTri{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    side::Int
    rt2::T

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    correction::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    mat::Matrix{T}
    mat2::Matrix{T} # TODO rename to imply mutates fact_mat
    mat3::Matrix{T}
    mat4::Matrix{T} # TODO could remove if we factorize mat instead of mat2, currently mat is not used in any other oracles
    offdiag_idxs
    inv_mat::Matrix{T}
    inv_vec::Vector{T}
    fact_mat

    function DoublyNonnegativeTri{T}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        cone.side = side = round(Int, sqrt(0.25 + 2 * dim) - 0.5)
        @assert side * (side + 1) == 2 * dim
        cone.offdiag_idxs = vcat([div(i * (i - 1), 2) .+ (1:(i - 1)) for i in 2:side]...)
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

reset_data(cone::DoublyNonnegativeTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

function setup_extra_data(cone::DoublyNonnegativeTri{T}) where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.mat = zeros(T, cone.side, cone.side)
    cone.mat2 = zero(cone.mat)
    cone.mat3 = zero(cone.mat)
    cone.mat4 = zero(cone.mat)
    cone.inv_vec = zeros(T, length(cone.offdiag_idxs))
    return cone
end

get_nu(cone::DoublyNonnegativeTri) = cone.dim

function set_initial_point(arr::AbstractVector{T}, cone::DoublyNonnegativeTri{T}) where T
    side = cone.side

    # for small side dimension, use closed-form solutions
    if side == 1
        on_diag = off_diag = one(T)
    elseif side == 2
        (on_diag, off_diag) = (sqrt(T(5)) / 2, inv(cone.rt2))
    else
        n = T(side)
        d = T(cone.dim)
        # root of this polynomial gives off-diagonal
        p1 = [-n - 1, 0, n ^ 2 + n + 7, 0, -2 * n ^ 2 - 8, 0, n ^ 2]
        # fallback values
        (on_diag, off_diag) = (n + 1, one(T))
        found_soln = false
        for offd_try in PolynomialRoots.roots(p1)
            offd_real = real(offd_try)
            # TODO this poly seems to always have real roots, prove
            if offd_real > 0
                # get trial point on the diagonal
                temp = d - (d - n) * abs2(offd_real)
                if temp > sqrt(eps(T))
                    ond_try = sqrt(temp / n)
                    denom = abs2(ond_try) + (n - 2) / cone.rt2 * ond_try * offd_real - (n - 1) * abs2(offd_real) / 2
                    # check s = -g(s) conditions
                    if ond_try * cone.rt2 + (n - 2) * offd_real ≈ ond_try * denom * cone.rt2 && denom ≈ abs2(offd_real) * (denom + 1)
                        (on_diag, off_diag) = (ond_try, offd_real)
                        found_soln = true
                        break
                    end
                end
            end
        end
        if !found_soln
            @warn("initial point inaccurate for DoublyNonnegativeTri cone dimension $(cone.dim)")
        end
    end

    arr .= off_diag
    k = 1
    @inbounds for i in 1:cone.side
        arr[k] = on_diag
        k += i + 1
    end

    return arr
end

function update_feas(cone::DoublyNonnegativeTri{T}) where T
    @assert !cone.feas_updated

    if all(>(eps(T)), cone.point)
        svec_to_smat!(cone.mat, cone.point, cone.rt2)
        copyto!(cone.mat2, cone.mat)
        cone.fact_mat = cholesky!(Symmetric(cone.mat2, :U), check = false)
        cone.is_feas = isposdef(cone.fact_mat)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::DoublyNonnegativeTri)
    @assert cone.is_feas

    cone.inv_mat = inv(cone.fact_mat) # TODO in-place
    smat_to_svec!(cone.grad, cone.inv_mat, cone.rt2)
    cone.grad .*= -1
    @. @views cone.inv_vec = inv(cone.point[cone.offdiag_idxs])
    @. @views cone.grad[cone.offdiag_idxs] -= cone.inv_vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::DoublyNonnegativeTri)
    @assert cone.grad_updated
    H = cone.hess.data

    symm_kron(H, cone.inv_mat, cone.rt2)
    for (inv_od, idx) in zip(cone.inv_vec, cone.offdiag_idxs)
        H[idx, idx] += abs2(inv_od)
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::DoublyNonnegativeTri)
    @assert is_feas(cone)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat3, view(arr, :, i), cone.rt2)
        copytri!(cone.mat3, 'U')
        rdiv!(cone.mat3, cone.fact_mat)
        ldiv!(cone.fact_mat, cone.mat3)
        smat_to_svec!(view(prod, :, i), cone.mat3, cone.rt2)
    end
    offdiags = cone.offdiag_idxs
    @views point_offdiags = cone.point[offdiags]
    @. @views prod[offdiags, :] += arr[offdiags, :] / point_offdiags / point_offdiags

    return prod
end

function correction(cone::DoublyNonnegativeTri, dir::AbstractVector)
    @assert cone.grad_updated

    S = copytri!(svec_to_smat!(cone.mat4, dir, cone.rt2), 'U')
    ldiv!(cone.fact_mat, S)
    rdiv!(S, cone.fact_mat.U)
    mul!(cone.mat3, S, S') # TODO use outer prod function
    smat_to_svec!(cone.correction, cone.mat3, cone.rt2)
    offdiags = cone.offdiag_idxs
    @views point_offdiags = cone.point[offdiags]
    @. @views cone.correction[offdiags] += abs2(dir[offdiags] / point_offdiags) / point_offdiags

    return cone.correction
end
