#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation matrices Ps

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO
- perform loop for calculating g and H in parallel
- scale the interior direction
=#

mutable struct WSOSInterpNonnegative2{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    initial_point::Vector{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    P::Matrix{R}
    Ls
    gs
    point::Vector{T}
    dual_point::Vector{T}

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

    tmpLL::Vector{Matrix{R}}
    tmpUL::Vector{Matrix{R}}
    tmpLU::Vector{Matrix{R}}
    tmpUU::Matrix{R}
    ΛF::Vector

    correction::Vector{T}

    # ideally input would have gs being polys then we have a general method for finding init support points and get the float64 gs, infer Ls
    function WSOSInterpNonnegative2{T, R}(
        initial_point::Vector{T},
        P::Matrix{R},
        Ls,
        gs;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        cone = new{T, R}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = length(initial_point)
        cone.initial_point = initial_point
        cone.P = P
        cone.Ls = Ls
        cone.gs = gs
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_data(cone::WSOSInterpNonnegative2{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    Ls = cone.Ls
    cone.tmpLL = [Matrix{R}(undef, L, L) for L in Ls]
    cone.tmpUL = [Matrix{R}(undef, dim, L) for L in Ls]
    cone.tmpLU = [Matrix{R}(undef, L, dim) for L in Ls]
    cone.tmpUU = Matrix{R}(undef, dim, dim)
    cone.ΛF = Vector{Any}(undef, length(Ls))
    cone.correction = zeros(T, dim)
    return
end

use_correction(::WSOSInterpNonnegative2) = false

get_nu(cone::WSOSInterpNonnegative2) = sum(cone.Ls)

# NOTE old procedure to get initial point. if support points equal the interp points used to get P the initial point is the ones vector
# function set_initial_point(arr::AbstractVector, cone::WSOSInterpNonnegative2)
#     interp_pts = cone.interp_pts
#     init_support_pts = cone.init_support_pts
#     (U, n) = size(interp_pts)
#     # could use fewer than U pts as support points (they don't have to be the same as the interp points) but easy enough to reuse
#     lagrange_polys = recover_lagrange_polys(interp_pts, 2 * cone.halfdeg)
#     for i in 1:U
#         @show typeof(init_support_pts[1, :])
#         arr[i] = sum(lagrange_polys[i]((init_support_pts[j, :]...)) for j in 1:size(init_support_pts, 1))
#     end
#     return arr
# end

set_initial_point(arr::AbstractVector, cone::WSOSInterpNonnegative2) = (arr .= cone.initial_point; arr)

# TODO order the k indices so that fastest and most recently infeasible k are first
# TODO can be done in parallel
function update_feas(cone::WSOSInterpNonnegative2)
    @assert !cone.feas_updated
    D = Diagonal(cone.point)

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ls)
        L = cone.Ls[k]
        g = cone.gs[k]
        # Λ = Pk' * Diagonal(point) * Pk
        # TODO mul!(A, B', Diagonal(x)) calls extremely inefficient method but doesn't need ULk
        @views Pk = cone.P[:, 1:L]
        ULk = cone.tmpUL[k]
        LLk = cone.tmpLL[k]
        D2 = Diagonal(D * g)
        mul!(ULk, D2, Pk)
        mul!(LLk, Pk', ULk)

        ΛFk = cholesky!(Hermitian(LLk, :L), check = false)
        if !isposdef(ΛFk)
            cone.is_feas = false
            break
        end
        cone.ΛF[k] = ΛFk
    end

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::WSOSInterpNonnegative2) = true

# TODO decide whether to compute the LUk' * LUk in grad or in hess (only diag needed for grad)
# TODO can be done in parallel
# TODO may be faster (but less numerically stable) with explicit inverse here
function update_grad(cone::WSOSInterpNonnegative2)
    @assert cone.is_feas

    cone.grad .= 0
    @inbounds for k in eachindex(cone.Ls)
        L = cone.Ls[k]
        g = cone.gs[k]
        @views Pk = cone.P[:, 1:L]
        LUk = cone.tmpLU[k]
        ldiv!(LUk, cone.ΛF[k].L, Pk')
        @inbounds for j in 1:cone.dim
            cone.grad[j] -= sum(abs2, view(LUk, :, j)) * g[j]
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpNonnegative2)
    @assert cone.grad_updated

    cone.hess .= 0
    @inbounds for k in eachindex(cone.Ls)
        g = cone.gs[k]
        LUk = cone.tmpLU[k]
        UUk = mul!(cone.tmpUU, LUk', LUk)
        @inbounds for j in 1:cone.dim, i in 1:j
            cone.hess.data[i, j] += abs2(UUk[i, j]) * g[i] * g[j]
        end
    end

    cone.hess_updated = true
    return cone.hess
end
