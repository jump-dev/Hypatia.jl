#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation matrices Ps

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO
- perform loop for calculating g and H in parallel
- scale the interior direction
=#

mutable struct WSOSInterpNonnegative{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    Ps::Vector{Matrix{R}}
    point::Vector{T}

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

    tmpLL::Vector{Matrix{R}}
    tmpUL::Vector{Matrix{R}}
    tmpLU::Vector{Matrix{R}}
    tmpUU::Matrix{R}
    ΛF::Vector

    function WSOSInterpNonnegative{T, R}(
        U::Int,
        Ps::Vector{Matrix{R}},
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        for Pk in Ps
            @assert size(Pk, 1) == U
        end
        cone = new{T, R}()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

WSOSInterpNonnegative{T, R}(dim::Int, Ps::Vector{Matrix{R}}) where {R <: RealOrComplex{T}} where {T <: Real} = WSOSInterpNonnegative{T, R}(dim, Ps, false)

function setup_data(cone::WSOSInterpNonnegative{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    Ps = cone.Ps
    cone.tmpLL = [Matrix{R}(undef, size(Pk, 2), size(Pk, 2)) for Pk in Ps]
    cone.tmpUL = [Matrix{R}(undef, dim, size(Pk, 2)) for Pk in Ps]
    cone.tmpLU = [Matrix{R}(undef, size(Pk, 2), dim) for Pk in Ps]
    cone.tmpUU = Matrix{R}(undef, dim, dim)
    cone.ΛF = Vector{Any}(undef, length(Ps))
    return
end

get_nu(cone::WSOSInterpNonnegative) = sum(size(Pk, 2) for Pk in cone.Ps)

# TODO find "central" initial point, like for other cones
set_initial_point(arr::AbstractVector, cone::WSOSInterpNonnegative) = (arr .= 1)

# TODO order the k indices so that fastest and most recently infeasible k are first
# TODO can be done in parallel
function update_feas(cone::WSOSInterpNonnegative)
    @assert !cone.feas_updated
    D = Diagonal(cone.point)

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ps)
        # Λ = Pk' * Diagonal(point) * Pk
        # TODO mul!(A, B', Diagonal(x)) calls extremely inefficient method but doesn't need ULk
        Pk = cone.Ps[k]
        ULk = cone.tmpUL[k]
        LLk = cone.tmpLL[k]
        mul!(ULk, D, Pk)
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

# TODO decide whether to compute the LUk' * LUk in grad or in hess (only diag needed for grad)
# TODO can be done in parallel
# TODO may be faster (but less numerically stable) with explicit inverse here
function update_grad(cone::WSOSInterpNonnegative)
    @assert cone.is_feas

    cone.grad .= 0
    @inbounds for k in eachindex(cone.Ps)
        LUk = cone.tmpLU[k]
        ldiv!(LUk, cone.ΛF[k].L, cone.Ps[k]')
        @inbounds for j in 1:cone.dim
            cone.grad[j] -= sum(abs2, view(LUk, :, j))
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpNonnegative)
    @assert cone.grad_updated

    cone.hess .= 0
    @inbounds for k in eachindex(cone.Ps)
        LUk = cone.tmpLU[k]
        UUk = mul!(cone.tmpUU, LUk', LUk)
        @inbounds for j in 1:cone.dim, i in 1:j
            cone.hess.data[i, j] += abs2(UUk[i, j])
        end
    end

    cone.hess_updated = true
    return cone.hess
end
