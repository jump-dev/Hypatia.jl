#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points Ps

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO
- can perform loop for calculating g and H in parallel
- scale the interior direction
=#

mutable struct WSOSPolyInterp{T <: HypReal, R <: HypRealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    Ps::Vector{Matrix{R}}
    point::AbstractVector{T}

    is_feas::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    tmpLL::Vector{Matrix{R}}
    tmpUL::Vector{Matrix{R}}
    tmpLU::Vector{Matrix{R}}
    tmpUU::Matrix{R}
    ΛFs::Vector
    F # TODO prealloc

    function WSOSPolyInterp{T, R}(dim::Int, Ps::Vector{Matrix{R}}, is_dual::Bool) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
        for k in eachindex(Ps)
            @assert size(Ps[k], 1) == dim
        end
        cone = new{T, R}()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.Ps = Ps
        return cone
    end
end

WSOSPolyInterp{T, R}(dim::Int, Ps::Vector{Matrix{R}}) where {R <: HypRealOrComplex{T}} where {T <: HypReal} = WSOSPolyInterp{T, R}(dim, Ps, false)

# TODO maybe only allocate the fields we use
function setup_data(cone::WSOSPolyInterp{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)

    Ps = cone.Ps
    cone.tmpLL = [Matrix{R}(undef, size(Pk, 2), size(Pk, 2)) for Pk in Ps]
    cone.tmpUL = [Matrix{R}(undef, dim, size(Pk, 2)) for Pk in Ps]
    cone.tmpLU = [Matrix{R}(undef, size(Pk, 2), dim) for Pk in Ps]
    cone.tmpUU = Matrix{R}(undef, dim, dim)
    cone.ΛFs = Vector{Any}(undef, length(Ps))
    return
end

get_nu(cone::WSOSPolyInterp) = sum(size(Pk, 2) for Pk in cone.Ps)

set_initial_point(arr::AbstractVector, cone::WSOSPolyInterp) = (arr .= 1)

reset_data(cone::WSOSPolyInterp) = (cone.is_feas = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO order the k indices so that fastest and most recently infeasible k are first
# TODO can be done in parallel
function update_feas(cone::WSOSPolyInterp)
    @assert !cone.is_feas
    D = Diagonal(cone.point)
    for k in eachindex(cone.Ps)
        # Λ = Pk' * Diagonal(point) * Pk
        # TODO mul!(A, B', Diagonal(x)) calls extremely inefficient method but doesn't need ULk
        Pk = cone.Ps[k]
        ULk = cone.tmpUL[k]
        LLk = cone.tmpLL[k]
        mul!(ULk, D, Pk)
        mul!(LLk, Pk', ULk)

        ΛFk = hyp_chol!(Hermitian(LLk, :L))
        if !isposdef(ΛFk)
            cone.is_feas = false
            break
        end
        cone.ΛFs[k] = ΛFk
    end
    cone.is_feas = true
    return cone.is_feas
end

# TODO decide whether to compute the hyp_AtA in grad or in hess (only diag needed for grad)
# TODO can be done in parallel
# TODO may be faster (but less numerically stable) with explicit inverse here
function update_grad(cone::WSOSPolyInterp)
    @assert cone.is_feas
    cone.grad .= 0
    for k in eachindex(cone.Ps)
        LUk = hyp_ldiv_chol_L!(cone.tmpLU[k], cone.ΛFs[k], cone.Ps[k]')
        for j in eachindex(cone.grad)
            cone.grad[j] -= sum(abs2, view(LUk, :, j))
        end
        # hyp_AtA!(UU, LUk)
        # for j in eachindex(cone.grad)
        #     cone.grad[j] -= real(UU[j, j])
        # end
    end
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSPolyInterp)
    @assert cone.grad_updated
    cone.hess .= 0
    for k in eachindex(cone.Ps)
        UUk = hyp_AtA!(cone.tmpUU[k], cone.tmpLU[k])
        cone.hess += abs2.(UUk)
        # for j in eachindex(g), i in 1:j
        #     cone.hess[i, j] += abs2(UU[i, j])
        # end
    end
    cone.hess_updated = true
    return cone.hess
end

# TODO? inv_hess and hess_prod! and inv_hess_prod!
