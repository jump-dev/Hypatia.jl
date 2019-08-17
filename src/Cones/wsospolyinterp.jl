#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points Ps

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO
- can perform loop for calculating g and H in parallel
- scale the interior direction
=#

mutable struct WSOSPolyInterp{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    Ps::Vector{Matrix{R}}
    point::AbstractVector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    tmpLL::Vector{Matrix{R}}
    tmpUL::Vector{Matrix{R}}
    tmpLU::Vector{Matrix{R}}
    tmpUU::Matrix{R}
    ΛFs::Vector
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact # TODO prealloc

    function WSOSPolyInterp{T, R}(dim::Int, Ps::Vector{Matrix{R}}, is_dual::Bool) where {R <: RealOrComplex{T}} where {T <: Real}
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

WSOSPolyInterp{T, R}(dim::Int, Ps::Vector{Matrix{R}}) where {R <: RealOrComplex{T}} where {T <: Real} = WSOSPolyInterp{T, R}(dim, Ps, false)

# TODO maybe only allocate the fields we use
function setup_data(cone::WSOSPolyInterp{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    Ps = cone.Ps
    cone.tmpLL = [Matrix{R}(undef, size(Pk, 2), size(Pk, 2)) for Pk in Ps]
    cone.tmpUL = [Matrix{R}(undef, dim, size(Pk, 2)) for Pk in Ps]
    cone.tmpLU = [Matrix{R}(undef, size(Pk, 2), dim) for Pk in Ps]
    cone.tmpUU = Matrix{R}(undef, dim, dim)
    cone.ΛFs = Vector{Any}(undef, length(Ps))
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

get_nu(cone::WSOSPolyInterp) = sum(size(Pk, 2) for Pk in cone.Ps)

set_initial_point(arr::AbstractVector, cone::WSOSPolyInterp) = (arr .= 1)

# TODO order the k indices so that fastest and most recently infeasible k are first
# TODO can be done in parallel
function update_feas(cone::WSOSPolyInterp)
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

        ΛFk = hyp_chol!(Hermitian(LLk, :L))
        if !isposdef(ΛFk)
            cone.is_feas = false
            break
        end
        cone.ΛFs[k] = ΛFk
    end
    cone.feas_updated = true
    return cone.is_feas
end

# TODO decide whether to compute the hyp_AtA in grad or in hess (only diag needed for grad)
# TODO can be done in parallel
# TODO may be faster (but less numerically stable) with explicit inverse here
function update_grad(cone::WSOSPolyInterp)
    @assert cone.is_feas
    cone.grad .= 0
    @inbounds for k in eachindex(cone.Ps)
        LUk = hyp_ldiv_chol_L!(cone.tmpLU[k], cone.ΛFs[k], cone.Ps[k]')
        @inbounds for j in 1:cone.dim
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
    @inbounds for k in eachindex(cone.Ps)
        UUk = hyp_AtA!(cone.tmpUU, cone.tmpLU[k])
        # cone.hess.data += abs2.(UUk)
        @inbounds for j in 1:cone.dim, i in 1:j
            cone.hess.data[i, j] += abs2(UUk[i, j])
        end
    end
    cone.hess_updated = true
    return cone.hess
end
