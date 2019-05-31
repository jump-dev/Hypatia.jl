#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points Ps

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO
- can perform loop for calculating g and H in parallel
- scale the interior direction
- check if gradient and hessian are correct for complex case
=#

mutable struct WSOSPolyInterp{T <: HypReal, R <: HypRealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    Ps::Vector{Matrix{R}}

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    Hi::Matrix{T}
    F # TODO prealloc
    tmpLL::Vector{Matrix{R}}
    tmpUL::Vector{Matrix{R}}
    tmpLU::Vector{Matrix{R}}
    tmpUU::Matrix{R}
    ΛFs::Vector

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

function setup_data(cone::WSOSPolyInterp{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
    dim = cone.dim
    cone.g = Vector{T}(undef, dim)
    cone.H = similar(cone.g, dim, dim)
    cone.H2 = similar(cone.H)
    cone.Hi = similar(cone.H)
    Ps = cone.Ps
    cone.tmpLL = [Matrix{R}(undef, size(Pk, 2), size(Pk, 2)) for Pk in Ps]
    cone.tmpUL = [Matrix{R}(undef, dim, size(Pk, 2)) for Pk in Ps]
    cone.tmpLU = [Matrix{R}(undef, size(Pk, 2), dim) for Pk in Ps]
    cone.tmpUU = Matrix{R}(undef, dim, dim)
    cone.ΛFs = Vector{Any}(undef, length(Ps))
    return
end

get_nu(cone::WSOSPolyInterp) = sum(size(Pk, 2) for Pk in cone.Ps)

set_initial_point(arr::AbstractVector{T}, cone::WSOSPolyInterp{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal} = (@. arr = one(T); arr)

function check_in_cone(cone::WSOSPolyInterp{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
    Ps = cone.Ps
    LLs = cone.tmpLL
    ULs = cone.tmpUL
    LUs = cone.tmpLU
    UU = cone.tmpUU
    ΛFs = cone.ΛFs
    D = Diagonal(cone.point)

    for k in eachindex(Ps) # TODO can be done in parallel
        # Λ = Pk' * Diagonal(point) * Pk
        # TODO LDLT calculation could be faster
        # TODO mul!(A, B', Diagonal(x)) calls extremely inefficient method but don't need ULk
        Pk = Ps[k]
        ULk = ULs[k]
        LLk = LLs[k]
        mul!(ULk, D, Pk)
        mul!(LLk, Pk', ULk)

        # pivoted cholesky and triangular solve method
        ΛFk = hyp_chol!(Hermitian(LLk, :L))
        if !isposdef(ΛFk)
            return false
        end
        ΛFs[k] = ΛFk
    end

    g = cone.g
    H = cone.H
    @. g = zero(T)
    @. H = zero(T)

    for k in eachindex(Ps) # TODO can be done in parallel, but need multiple tmp3s
        # TODO may be faster (but less numerically stable) with explicit inverse here
        LUk = hyp_ldiv_chol_L!(LUs[k], ΛFs[k], Ps[k]')
        hyp_AtA!(UU, LUk)

        for j in eachindex(g)
            @inbounds g[j] -= real(UU[j, j])
            for i in 1:j
                @inbounds H[i, j] += abs2(UU[i, j])
            end
        end
    end

    return factorize_hess(cone)
end
