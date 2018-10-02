#=
Copyright 2018, Chris Coey and contributors

power cone parametrized by powers vector α belonging to the unit simplex
(z, x) : prod_i x_i^alpha_i >= abs(z), x >= 0
barrier from Roy & Xiao 2018 (theorem 1) is
-log(prod_i x_i^(2*alpha_i) - z^2) - sum_i (1 - alpha_i) log(x_i)
=#

mutable struct PowerCone <: PrimitiveCone
    dim::Int
    alpha::Vector{Float64}
    barfun::Function
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    diffres
    H::Matrix{Float64} # TODO could be faster as StaticArray
    H2::Matrix{Float64}
    F

    function PowerCone(alpha::Vector{Float64})
        prm = new()
        dim = length(alpha) + 1
        @assert dim >= 3
        @assert all(ai >= 0.0 for ai in alpha)
        @assert sum(alpha) == 1.0
        prm.dim = dim
        prm.alpha = alpha
        prm.g = Vector{Float64}(undef, dim)
        prm.diffres = DiffResults.HessianResult(prm.g)
        prm.barfun = (pnt -> -log(prod(pnt[i+1]^(alpha[i] + alpha[i]) for i in 1:dim-1) - abs2(pnt[1])) - sum((1.0 - alpha[i])*log(pnt[i+1]) for i in 1:dim-1))
        prm.H = similar(prm.g, dim, dim)
        prm.H2 = copy(prm.H)
        return prm
    end
end

dimension(prm::PowerCone) = prm.dim
barrierpar_prm(prm::PowerCone) = prm.dim
getintdir_prm!(arr::AbstractVector{Float64}, prm::PowerCone) = (@. arr = 1.0; arr[1] = 0.0; arr)
loadpnt_prm!(prm::PowerCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::PowerCone)
    if any(prm.pnt[i+1] <= 0.0 for i in 1:prm.dim-1)
        return false
    end
    if prod(prm.pnt[i+1]^prm.alpha[i] for i in 1:prm.dim-1) <= abs(prm.pnt[1])
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults

    prm.diffres = ForwardDiff.hessian!(prm.diffres, prm.barfun, prm.pnt)

    prm.g .= DiffResults.gradient(prm.diffres)
    prm.H .= DiffResults.hessian(prm.diffres)

    @. prm.H2 = prm.H
    prm.F = cholesky!(Symmetric(prm.H2), check=false) # bunchkaufman if it fails
    if !issuccess(prm.F)
        @. prm.H2 = prm.H
        prm.F = bunchkaufman!(Symmetric(prm.H2))
    end

    return true
end

calcg_prm!(g::AbstractVector{Float64}, prm::PowerCone) = (@. g = prm.g; g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::PowerCone) = ldiv!(prod, prm.F, arr)
calcHarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::PowerCone) = mul!(prod, prm.H, arr)
