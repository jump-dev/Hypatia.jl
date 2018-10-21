#=
Copyright 2018, Chris Coey and contributors

power cone parametrized by powers vector α belonging to the unit simplex
(u, v) : abs(u) <= prod_i v_i^alpha_i, v >= 0
barrier from Roy & Xiao 2018 (theorem 1) is
-log(prod_i v_i^(2*alpha_i) - u^2) - sum_i (1 - alpha_i) log(v_i)
=#

mutable struct PowerCone <: PrimitiveCone
    dim::Int
    alpha::Vector{Float64}
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64} # TODO could be faster as StaticArray
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function PowerCone(alpha::Vector{Float64})
        prmtv = new()
        dim = length(alpha) + 1
        @assert dim >= 3
        @assert all(ai >= 0.0 for ai in alpha)
        @assert sum(alpha) == 1.0
        prmtv.dim = dim
        prmtv.alpha = alpha
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        function barfun(pnt)
            u = pnt[1]
            v = view(pnt, 2:dim)
            return -log(prod(v[i]^(2.0*alpha[i]) for i in eachindex(alpha)) - abs2(u)) - sum((1.0 - alpha[i])*log(v[i]) for i in eachindex(alpha))
        end
        prmtv.barfun = barfun
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

dimension(prmtv::PowerCone) = prmtv.dim
barrierpar_prmtv(prmtv::PowerCone) = prmtv.dim
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::PowerCone) = (@. arr = 1.0; arr[1] = 0.0; arr)
loadpnt_prmtv!(prmtv::PowerCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::PowerCone)
    if any(prmtv.pnt[i+1] <= 0.0 for i in 1:prmtv.dim-1)
        return false
    end
    if prod(prmtv.pnt[i+1]^prmtv.alpha[i] for i in 1:prmtv.dim-1) <= abs(prmtv.pnt[1]) # TODO may be better to check this in log-space
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.pnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)

    @. prmtv.H2 = prmtv.H
    prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
    return issuccess(prmtv.F)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::PowerCone) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::PowerCone) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::PowerCone) = mul!(prod, prmtv.H, arr)
