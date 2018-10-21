#=
Copyright 2018, Chris Coey and contributors

log-sum-exp cone
(z, y, x) : z >= y*sum_i exp(x_i/y), y >= 0
barrier???
=#

mutable struct LogSumExpCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64} # TODO could be faster as StaticArray
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function LogSumExpCone(dim::Int)
        @assert dim >= 4
        prmtv = new()
        prmtv.dim = dim
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        function barfun(zyx) # TODO inplace ops
            z = zyx[1]; y = zyx[2]; x = zyx[3:dim]
            n = length(x)
            # invy = inv(y)
            # w = [exp((xi - z)*invy) for xi in x]
            # w *= y/sum(w)
            # return -sum(log(z + y*log(w[i]) - x[i] - y*log(y)) + log(w[i]) + log(y) for i in eachindex(w))
            invy = inv(y)
            w = [exp((xi - z)*invy) for xi in x]
            sumw = sum(w)
            # lws = [(xi - z)*invy - log(sumw) for xi in x]
            # return -sum(log(z + y*lws[i] - x[i]) for i in eachindex(w)) - sum(lws) - 2.0*n*log(y)
            # return -sum(log(z + y*((x[i] - z)*invy - log(sumw)) - x[i]) for i in eachindex(w)) - sum(lws) - 2.0*n*log(y)
            # return -n*log(-y*log(sumw)) - sum(lws) - 2.0*n*log(y)
            # return -n*log(-log(sumw)) - sum(lws) - 3.0*n*log(y)
            return -n*log(-log(sumw)) - 3.0*n*log(y) - sum((xi - z)*invy for xi in x) + n*log(sumw)
        end
        prmtv.barfun = barfun
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

dimension(prmtv::LogSumExpCone) = prmtv.dim
barrierpar_prmtv(prmtv::LogSumExpCone) = 3*(prmtv.dim - 2)
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::LogSumExpCone) = (@. arr = 0.0; arr[1] = 2*log(prmtv.dim); arr[2] = 1.0; arr) # TODO change this to balance norm of initial s and z
loadpnt_prmtv!(prmtv::LogSumExpCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::LogSumExpCone)
    z = prmtv.pnt[1]; y = prmtv.pnt[2]; x = view(prmtv.pnt, 3:prmtv.dim)
    if y <= 0.0
        return false
    end
    invy = inv(y)
    if 1.0 <= sum(exp((xi - z)*invy) for xi in x)
        return false
    end

    @show z
    @show y
    @show x
    invy = inv(y)
    w = [exp(xi - z) for xi in x]
    w *= y/sum(w)
    @show w


    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.pnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)

    # @show prmtv.g
    # @show prmtv.H

    @. prmtv.H2 = prmtv.H
    prmtv.F = cholesky!(Symmetric(prmtv.H2), Val(true), check=false) # bunchkaufman if it fails
    if !isposdef(prmtv.F)
        @. prmtv.H2 = prmtv.H
        prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
        return issuccess(prmtv.F)
    end
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::LogSumExpCone) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::LogSumExpCone) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::LogSumExpCone) = mul!(prod, prmtv.H, arr)
