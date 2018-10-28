#=
Copyright 2018, Chris Coey and contributors

(u, v, W) : u >= v*logdet(W/v)

barrier
-log(v*logdet(W/v) - u) - logdet(W) - log(v)
which I just guessed, based on analogy to hypoperlog barrier

TODO use triangle with rescaling
=#

mutable struct HypoPerLogdet <: PrimitiveCone
    usedual::Bool
    dim::Int
    side::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function HypoPerLogdet(dim::Int, isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        side = round(Int, sqrt(dim - 2))
        prmtv.side = side
        prmtv.mat = Matrix{Float64}(undef, side, side)
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        function barfun(pnt)
            u = pnt[1]
            v = pnt[2]
            W = reshape(pnt[3:end], side, side)
            return -log(v*logdet(W/v) - u) - logdet(W) - log(v)
        end
        prmtv.barfun = barfun
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

HypoPerLogdet(dim::Int) = HypoPerLogdet(dim, false)

dimension(prmtv::HypoPerLogdet) = prmtv.dim
barrierpar_prmtv(prmtv::HypoPerLogdet) = prmtv.side + 2

function getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::HypoPerLogdet)
    arr[1] = -1.0
    arr[2] = 1.0
    arr[3:end] = vec(Matrix(1.0I, prmtv.side, prmtv.side))
    return arr
end

loadpnt_prmtv!(prmtv::HypoPerLogdet, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::HypoPerLogdet)
    pnt = prmtv.pnt
    u = pnt[1]
    v = pnt[2]
    W = reshape(pnt[3:end], prmtv.side, prmtv.side)
    if u >= v*logdet(W/v)
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

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::HypoPerLogdet) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::HypoPerLogdet) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::HypoPerLogdet) = mul!(prod, prmtv.H, arr)
