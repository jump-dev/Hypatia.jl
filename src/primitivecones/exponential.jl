#=
Copyright 2018, Chris Coey and contributors

exponential cone
(x, y, z) : z >= y*exp(x/y), y >= 0
barrier from Skajaa & Ye 2014 is
-log (y log (z/y) - x) - log z - log y
=#

mutable struct ExponentialCone <: PrimitiveCone
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64} # TODO could be faster as StaticArray
    H2::Matrix{Float64}
    F

    function ExponentialCone()
        prmtv = new()
        prmtv.g = Vector{Float64}(undef, 3)
        prmtv.H = similar(prmtv.g, 3, 3)
        prmtv.H2 = similar(prmtv.H)
        return prmtv
    end
end

dimension(prmtv::ExponentialCone) = 3
barrierpar_prmtv(prmtv::ExponentialCone) = 3
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::ExponentialCone) = (arr[1] = 0.0; arr[2] = 0.5; arr[3] = 1.0; arr) # TODO change this to balance norm of initial s and z
loadpnt_prmtv!(prmtv::ExponentialCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::ExponentialCone)
    x = prmtv.pnt[1]; y = prmtv.pnt[2]; z = prmtv.pnt[3]
    if y <= 0.0 || z <= 0.0
        return false
    end

    lzy = log(z/y)
    ylzy = y*lzy
    ylzyx = ylzy - x
    if ylzyx <= 0.0
        return false
    end

    # gradient
    iylzyx = inv(ylzyx)
    g = prmtv.g
    g[1] = iylzyx # 1/(-x + y log(z/y))
    g[2] = iylzyx * (y - x - 2*ylzyx) / y # (x + y - 2 y log(z/y))/(y (-x + y log(z/y)))
    g[3] = (-1 - y*iylzyx) / z # (-1 + y/(x - y log(z/y)))/z

    # Hessian
    yz = y/z
    H = prmtv.H
    H[1,1] = abs2(iylzyx)
    H[1,2] = H[2,1] = -(lzy - 1.0)*H[1,1]
    H[1,3] = H[3,1] = -yz*H[1,1]
    H[2,2] = abs2(lzy - 1.0)*H[1,1] + iylzyx/y + inv(abs2(y))
    H[2,3] = H[3,2] = yz*(lzy - 1.0)*H[1,1] - iylzyx/z
    H[3,3] = abs2(yz)*H[1,1] + yz/z*iylzyx + inv(abs2(z))

    @. prmtv.H2 = prmtv.H
    prmtv.F = cholesky!(Symmetric(prmtv.H2), Val(true), check=false) # bunchkaufman if it fails
    if !isposdef(prmtv.F)
        @. prmtv.H2 = prmtv.H
        prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
        return issuccess(prmtv.F)
    end
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::ExponentialCone) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::ExponentialCone) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::ExponentialCone) = mul!(prod, prmtv.H, arr)
