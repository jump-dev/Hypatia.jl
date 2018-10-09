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
        prm = new()
        prm.g = Vector{Float64}(undef, 3)
        prm.H = similar(prm.g, 3, 3)
        prm.H2 = copy(prm.H)
        return prm
    end
end

dimension(prm::ExponentialCone) = 3
barrierpar_prm(prm::ExponentialCone) = 3
getintdir_prm!(arr::AbstractVector{Float64}, prm::ExponentialCone) = (arr[1] = 0.0; arr[2] = 0.5; arr[3] = 1.0; arr)
loadpnt_prm!(prm::ExponentialCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::ExponentialCone)
    x = prm.pnt[1]; y = prm.pnt[2]; z = prm.pnt[3]
    if (y < 1e-9) || (z < 1e-12)
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
    g = prm.g
    g[1] = iylzyx # 1/(-x + y log(z/y))
    g[2] = iylzyx * (y - x - 2*ylzyx) / y # (x + y - 2 y log(z/y))/(y (-x + y log(z/y)))
    g[3] = (-1 - y*iylzyx) / z # (-1 + y/(x - y log(z/y)))/z

    # Hessian
    yz = y/z
    H = prm.H
    H[1,1] = abs2(iylzyx)
    H[1,2] = H[2,1] = -(lzy - 1.0)*H[1,1]
    H[1,3] = H[3,1] = -yz*H[1,1]
    H[2,2] = abs2(lzy - 1.0)*H[1,1] + iylzyx/y + inv(abs2(y))
    H[2,3] = H[3,2] = yz*(lzy - 1.0)*H[1,1] - iylzyx/z
    H[3,3] = abs2(yz)*H[1,1] + yz/z*iylzyx + inv(abs2(z))

    @. prm.H2 = prm.H
    prm.F = cholesky!(Symmetric(prm.H2), Val(true), check=false) # bunchkaufman if it fails
    if !isposdef(prm.F)
        @. prm.H2 = prm.H
        prm.F = bunchkaufman!(Symmetric(prm.H2), true, check=false)
        return issuccess(prm.F)
    end
    return true
end

calcg_prm!(g::AbstractVector{Float64}, prm::ExponentialCone) = (@. g = prm.g; g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::ExponentialCone) = ldiv!(prod, prm.F, arr)
calcHarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::ExponentialCone) = mul!(prod, prm.H, arr)
