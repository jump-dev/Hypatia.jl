#=
Copyright 2018, Chris Coey and contributors

exponential cone
(u, v, w) : u >= v*exp(w/v), v >= 0
barrier from Skajaa & Ye 2014 is
-log (v log (u/v) - w) - log u - log v
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
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::ExponentialCone) = (arr[1] = 1.0; arr[2] = 0.5; arr[3] = 0.0; arr)
loadpnt_prmtv!(prmtv::ExponentialCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::ExponentialCone)
    u = prmtv.pnt[1]; v = prmtv.pnt[2]; w = prmtv.pnt[3]
    if (v <= 0.0) || (u <= 0.0)
        return false
    end

    luv = log(u/v)
    vluv = v*luv
    vluvw = vluv - w
    if vluvw <= 0.0
        return false
    end

    # gradient
    ivluvw = inv(vluvw)
    g = prmtv.g
    g[1] = -(1.0 + v*ivluvw)/u
    g[2] = ivluvw*(v - w - 2.0*vluvw)/v
    g[3] = ivluvw

    # Hessian
    vu = v/u
    H = prmtv.H
    H[3,3] = abs2(ivluvw)
    H[3,2] = H[2,3] = -(luv - 1.0)*H[3,3]
    H[3,1] = H[1,3] = -vu*H[3,3]
    H[2,2] = abs2(luv - 1.0)*H[3,3] + ivluvw/v + inv(abs2(v))
    H[2,1] = H[1,2] = vu*(luv - 1.0)*H[3,3] - ivluvw/u
    H[1,1] = abs2(vu)*H[3,3] + vu/u*ivluvw + inv(abs2(u))

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
