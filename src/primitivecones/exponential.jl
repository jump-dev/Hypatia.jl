
# exponential cone (MathOptInterface definition)
# z >= y*exp(x/y), y > 0
# barrier from Skajaa & Ye 2014 is
# -log (y log (z/y) - x) - log z - log y
mutable struct ExponentialCone <: PrimitiveCone
    pnt::AbstractVector{Float64}
    g::AbstractVector{Float64}
    Hi::Symmetric{Float64,Array{Float64,2}} # TODO could be faster as StaticArray

    function ExponentialCone()
        prm = new()
        prm.g = Vector{Float64}(undef, 3)
        prm.Hi = Symmetric(similar(prm.g, 3, 3))
        return prm
    end
end

dimension(prm::ExponentialCone) = 3
barrierpar_prm(prm::ExponentialCone) = 3.0
getintdir_prm!(arr::AbstractVector{Float64}, prm::ExponentialCone) = (arr[1] = 0.0; arr[2] = 0.5; arr[3] = 1.0; arr)
loadpnt_prm!(prm::ExponentialCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::ExponentialCone)
    x = prm.pnt[1]; y = prm.pnt[2]; z = prm.pnt[3]
    if (y < 1e-9) || (z < 1e-12)
        return false
    end

    ylzy = y*log(z/y)
    dist = ylzy - x
    if dist <= 0.0
        return false
    end
    invdist = inv(dist)

    # gradient
    g = prm.g
    g[1] = invdist # 1/(-x + y log(z/y))
    g[2] = invdist * (y - x - 2*dist) / y # (x + y - 2 y log(z/y))/(y (-x + y log(z/y)))
    g[3] = (-1 - y*invdist) / z # (-1 + y/(x - y log(z/y)))/z

    # Hessian inverse
    HiU = prm.Hi.data # upper triangle
    den = 2*y + dist
    invden = inv(den)
    # TODO combine more repeated subexpressions
    HiU[1,1] = -(-2*ylzy^3 + (4*x - y)*abs2(ylzy) + (-3*abs2(x) + 2*y*x - 2*abs2(y))*ylzy + x*(abs2(x) - 2*y*x + 2*abs2(y))) * invden # (-2 y^3 log^3(z/y) + (4 x - y) y^2 log^2(z/y) + y (-3 x^2 + 2 y x - 2 y^2) log(z/y) + x (x^2 - 2 y x + 2 y^2))/(x - 2 y - y log(z/y))
    HiU[1,2] = y * (abs2(ylzy) - x*ylzy + x*y) * invden  # (y^2 (y log^2(z/y) - x log(z/y) + x))/(-x + 2 y + y log(z/y))
    HiU[1,3] = y * z * (2*ylzy - x) * invden # (y z (2 y log(z/y) - x))/(-x + 2 y + y log(z/y))
    HiU[2,2] = abs2(y) * (1 - y*invden) # (y^2 (-x + y + y log(z/y)))/(-x + 2 y + y log(z/y))
    HiU[2,3] = abs2(y) * z * invden # (y^2 z)/(-x + 2 y + y log(z/y))
    HiU[3,3] = abs2(z) * (1 - y*invden) # (z^2 (-x + y + y log(z/y)))/(-x + 2 y + y log(z/y)))

    return true
end

calcg_prm!(g::AbstractVector{Float64}, prm::ExponentialCone) = (g .= prm.g; g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::ExponentialCone) = mul!(prod, prm.Hi, arr)
