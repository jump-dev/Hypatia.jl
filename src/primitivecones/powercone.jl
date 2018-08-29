
# power cone (MathOptInterface definition) parametrized by power α
# x^α * y^(1-α) >= abs(z), x,y >= 0
# barrier from Skajaa & Ye 2014 is
# -log((x^α * y^(1-α))^2 - z^2) - (1-α)*log(x) - α*log(y)
mutable struct PowerCone <: PrimitiveCone
    exponent::Float64
    pnt::AbstractVector{Float64}
    g::AbstractVector{Float64}
    Hi::Symmetric{Float64,Array{Float64,2}} # TODO could be faster as StaticArray

    function PowerCone(exponent::Float64)
        prm = new()
        prm.exponent = exponent
        prm.g = Vector{Float64}(undef, 3)
        prm.Hi = Symmetric(similar(prm.g, 3, 3))
        return prm
    end
end

dimension(prm::PowerCone) = 3
barrierpar_prm(prm::PowerCone) = 3.0
getintdir_prm!(arr::AbstractVector{Float64}, prm::PowerCone) = (arr[1] = 1.0; arr[2] = 1.0; arr[3] = 0.0; arr)
loadpnt_prm!(prm::PowerCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::PowerCone)
    x = prm.pnt[1]; y = prm.pnt[2]; z = prm.pnt[3]
    if (x <= 0.0) || (y <= 0.0)
        return false
    end
    α = prm.exponent

    lhs = x^α * y^(1-α)
    dist = lhs - abs(z)
    if dist <= 0.0
        return false
    end

    # gradient
    g = prm.g
    g[1] = # (α - (2 α y^2 x^(2 α))/(y^2 x^(2 α) - z^2 y^(2 α)) - 1)/x
    g[2] = # ((α - 2) y^2 x^(2 α) + α z^2 y^(2 α))/(y (y^2 x^(2 α) - z^2 y^(2 α)))
    g[3] = # (2 z)/(x^(2 α) y^(2 - 2 α) - z^2)

    # Hessian inverse
    HiU = prm.Hi.data # upper triangle

    HiU[1,1] =  # ((x^2 (2 y^(2 α + 2) z^2 (2 α^2 - 3 α + 1) x^(2 α) + y^4 (α - 2) x^(4 α) + y^(4 α) z^4 α))/(2 y^(2 α + 2) z^2 (1 - 2 α)^2 x^(2 α) + y^4 (α^2 - α - 2) x^(4 α) - y^(4 α) z^4 (α - 1) α)
    HiU[1,2] =  # (4 x^(2 α + 1) y^(2 α + 3) z^2 (α - 1) α)/(2 y^(2 α + 2) z^2 (1 - 2 α)^2 x^(2 α) + y^4 (α^2 - α - 2) x^(4 α) - y^(4 α) z^4 (α - 1) α)
    HiU[1,3] =  # (2 x^(2 α + 1) y^2 z α (y^2 (α - 2) x^(2 α) + y^(2 α) z^2 α))/(2 y^(2 α + 2) z^2 (1 - 2 α)^2 x^(2 α) + y^4 (α^2 - α - 2) x^(4 α) - y^(4 α) z^4 (α - 1) α)

    HiU[2,2] =  # -(-2 y^(2 α + 4) z^2 α (2 α - 1) x^(2 α) + y^6 (α + 1) x^(4 α) + y^(4 α + 2) z^4 (α - 1))/(2 y^(2 α + 2) z^2 (1 - 2 α)^2 x^(2 α) + y^4 (α^2 - α - 2) x^(4 α) - y^(4 α) z^4 (α - 1) α)
    HiU[2,3] =  # (2 x^(2 α) y^3 z (α - 1) (y^2 (α + 1) x^(2 α) + y^(2 α) z^2 (α - 1)))/(2 y^(2 α + 2) z^2 (1 - 2 α)^2 x^(2 α) + y^4 (α^2 - α - 2) x^(4 α) - y^(4 α) z^4 (α - 1) α)

    HiU[3,3] =  # (y^(4 α + 2) z^4 (11 α^2 - 11 α + 2) x^(2 α) - 3 y^(2 α + 4) z^2 (α - 1) α x^(4 α) + y^6 (α^2 - α - 2) x^(6 α) - y^(6 α) z^6 (α - 1) α)/(4 y^(4 α + 2) z^2 (1 - 2 α)^2 x^(2 α) + 2 y^(2 α + 4) (α^2 - α - 2) x^(4 α) - 2 y^(6 α) z^4 (α - 1) α))

    return true
end

calcg_prm!(g::AbstractVector{Float64}, prm::PowerCone) = (g .= prm.g; g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::PowerCone) = mul!(prod, prm.Hi, arr)
