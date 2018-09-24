
# nonnegative orthant cone
# barrier is -sum_j ln x_j
# from Nesterov & Todd "Self-Scaled Barriers and Interior-Point Methods for Convex Programming"
mutable struct NonnegativeCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    invpnt::Vector{Float64}

    function NonnegativeCone(dim::Int)
        prm = new()
        prm.dim = dim
        prm.invpnt = Vector{Float64}(undef, dim)
        return prm
    end
end

dimension(prm::NonnegativeCone) = prm.dim
barrierpar_prm(prm::NonnegativeCone) = prm.dim
getintdir_prm!(arr::AbstractVector{Float64}, prm::NonnegativeCone) = (arr .= 1.0; arr)
loadpnt_prm!(prm::NonnegativeCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)
incone_prm(prm::NonnegativeCone) = all(x -> (x > 0.0), prm.pnt)

function calcg_prm!(g::AbstractVector{Float64}, prm::NonnegativeCone)
    prm.invpnt = inv.(prm.pnt)
    g .= -1.0 * prm.invpnt
    return g
end

calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::NonnegativeCone) = (prod .= abs2.(prm.pnt) .* arr; prod)
calcHarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::NonnegativeCone) = (prod .= abs2.(prm.invpnt) .* arr; prod)
