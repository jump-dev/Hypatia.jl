
# nonnegative orthant cone
mutable struct NonnegativeCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}

    function NonnegativeCone(dim::Int)
        prm = new()
        prm.dim = dim
        return prm
    end
end

dimension(prm::NonnegativeCone) = prm.dim
barrierpar_prm(prm::NonnegativeCone) = prm.dim
getintdir_prm!(arr::AbstractVector{Float64}, prm::NonnegativeCone) = (arr .= 1.0; arr)
loadpnt_prm!(prm::NonnegativeCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)
incone_prm(prm::NonnegativeCone) = all(x -> (x > 0.0), prm.pnt)
calcg_prm!(g::AbstractVector{Float64}, prm::NonnegativeCone) = (g .= inv.(prm.pnt) .* -1.0; g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::NonnegativeCone) = (prod .= abs2.(prm.pnt) .* arr; prod)
