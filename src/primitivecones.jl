
#=
predefined standard primitive cone types
=#

abstract type PrimitiveCone end

# nonnegative orthant cone
mutable struct NonnegCone <: PrimitiveCone
    dim::Int
    pnt # maybe don't need to ever update this - it always points to the same view of same array

    function NonnegCone(dim::Int)
        prm = new()
        prm.dim = dim
        return prm
    end
end

dimension(prm::NonnegCone) = prm.dim
barrierpar_prm(prm::NonnegCone) = prm.dim
loadpnt_prm!(prm::NonnegCone, pnt) = (prm.pnt = pnt)
incone_prm(prm::NonnegCone) = all(x -> (x > 0.0), prm.pnt)
calcg_prm!(g, prm::NonnegCone) = (g .= -inv.(prm.pnt)) # TODO use view
calcHiprod_prm!(prod, arr, prm::NonnegCone) = (prod .= abs2.(prm.pnt) .* arr)
calcLiprod_prm!(prod, arr, prm::NonnegCone) = (prod .= prm.pnt .* arr)

# polynomial (weighted) sum of squares cone (parametrized by ip and ipwt)
mutable struct SumOfSquaresCone <: PrimitiveCone
    dim::Int
    ip::Matrix{Float64}
    ipwt::Vector{Matrix{Float64}}
    pnt
    g::Vector{Float64}
    F

    function SumOfSquaresCone(dim::Int, ip::Matrix{Float64}, ipwt::Vector{Matrix{Float64}})
        prm = new()
        prm.dim = dim
        prm.ip = ip
        prm.ipwt = ipwt
        # prm.g = similar(prm.pnt)
        # TODO prealloc etc
        return prm
    end
end

dimension(prm::SumOfSquaresCone) = prm.dim
barrierpar_prm(prm::SumOfSquaresCone) = size(prm.ip, 2) + sum(size(ipwtj, 2) for ipwtj in prm.ipwt)
loadpnt_prm!(prm::SumOfSquaresCone, pnt) = (prm.pnt = pnt)

function incone_prm(prm::SumOfSquaresCone)
    F = cholesky!(Symmetric(prm.ip'*Diagonal(prm.pnt)*prm.ip), check=false) # TODO do inplace. TODO could this cholesky of P'DP be faster?
    if !issuccess(F)
        return false
    end
    Vp = F.L\prm.ip' # TODO prealloc
    VtVp = Vp'*Vp
    g = -diag(VtVp)
    H = VtVp.^2

    for j in 1:length(prm.ipwt)
        F = cholesky!(Symmetric(prm.ipwt[j]'*Diagonal(prm.pnt)*prm.ipwt[j]), check=false)
        if !issuccess(F)
            return false
        end
        Vp = F.L\prm.ipwt[j]'
        VtVp = Vp'*Vp
        g .-= diag(VtVp)
        H .+= VtVp.^2
    end

    F = cholesky!(H, check=false)
    if !issuccess(F)
        return false
    end

    prm.g = g
    prm.F = F
    return true
end

calcg_prm!(g, prm::SumOfSquaresCone) = (g .= prm.g)
calcHiprod_prm!(prod, arr, prm::SumOfSquaresCone) = (prod .= prm.F.U\(prm.F.L\arr)) # TODO do in-place
calcLiprod_prm!(prod, arr, prm::SumOfSquaresCone) = (prod .= prm.F.L\arr)
