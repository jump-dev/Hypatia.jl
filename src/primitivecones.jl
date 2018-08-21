
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
    ippnt::Matrix{Float64}
    ipwtpnt::Vector{Matrix{Float64}}
    Vp::Matrix{Float64}
    Vpwt::Vector{Matrix{Float64}}
    VtVp::Matrix{Float64}
    gtmp::Vector{Float64}
    Htmp::Matrix{Float64}

    function SumOfSquaresCone(dim::Int, ip::Matrix{Float64}, ipwt::Vector{Matrix{Float64}})
        prm = new()
        prm.dim = dim
        prm.ip = ip
        prm.ipwt = ipwt
        prm.g = similar(ip, dim)
        prm.ippnt = similar(ip, size(ip, 2), size(ip, 2))
        prm.ipwtpnt = [similar(ip, size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        prm.Vp = similar(ip, size(ip, 2), dim)
        prm.Vpwt = [similar(ip, size(ipwtj, 2), dim) for ipwtj in ipwt]
        prm.VtVp = similar(ip, dim, dim)
        prm.gtmp = similar(ip, dim)
        prm.Htmp = similar(prm.VtVp)
        return prm
    end
end

dimension(prm::SumOfSquaresCone) = prm.dim
barrierpar_prm(prm::SumOfSquaresCone) = size(prm.ip, 2) + sum(size(ipwtj, 2) for ipwtj in prm.ipwt)
loadpnt_prm!(prm::SumOfSquaresCone, pnt) = (prm.pnt = pnt)

function incone_prm(prm::SumOfSquaresCone)
    prm.ippnt .= prm.ip'*Diagonal(prm.pnt)*prm.ip
    F = cholesky!(Symmetric(prm.ippnt), check=false) # TODO use structure cholesky of P'DP to speed up chol?
    if !issuccess(F)
        return false
    end
    prm.Vp .= F.L\prm.ip' # TODO do in-place
    prm.VtVp .= prm.Vp'*prm.Vp
    prm.gtmp .= -diag(prm.VtVp)
    prm.Htmp .= prm.VtVp.^2

    for j in 1:length(prm.ipwt)
        prm.ipwtpnt[j] .= prm.ipwt[j]'*Diagonal(prm.pnt)*prm.ipwt[j]
        F = cholesky!(Symmetric(prm.ipwtpnt[j]), check=false)
        if !issuccess(F)
            return false
        end
        prm.Vpwt[j] .= F.L\prm.ipwt[j]'
        prm.VtVp .= prm.Vpwt[j]'*prm.Vpwt[j]
        prm.gtmp .-= diag(prm.VtVp)
        prm.Htmp .+= prm.VtVp.^2
    end

    F = cholesky!(prm.Htmp, check=false)
    if !issuccess(F)
        return false
    end
    prm.g .= prm.gtmp
    prm.F = F
    return true
end

calcg_prm!(g, prm::SumOfSquaresCone) = (g .= prm.g)
calcHiprod_prm!(prod, arr, prm::SumOfSquaresCone) = (prod .= prm.F.U\(prm.F.L\arr)) # TODO do in-place
calcLiprod_prm!(prod, arr, prm::SumOfSquaresCone) = (prod .= prm.F.L\arr)
