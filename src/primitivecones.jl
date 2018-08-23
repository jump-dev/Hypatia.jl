
#=
predefined standard primitive cone types
=#

abstract type PrimitiveCone end

# nonnegative orthant cone
mutable struct NonnegCone <: PrimitiveCone
    dim::Int
    pnt

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
calcg_prm!(g, prm::NonnegCone) = (g .= inv.(prm.pnt) .* -1.0)
calcHiprod_prm!(prod, arr, prm::NonnegCone) = (prod .= abs2.(prm.pnt) .* arr)
calcLiprod_prm!(prod, arr, prm::NonnegCone) = (prod .= prm.pnt .* arr)

# polynomial (weighted) sum of squares cone (parametrized by ip and ipwt)
mutable struct SumOfSquaresCone <: PrimitiveCone
    dim::Int
    ipwt::Vector{Matrix{Float64}}
    pnt
    g::Vector{Float64}
    H::Matrix{Float64}
    F
    ipwtpnt::Vector{Matrix{Float64}}
    Vp::Vector{Matrix{Float64}}
    Vp2::Matrix{Float64}

    function SumOfSquaresCone(dim::Int, ipwt::Vector{Matrix{Float64}})
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == dim
        end
        prm = new()
        prm.dim = dim
        prm.ipwt = ipwt
        prm.g = similar(ipwt[1], dim)
        prm.H = similar(ipwt[1], dim, dim)
        prm.ipwtpnt = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        prm.Vp = [similar(ipwt[1], dim, size(ipwtj, 2)) for ipwtj in ipwt]
        prm.Vp2 = similar(ipwt[1], dim, dim)
        return prm
    end
end

dimension(prm::SumOfSquaresCone) = prm.dim
barrierpar_prm(prm::SumOfSquaresCone) = sum(size(ipwtj, 2) for ipwtj in prm.ipwt)
loadpnt_prm!(prm::SumOfSquaresCone, pnt) = (prm.pnt = pnt)

function incone_prm(prm::SumOfSquaresCone)
    prm.g .= 0.0
    prm.H .= 0.0

    for j in eachindex(prm.ipwt) # TODO can do this loop in parallel (use separate Vp2[j])
        # prm.ipwtpnt[j] .= prm.ipwt[j]'*Diagonal(prm.pnt)*prm.ipwt[j]
        mul!(prm.Vp[j], Diagonal(prm.pnt), prm.ipwt[j])
        mul!(prm.ipwtpnt[j], prm.ipwt[j]', prm.Vp[j])

        F = cholesky!(Symmetric(prm.ipwtpnt[j]), check=false)
        if !issuccess(F)
            return false
        end

        prm.Vp[j] .= prm.ipwt[j]/F.U # TODO in-place syntax should work but ldiv! is very broken for triangular matrices in 0.7
        mul!(prm.Vp2, prm.Vp[j], prm.Vp[j]') # TODO if parallel, need to use separate Vp2[j] # TODO this is much slower than it should be on 0.7

        prm.g .-= diag(prm.Vp2)
        prm.H .+= abs2.(prm.Vp2)
    end

    prm.F = cholesky!(prm.H, check=false)
    if !issuccess(prm.F)
        return false
    end
    return true
end

calcg_prm!(g, prm::SumOfSquaresCone) = (g .= prm.g)
calcHiprod_prm!(prod, arr, prm::SumOfSquaresCone) = ldiv!(prod, prm.F, arr)
calcLiprod_prm!(prod, arr, prm::SumOfSquaresCone) = ldiv!(prm.F.U', arr, prod) # TODO this in-place syntax should not work (arguments order wrong, should accept F.L) but ldiv! is very broken for triangular matrices in 0.7
