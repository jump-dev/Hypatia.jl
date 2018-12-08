#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial matrix cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

mutable struct WSOSPolyInterpMat <: PrimitiveCone
    usedual::Bool
    dim::Int
    r::Int
    u::Int
    matpnt::Array{Float64,3}
    # mat::Matrix{Float64}
    ipwt::Vector{Matrix{Float64}}
    pnt::AbstractVector{Float64}
    scalpnt::Vector{Float64}
    scal::Float64
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function WSOSPolyInterpMat(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}, isdual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == u
        end
        prmtv = new()
        prmtv.usedual = !isdual # using dual barrier
        dim = u * r^2
        prmtv.dim = dim
        prmtv.r = r
        prmtv.u = u
        prmtv.matpnt = Array{Float64,3}(undef, u, r, r)
        # prmtv.mat = zeros(u*r, u*r)
        prmtv.ipwt = ipwt
        prmtv.scalpnt = similar(ipwt[1], dim)
        prmtv.g = similar(ipwt[1], dim)
        prmtv.H = similar(ipwt[1], dim, dim)
        prmtv.H2 = similar(prmtv.H)
        prmtv.barfun = (x -> barfun(x, ipwt, r, u))
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

# naively separate calculating barrier from calculating in cone
function barfun(matpnt, ipwt::Vector{Matrix{Float64}}, r::Int, u::Int)
    ret = 0.0
    for ipwtj in ipwt
        l = size(ipwtj, 2)
        mat = similar(matpnt, l*r, l*r)
        mat .= 0.0
        for j in 1:l, i in 1:l
            mat[(i-1)*r+1:i*r, (j-1)*r+1:j*r] .+= sum(ipwtj[ui,i] * ipwtj[ui,j] * matpnt[:,:,ui] for ui in 1:u)
        end
        ret -= logdet(mat)
    end
    return ret
end
function inconefun(matpnt, ipwt::Vector{Matrix{Float64}}, r::Int, u::Int)
    ret = true
    for ipwtj in ipwt
        l = size(ipwtj, 2)
        mat = zeros(l*r, l*r)
        for j in 1:l, i in 1:l
            # @show [ipwtj[ui,i] * ipwtj[ui,j] for ui in 1:u]
            mat[(i-1)*r+1:i*r, (j-1)*r+1:j*r] .+= sum(ipwtj[ui,i] * ipwtj[ui,j] * matpnt[:,:,ui] for ui in 1:u)
            # @show mat
        end
        if !isposdef(mat)
            ret = false
        end
    end
    return ret
end


WSOSPolyInterpMat(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpMat(r, u, ipwt, false)

dimension(prmtv::WSOSPolyInterpMat) = prmtv.dim
barrierpar_prmtv(prmtv::WSOSPolyInterpMat) = prmtv.r * sum(size(ipwtj, 2) for ipwtj in prmtv.ipwt)
# sum of diagonal matrices with interpolant polynomial repeating on the diagonal
function getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSPolyInterpMat)
    arr .= vcat([Matrix{Float64}(I, prmtv.r, prmtv.r)[:] for ui in 1:prmtv.u]...) # TODO tidy
    return arr
end
loadpnt_prmtv!(prmtv::WSOSPolyInterpMat, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::WSOSPolyInterpMat, scal::Float64)
    prmtv.scal = scal
    @. prmtv.scalpnt = prmtv.pnt/prmtv.scal
    prmtv.matpnt = reshape(prmtv.scalpnt, prmtv.r, prmtv.r, prmtv.u)
    if !inconefun(prmtv.matpnt, prmtv.ipwt, prmtv.r, prmtv.u)
        return false
    end
    # prmtv.mat = makemat(prmtv.matpnt, prmtv.ipwt, prmtv.r, prmtv.u)
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.matpnt)
    # will modify prmtv.mat
    # prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.matpnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)
    # factorization of Hessian used later
    return factH(prmtv)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterpMat) = (@. g = prmtv.g/prmtv.scal; g)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat) = (mul!(prod, Symmetric(prmtv.H), arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
function calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat)
    ldiv!(prod, prmtv.F, arr)
    @. prod = prod * prmtv.scal * prmtv.scal
    return prod
end
