#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial matrix cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

An an r-by-r matrix of polynomials, given a basis of size U is represented by a vectors of size r*(r+1)/2*U.
The first r*(r+1)/2 elements correspond to the first matrix of coefficients, the second  r*(r+1)/2 elements correspond to the second matrix of coefficients, and so on.
The coefficients within each matrix are ordered row wise from the lower triangle of the matrix.
=#

mutable struct WSOSPolyInterpMat <: PrimitiveCone
    usedual::Bool
    dim::Int
    r::Int
    u::Int
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
        dim = u * div(r * (r+1), 2)
        prmtv.dim = dim
        prmtv.r = r
        prmtv.u = u
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

function vectomatidx(blockind::Int)
    i = floor(Int, (1+sqrt(8*blockind-7))/2)
    j = blockind - div(i*(i-1), 2)
    return (i, j)
end

# naively separate calculating barrier from calculating in cone
function barfun(scalpnt, ipwt::Vector{Matrix{Float64}}, r::Int, u::Int)
    ret = 0.0
    dim = div(r*(r+1), 2)
    for ipwtj in ipwt
        l = size(ipwtj, 2)
        mat = similar(scalpnt, l*r, l*r)
        mat .= 0.0
        # loop over degree 2d basis polynomials
        for ui in 1:u
            # loop over indices of degree d basis polynomials
            for lj in 1:l, li in 1:lj
                # loop over indices inside triangle slice
                outervecind = dim * (ui-1)
                for blockind in 1:dim
                    vecind = outervecind + blockind
                    (rj, ri) = vectomatidx(blockind)
                    mat[(li-1)*r+ri, (lj-1)*r+rj] += ipwtj[ui,li] * ipwtj[ui,lj] * scalpnt[vecind] # matpnt[ri,rj,ui]
                end # l
                # TODO don't duplicate
                if li < lj && ui == u
                    mat[(li-1)*r+1:li*r, (lj-1)*r+1:lj*r] = Symmetric(mat[(li-1)*r+1:li*r, (lj-1)*r+1:lj*r])
                end
            end # dim
        end # u
        ret -= logdet(Symmetric(mat))
    end # ipwt
    return ret
end

function inconefun(scalpnt, ipwt::Vector{Matrix{Float64}}, r::Int, u::Int)
    dim = div(r*(r+1), 2)
    for ipwtj in ipwt
        l = size(ipwtj, 2)
        mat = zeros(l*r, l*r)
        # loop over degree 2d basis polynomials
        for ui in 1:u
            # loop over indices of degree d basis polynomials
            for lj in 1:l, li in 1:lj
                # loop over indices inside triangle slice
                outervecind = dim * (ui-1)
                for blockind in 1:dim
                    vecind = outervecind + blockind
                    (rj, ri) = vectomatidx(blockind)
                    mat[(li-1)*r+ri, (lj-1)*r+rj] += ipwtj[ui,li] * ipwtj[ui,lj] * scalpnt[vecind] # matpnt[ri,rj,ui]
                end # l
                # TODO don't duplicate
                if li < lj && ui == u
                    mat[(li-1)*r+1:li*r, (lj-1)*r+1:lj*r] = Symmetric(mat[(li-1)*r+1:li*r, (lj-1)*r+1:lj*r])
                end
            end # dim
        end # u
        # @show Symmetric(mat)
        if !isposdef(Symmetric(mat))
            return false
        end
    end # ipwt
    return true
end

WSOSPolyInterpMat(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpMat(r, u, ipwt, false)

dimension(prmtv::WSOSPolyInterpMat) = prmtv.dim
barrierpar_prmtv(prmtv::WSOSPolyInterpMat) = prmtv.r * sum(size(ipwtj, 2) for ipwtj in prmtv.ipwt)
# sum of diagonal matrices with interpolant polynomial repeating on the diagonal
function getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSPolyInterpMat)
    arr .= 0.0
    vecind = 0
    for ui in 1:prmtv.u
        for j in 1:prmtv.r, i in 1:j
            vecind += 1
            if i == j
                arr[vecind] = 1.0
            end
        end
    end
    return arr
end
loadpnt_prmtv!(prmtv::WSOSPolyInterpMat, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::WSOSPolyInterpMat, scal::Float64)
    prmtv.scal = 1.0
    @. prmtv.scalpnt = prmtv.pnt/prmtv.scal
    if !inconefun(prmtv.scalpnt, prmtv.ipwt, prmtv.r, prmtv.u)
        return false
    end
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.scalpnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)
    return factH(prmtv)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterpMat) = (@. g = prmtv.g/prmtv.scal; g)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat) = (mul!(prod, Symmetric(prmtv.H), arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
function calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat)
    ldiv!(prod, prmtv.F, arr)
    @. prod = prod * prmtv.scal * prmtv.scal
    return prod
end
