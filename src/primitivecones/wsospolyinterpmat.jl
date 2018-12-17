#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial matrix cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

# TODO have half as many variables matrices are symmetric

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
        prmtv.matpnt = Array{Float64,3}(undef, u, r, r) # unused
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

function naive_blow_up(P::Matrix{Float64}, r::Int)
    (u, l) = size(P)
    Pbig = zeros(u*r, l*r)
    id = Matrix{Float64}(I, r, r)
    for j in 1:l, i in 1:u
        Pbig[(i-1)*r+1:i*r, (j-1)*r+1:j*r] = P[i,j] * id
    end
    return Pbig
end

function naive_diagonalizeQ(q::Vector{Float64}, u::Int, r::Int)
    Qbig = zeros(u*r, u*r)
    Q = reshape(q, r, r, u)
    for i in 1:u
        Qbig[(i-1)*r+1:i*r, (i-1)*r+1:i*r] = Q[:,:,i]
    end
    return Qbig
end

# naively separate calculating barrier from calculating in cone
function barfun(scalpnt, ipwt::Vector{Matrix{Float64}}, r::Int, u::Int)
    matpnt = reshape(scalpnt, r, r, u)
    ret = 0.0
    for ipwtj in ipwt
        l = size(ipwtj, 2)
        mat = similar(matpnt, l*r, l*r)
        mat .= 0.0
        for j in 1:l, i in 1:l
            mat[(i-1)*r+1:i*r, (j-1)*r+1:j*r] .+= sum(ipwtj[ui,i] * ipwtj[ui,j] * 0.5*(matpnt[:,:,ui] + matpnt[:,:,ui]') for ui in 1:u)
        end
        ret -= logdet(mat)
    end
    return ret
end
function inconefun(scalpnt, ipwt::Vector{Matrix{Float64}}, r::Int, u::Int)
    matpnt = reshape(scalpnt, r, r, u)
    # @show matpnt
    ret = true
    for ipwtj in ipwt
        l = size(ipwtj, 2)
        mat = zeros(l*r, l*r)
        for j in 1:l, i in 1:l
            mat[(i-1)*r+1:i*r, (j-1)*r+1:j*r] .+= sum(ipwtj[ui,i] * ipwtj[ui,j] * 0.5*(matpnt[:,:,ui] + matpnt[:,:,ui]') for ui in 1:u)
        end
        # @show mat
        if !isposdef(mat)
            # @show mat
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
    # arr .= 1.0 # would give a positive semidefinite matrix but not positive definite which we want to begin with
    arr .= vcat([Matrix{Float64}(I, prmtv.r, prmtv.r)[:] for ui in 1:prmtv.u]...) # TODO tidy
    return arr
end
loadpnt_prmtv!(prmtv::WSOSPolyInterpMat, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::WSOSPolyInterpMat, scal::Float64)
    prmtv.scal = 1.0
    @. prmtv.scalpnt = prmtv.pnt/prmtv.scal
    if !inconefun(prmtv.scalpnt, prmtv.ipwt, prmtv.r, prmtv.u)
        return false
    end
    # prmtv.mat = makemat(prmtv.matpnt, prmtv.ipwt, prmtv.r, prmtv.u)
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.scalpnt)
    # prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.matpnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)
    # factorization of Hessian used later
    # @show prmtv.H
    return factH(prmtv)
    # @. prmtv.H2 = prmtv.H
    # prmtv.F = cholesky!(Symmetric(prmtv.H2, :U), Val(true), check=false)
    return true




    # @. prmtv.g = 0.0
    # @. prmtv.H = 0.0
    # r = prmtv.r
    # bigQ = naive_diagonalizeQ(prmtv.scalpnt, prmtv.u, r)
    #
    # for P in prmtv.ipwt
    #     bigP = naive_blow_up(P, prmtv.r)
    #
    #     level1 = bigP' * bigQ * bigP
    #     F = cholesky(Symmetric(level1, :L), Val(true), check=false)
    #     if !isposdef(F)
    #         @show level1
    #         return false
    #     end
    #     level1r = view(bigP', F.p, :)
    #     level2 = bigP * ldiv!(F.L, level1r)
    #
    #     vector_ind = 0
    #     @inbounds for ui in 1:prmtv.u
    #         for rj in 1:r, ri in 1:r
    #             vector_ind += 1
    #             prmtv.g[vector_ind] -= level2[(ui-1)*r+ri, (ui-1)*r+rj]
    #         end
    #     end
    # end
    #
    # return factH(prmtv)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterpMat) = (@. g = prmtv.g/prmtv.scal; g)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat) = (mul!(prod, Symmetric(prmtv.H), arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
function calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat)
    ldiv!(prod, prmtv.F, arr)
    @. prod = prod * prmtv.scal * prmtv.scal
    return prod
end
