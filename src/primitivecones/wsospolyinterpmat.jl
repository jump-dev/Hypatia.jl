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
    ipwt::Vector{Matrix{Float64}}
    pnt::AbstractVector{Float64}
    scalpnt::Vector{Float64}
    scal::Float64
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    mat::Vector{Matrix{Float64}}
    matfact::Vector{CholeskyPivoted{Float64,Array{Float64,2}}}

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
        prmtv.mat = [similar(ipwt[1], size(ipwtj, 2) * r, size(ipwtj, 2) * r) for ipwtj in ipwt]
        prmtv.matfact = Vector{CholeskyPivoted{Float64,Array{Float64,2}}}(undef, length(ipwt))
        return prmtv
    end
end

function buildmat!(prmtv::WSOSPolyInterpMat, pnt)
    (R, U) = (prmtv.r, prmtv.u)
    for (j, ipwtj) in enumerate(prmtv.ipwt)
        L = size(ipwtj, 2)
        mat = prmtv.mat[j]
        mat .= 0.0

        for l in 1:L, k in 1:l
            (bl, bk) = ((l-1)*R, (k-1)*R)
            uo = 0
            for p in 1:R, q in 1:p
                val = sum(ipwtj[u,l] * ipwtj[u,k] * pnt[uo+u] for u in 1:U)
                bp = bl + p
                bq = bk + q
                if p == q
                    mat[bp,bq] = val
                else
                    mat[bl+p, bk+q] = mat[bl+q, bk+p] = rt2i*val
                end
                uo += U
            end
        end
        prmtv.matfact[j] = cholesky!(Symmetric(mat, :L), Val(true), check=false)
        if !isposdef(prmtv.matfact[j])
            return false
        end
    end
    return true
end

WSOSPolyInterpMat(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpMat(r, u, ipwt, false)

dimension(prmtv::WSOSPolyInterpMat) = prmtv.dim
barrierpar_prmtv(prmtv::WSOSPolyInterpMat) = prmtv.r * sum(size(ipwtj, 2) for ipwtj in prmtv.ipwt)

function getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSPolyInterpMat)
    # sum of diagonal matrices with interpolant polynomial repeating on the diagonal
    idx = 1
    for i in 1:prmtv.r, j in 1:i, u in 1:prmtv.u
        if i == j
            arr[idx] = 1.0
        else
            arr[idx] = 0.0
        end
        idx += 1
    end
    return arr
end

loadpnt_prmtv!(prmtv::WSOSPolyInterpMat, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::WSOSPolyInterpMat, scal::Float64)
    @timeit to "incone" begin
    prmtv.scal = 1.0
    @. prmtv.scalpnt = prmtv.pnt/prmtv.scal
    @timeit to "buildmat" begin
        if !(buildmat!(prmtv, prmtv.scalpnt))
            return false
        end
    end

    prmtv.g .= 0.0
    prmtv.H .= 0.0
    for (j, ipwtj) in enumerate(prmtv.ipwt)
        @timeit to "getting g" begin
        @timeit to "inversion" Winv = inv(prmtv.matfact[j])
        L = size(ipwtj, 2)

        idx = 0
        # outer indices for W
        for p in 1:prmtv.r,  q in 1:p,  u in 1:prmtv.u
            idx += 1
            # sum for gradient
            for k in 1:L, l in 1:L
                (bk, bl) = ((k-1)*prmtv.r, (l-1)*prmtv.r)
                # TODO avoid some doubling up
                if p == q
                    prmtv.g[idx] -= ipwtj[u,k] * ipwtj[u,l] * Winv[bk+p, bl+q]
                else
                    prmtv.g[idx] -= ipwtj[u,k] * ipwtj[u,l] * Winv[bk+p, bl+q] * rt2
                end
            end
            # hessian
            idx2 = 0
            @timeit to "computing H" begin
            for p2 in 1:prmtv.r,  q2 in 1:p2,  u2 in 1:prmtv.u
                idx2 += 1
                sum1 = 0.0
                sum2 = 0.0
                sum3 = 0.0
                sum4 = 0.0
                for k2 in 1:L, l2 in 1:L
                    (bk2, bl2) = ((k2-1)*prmtv.r, (l2-1)*prmtv.r)
                    sum1 += Winv[bk2+p, bl2+p2] * ipwtj[u,k2] * ipwtj[u2,l2]
                    sum2 += Winv[bk2+q, bl2+q2] * ipwtj[u,k2] * ipwtj[u2,l2]
                    sum3 += Winv[bk2+p, bl2+q2] * ipwtj[u,k2] * ipwtj[u2,l2]
                    sum4 += Winv[bk2+q, bl2+p2] * ipwtj[u,k2] * ipwtj[u2,l2]
                end
                if p == q
                    if p2 == q2
                        prmtv.H[idx, idx2] += sum1 * sum2
                    else
                        prmtv.H[idx, idx2] += (sum1 * sum2) * rt2i + (sum3 * sum4) * rt2i
                    end
                else
                    if p2 == q2
                        prmtv.H[idx, idx2] += (sum1 * sum2) * rt2i + (sum3 * sum4) * rt2i
                    else
                        prmtv.H[idx, idx2] += (sum1 * sum2) + (sum3 * sum4)
                    end
                end
            end
            end #timing hessian
        end
        end # timing getting g
    end
    end # timing incone
    return factH(prmtv)
end

# calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterpMat) = (@. g = prmtv.g/prmtv.scal; g)
# calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat) = (mul!(prod, Symmetric(prmtv.H), arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
# calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat) = (ldiv!(prod, prmtv.F, arr); @. prod = prod * prmtv.scal * prmtv.scal; prod)
