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
        prmtv.mat = [similar(ipwt[1], size(ipwtj, 2) * r, size(ipwtj, 2) * r) for ipwtj in ipwt]
        prmtv.barfun = (x -> barfun(x, ipwt, r, u, true))
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

# calculate barrier value
function barfun(pnt, ipwt::Vector{Matrix{Float64}}, R::Int, U::Int, calc_barval::Bool)
    barval = 0.0

    for ipwtj in ipwt
        L = size(ipwtj, 2)
        mat = similar(pnt, L*R, L*R)
        mat .= 0.0

        for l in 1:L, k in 1:l
            (bl, bk) = ((l-1)*R, (k-1)*R)
            uo = 0
            for p in 1:R, q in 1:p
                val = sum(ipwtj[u,l] * ipwtj[u,k] * pnt[uo+u] for u in 1:U)
                # FIXME inconsistency between ordering of the variables in the outer/inner indexing between this mat and the point in the cone differentiating more confusing
                if p == q
                    mat[bl+p, bk+q] = val
                else
                    mat[bl+p, bk+q] = mat[bl+q, bk+p] = rt2i*val
                end
                uo += U
            end
        end

        F = cholesky!(Symmetric(mat, :L), check=false)
        if !isposdef(F)
            return NaN
        end
        if calc_barval
            barval -= logdet(F)
        end
    end

    return barval
end

function barfun!(pnt, prmtv::WSOSPolyInterpMat)
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
        mat .= Symmetric(mat, :L)
    end
    return nothing
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
    @timeit to "actual barfun" begin
    if isnan(barfun(prmtv.scalpnt, prmtv.ipwt, prmtv.r, prmtv.u, false))
        return false
    end
    end
    @timeit to "needless barfun" barfun!(prmtv.scalpnt, prmtv)
    # @timeit to "autodiff" prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.scalpnt)

    prmtv.g .= 0.0
    prmtv.H .= 0.0
    der11 = 0.0
    der44 = 0.0
    for (j, ipwtj) in enumerate(prmtv.ipwt)
        @timeit to "getting g" begin
        @timeit to "needless inversion" Winv = inv(prmtv.mat[j])
        L = size(ipwtj, 2)
        # der11 += sum(ipwtj[1,k] * ipwtj[1,l] * Winv[(k-1)*prmtv.r+1, (l-1)*prmtv.r+1] for k in 1:L, l in 1:L)^2
        # der44 += sum(ipwtj[1,k] * ipwtj[1,l] * Winv[(k-1)*prmtv.r+1, (l-1)*prmtv.r+1] for k in 1:L, l in 1:L) *
        #          sum(ipwtj[1,k] * ipwtj[1,l] * Winv[(k-1)*prmtv.r+2, (l-1)*prmtv.r+2] for k in 1:L, l in 1:L) +
        #          sum(ipwtj[1,k] * ipwtj[1,l] * Winv[(k-1)*prmtv.r+1, (l-1)*prmtv.r+2] for k in 1:L, l in 1:L) *
        #          sum(ipwtj[1,k] * ipwtj[1,l] * Winv[(k-1)*prmtv.r+2, (l-1)*prmtv.r+1] for k in 1:L, l in 1:L)

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
    # tempg = DiffResults.gradient(prmtv.diffres)
    tempH = DiffResults.hessian(prmtv.diffres)
    # @show prmtv.H
    # @show tempH[4,4]
    # @show prmtv.H ./ tempH
    # @show prmtv.g
    # @show der44
    # @show  tempH[4,4] / der44
    # @show tempH[1,2]
    # prmtv.H .= DiffResults.hessian(prmtv.diffres)
    end # timing incone
    return factH(prmtv)
end

# calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterpMat) = (@. g = prmtv.g/prmtv.scal; g)
# calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat) = (mul!(prod, Symmetric(prmtv.H), arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
# calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat) = (ldiv!(prod, prmtv.F, arr); @. prod = prod * prmtv.scal * prmtv.scal; prod)
