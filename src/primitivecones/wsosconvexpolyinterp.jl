#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

=#

mutable struct WSOSConvexPolyInterp <: PrimitiveCone
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

    function WSOSConvexPolyInterp(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}, isdual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == u
        end
        prmtv = new()
        prmtv.usedual = !isdual # using dual barrier
        dim = u
        prmtv.dim = dim
        prmtv.r = r
        prmtv.u = u
        prmtv.ipwt = ipwt
        prmtv.scalpnt = similar(ipwt[1], dim)
        prmtv.g = similar(ipwt[1], dim)
        prmtv.H = similar(ipwt[1], dim, dim)
        prmtv.H2 = similar(prmtv.H)
        prmtv.barfun = (x -> wsosconvexbarfun(x, ipwt, r, u, true))
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

# calculate barrier value
function wsosconvexbarfun(pnt, ipwt::Vector{Matrix{Float64}}, R::Int, U::Int, calc_barval::Bool)
    barval = 0.0

    for ipwtj in ipwt
        L = size(ipwtj, 2)
        mat = similar(pnt, L*R, L*R)
        mat .= 0.0

        for l in 1:L, k in 1:l
            (bl, bk) = ((l-1)*R, (k-1)*R)
            for p in 1:R, q in 1:p
                # val = sum(ipwtj[u,l,p,q] * ipwtj[u,k,p,q] * pnt[u] for u in 1:U)
                val = sum(ipwtj[u,l] * ipwtj[u,k] * pnt[u] for u in 1:U)
                if p == q
                    mat[bl+p, bk+q] = val
                else
                    mat[bl+p, bk+q] = mat[bl+q, bk+p] = val
                end
            end
        end

        # !calc_barval && @show mat

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

WSOSConvexPolyInterp(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}) = WSOSConvexPolyInterp(r, u, ipwt, false)

dimension(prmtv::WSOSConvexPolyInterp) = prmtv.dim



barrierpar_prmtv(prmtv::WSOSConvexPolyInterp) = prmtv.r * sum(size(ipwtj, 2) for ipwtj in prmtv.ipwt)



# getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSConvexPolyInterp) = (@. arr = 1.0; arr)

# function getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSConvexPolyInterp)
#     # sum of diagonal matrices with interpolant polynomial repeating on the diagonal
#     idx = 1
#     for u in 1:prmtv.u
#         if i == j
#             arr[idx] = 1.0
#         else
#             arr[idx] = 0.0
#         end
#         idx += 1
#     end
#     return arr
# end

loadpnt_prmtv!(prmtv::WSOSConvexPolyInterp, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::WSOSConvexPolyInterp, scal::Float64)
    prmtv.scal = 1.0
    @. prmtv.scalpnt = prmtv.pnt/prmtv.scal
    if isnan(wsosconvexbarfun(prmtv.scalpnt, prmtv.ipwt, prmtv.r, prmtv.u, false))
        return false
    end
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.scalpnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)
    return factH(prmtv)
end

# calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSConvexPolyInterp) = (@. g = prmtv.g/prmtv.scal; g)
# calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSConvexPolyInterp) = (mul!(prod, Symmetric(prmtv.H), arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
# calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSConvexPolyInterp) = (ldiv!(prod, prmtv.F, arr); @. prod = prod * prmtv.scal * prmtv.scal; prod)
