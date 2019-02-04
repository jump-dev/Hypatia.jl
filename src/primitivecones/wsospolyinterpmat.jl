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
    P0::Matrix{Float64}
    weight_vecs::Vector{Vector{Float64}}
    lower_dims::Vector{Int}
    pnt::AbstractVector{Float64}
    scalpnt::Vector{Float64}
    scal::Float64
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function WSOSPolyInterpMat(
        r::Int,
        u::Int,
        P0::Matrix{Float64},
        weight_vecs::Vector{Vector{Float64}},
        lower_dims::Vector{Int},
        isdual::Bool
        )

        @assert size(P0, 1) == u
        for wv in weight_vecs
            @assert length(wv) == u
        end
        prmtv = new()
        prmtv.usedual = !isdual # using dual barrier
        dim = u * div(r * (r+1), 2)
        prmtv.dim = dim
        prmtv.r = r
        prmtv.u = u
        prmtv.P0 = P0
        prmtv.weight_vecs = weight_vecs
        prmtv.lower_dims = lower_dims
        prmtv.scalpnt = similar(P0, dim)
        prmtv.g = similar(P0, dim)
        prmtv.H = similar(P0, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        prmtv.barfun = (x -> barfun(x, P0, weight_vecs, lower_dims, r, u, true))
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

# calculate barrier value
function barfun(
    pnt,
    P0::Matrix{Float64},
    weight_vecs::Vector{Vector{Float64}},
    lower_dims::Vector{Int},
    R::Int,
    U::Int,
    calc_barval::Bool
    )

    barval = 0.0
    for j in eachindex(weight_vecs)
        L = lower_dims[j]
        mat = similar(pnt, L*R, L*R)
        mat .= 0.0

        for l in 1:L, k in 1:l
            (bl, bk) = ((l-1)*R, (k-1)*R)
            uo = 0
            for p in 1:R, q in 1:p
                val = sum(ipwtj[u,l] * ipwtj[u,k] * pnt[uo+u] * weight_vecs[uo+u] for u in 1:U)
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

        # if !calc_barval
        #     @show eigvals(Symmetric(mat, :L))
        #     @show cond(Symmetric(mat, :L))
        # end

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

WSOSPolyInterpMat(r::Int, u::Int, P0::Matrix{Float64}, weight_vecs::Vector{Vector{Float64}}, lower_dims::Vector{Int}) = WSOSPolyInterpMat(r, u, P0, weight_vecs, lower_dims, false)

dimension(prmtv::WSOSPolyInterpMat) = prmtv.dim
barrierpar_prmtv(prmtv::WSOSPolyInterpMat) = prmtv.r * sum(prmtv.lower_dims) # TODO exclude P0

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
    prmtv.scal = 1.0
    @. prmtv.scalpnt = prmtv.pnt/prmtv.scal
    if isnan(barfun(prmtv.scalpnt, prmtv.P0, prmtv.weight_vecs, prmtv.lower_dims, prmtv.r, prmtv.u, false))
        return false
    end
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.scalpnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)
    return factH(prmtv)
end

# calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterpMat) = (@. g = prmtv.g/prmtv.scal; g)
# calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat) = (mul!(prod, Symmetric(prmtv.H), arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
# calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterpMat) = (ldiv!(prod, prmtv.F, arr); @. prod = prod * prmtv.scal * prmtv.scal; prod)
