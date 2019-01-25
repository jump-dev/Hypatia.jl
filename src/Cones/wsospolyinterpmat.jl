#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial matrix cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

mutable struct WSOSPolyInterpMat <: Cone
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
        cone = new()
        cone.usedual = !isdual # using dual barrier
        dim = u * div(r * (r + 1), 2)
        cone.dim = dim
        cone.r = r
        cone.u = u
        cone.ipwt = ipwt
        cone.scalpnt = similar(ipwt[1], dim)
        cone.g = similar(ipwt[1], dim)
        cone.H = similar(ipwt[1], dim, dim)
        cone.H2 = similar(cone.H)
        cone.barfun = (x -> barfun(x, ipwt, r, u, true))
        cone.diffres = DiffResults.HessianResult(cone.g)
        return cone
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

WSOSPolyInterpMat(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpMat(r, u, ipwt, false)

dimension(cone::WSOSPolyInterpMat) = cone.dim
get_nu(cone::WSOSPolyInterpMat) = cone.r * sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

function set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterpMat)
    # sum of diagonal matrices with interpolant polynomial repeating on the diagonal
    idx = 1
    for i in 1:cone.r, j in 1:i, u in 1:cone.u
        if i == j
            arr[idx] = 1.0
        else
            arr[idx] = 0.0
        end
        idx += 1
    end
    return arr
end

loadpnt!(cone::WSOSPolyInterpMat, pnt::AbstractVector{Float64}) = (cone.pnt = pnt)

function incone(cone::WSOSPolyInterpMat, scal::Float64)
    cone.scal = 1.0
    @. cone.scalpnt = cone.pnt/cone.scal
    if isnan(barfun(cone.scalpnt, cone.ipwt, cone.r, cone.u, false))
        return false
    end
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.scalpnt)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)
    return factH(cone)
end

# calcg!(g::AbstractVector{Float64}, cone::WSOSPolyInterpMat) = (@. g = cone.g/cone.scal; g)
# calcHarr!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::WSOSPolyInterpMat) = (mul!(prod, Symmetric(cone.H), arr); @. prod = prod / cone.scal / cone.scal; prod)
# calcHiarr!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::WSOSPolyInterpMat) = (ldiv!(prod, cone.F, arr); @. prod = prod * cone.scal * cone.scal; prod)
