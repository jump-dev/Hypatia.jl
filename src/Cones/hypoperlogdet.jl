#=
Copyright 2018, Chris Coey and contributors

(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle i.e. svec space) symmetric positive define matrix
(smat space) (u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeConeTriangle definition)

barrier (guessed, based on analogy to hypoperlog barrier)
-log(v*logdet(W/v) - u) - logdet(W) - log(v)

TODO only use one decomposition on Symmetric(W) for isposdef and logdet
TODO symbolically calculate gradient and Hessian
=#

mutable struct HypoPerLogdet <: Cone
    usedual::Bool
    dim::Int
    side::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function HypoPerLogdet(dim::Int, isdual::Bool)
        cone = new()
        cone.usedual = isdual
        cone.dim = dim
        side = round(Int, sqrt(0.25 + 2*(dim - 2)) - 0.5)
        cone.side = side
        cone.mat = Matrix{Float64}(undef, side, side)
        cone.g = Vector{Float64}(undef, dim)
        cone.H = similar(cone.g, dim, dim)
        cone.H2 = similar(cone.H)
        function barfun(pnt)
            u = pnt[1]
            v = pnt[2]
            W = similar(pnt, side, side)
            svec_to_smat!(W, view(pnt, 3:dim))
            return -log(v*logdet(W/v) - u) - logdet(W) - log(v)
        end
        cone.barfun = barfun
        cone.diffres = DiffResults.HessianResult(cone.g)
        return cone
    end
end

HypoPerLogdet(dim::Int) = HypoPerLogdet(dim, false)

dimension(cone::HypoPerLogdet) = cone.dim
get_nu(cone::HypoPerLogdet) = cone.side + 2

function set_initial_point(arr::AbstractVector{Float64}, cone::HypoPerLogdet)
    arr[1] = -1.0
    arr[2] = 1.0
    smat_to_svec!(view(arr, 3:cone.dim), Matrix(1.0I, cone.side, cone.side))
    return arr
end

loadpnt!(cone::HypoPerLogdet, pnt::AbstractVector{Float64}) = (cone.pnt = pnt)

function incone(cone::HypoPerLogdet, scal::Float64)
    pnt = cone.pnt
    u = pnt[1]
    v = pnt[2]
    W = cone.mat
    svec_to_smat!(W, view(pnt, 3:cone.dim))
    if v <= 0.0 || !isposdef(Symmetric(W)) || u >= v*logdet(Symmetric(W)/v) # TODO only use one decomposition on Symmetric(W) for isposdef and logdet
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.pnt)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)

    return factorize_hess(cone)
end
