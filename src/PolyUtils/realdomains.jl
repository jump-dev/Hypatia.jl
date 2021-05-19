#=
definitions of real domains for polynomials
=#

"""
$(TYPEDEF)

Real domains for polynomial constraints.
"""
abstract type Domain{T <: Real} end

"""
$(SIGNATURES)

Dimension of a domain.
"""
function dimension(dom::Domain)::Int end

"""
$(SIGNATURES)

Sample `npts` points from the interior of the domain.
"""
function sample(dom::Domain, npts::Int)::Matrix end

"""
$(SIGNATURES)

Degree of a polynomial constraints defining a domain.
"""
function degree(dom::Domain)::Int end

"""
$(SIGNATURES)

Evaluations of the polynomial domain constraints at the points.
"""
function weights(dom::Domain, pts::AbstractMatrix)::Vector{Vector} end

"""
$(TYPEDEF)

Real vectors ``x ∈ ℝⁿ`` of dimension `n::Int`.
"""
mutable struct FreeDomain{T <: Real} <: Domain{T}
    n::Int
    function FreeDomain{T}(n::Int) where {T <: Real}
        @assert n >= 1
        dom = new{T}()
        dom.n = n
        return dom
    end
end

dimension(dom::FreeDomain) = dom.n

sample(dom::FreeDomain{T}, npts::Int) where {T <: Real} =
    sample(BoxDomain{T}(-ones(T, dom.n), ones(T, dom.n)), npts)::Matrix{T}

degree(::FreeDomain) = 0

weights(::FreeDomain{T}, ::AbstractMatrix{T}) where {T <: Real} = Vector{T}[]::Vector{Vector{T}}

"""
$(TYPEDEF)

Hyperbox ``x ∈ [l, u]`` with lower bounds `l::Vector{T}` and upper bounds
`u::Vector{T}`.
"""
mutable struct BoxDomain{T <: Real} <: Domain{T}
    l::Vector{T}
    u::Vector{T}
    function BoxDomain{T}(l::Vector{<:Real}, u::Vector{<:Real}) where {T <: Real}
        @assert length(l) == length(u)
        dom = new{T}()
        dom.l = l
        dom.u = u
        return dom
    end
end

dimension(dom::BoxDomain) = length(dom.l)

function sample(dom::BoxDomain{T}, npts::Int) where {T <: Real}
    dim = dimension(dom)
    pts = rand(T, npts, dim) .- T(0.5)
    shift = (dom.u + dom.l) .* T(0.5)
    for i in 1:npts
        @views pts[i, :] = pts[i, :] .* (dom.u - dom.l) + shift
    end
    return pts::Matrix{T}
end

degree(::BoxDomain) = 2

function weights(dom::BoxDomain{T}, pts::AbstractMatrix{T}) where {T <: Real}
    @views g = [(pts[:, i] .- dom.l[i]) .* (dom.u[i] .- pts[:, i]) for
        i in 1:size(pts, 2)]
    @assert all(all(gi .>= 0) for gi in g)
    return g::Vector{Vector{T}}
end


# for hyperball and hyperellipse
function ball_sample(dom::Domain{T}, npts::Int) where {T <: Real}
    dim = dimension(dom)
    pts = T.(randn(npts, dim)) # randn doesn't work with all real types
    norms = sum(abs2, pts, dims = 2)
    pts ./= sqrt.(norms) # scale
    norms ./= 2
    gammainv = [gamma_inc(ai, dim / 2)[2] ^ inv(dim) for ai in norms]
    pts .*= gammainv
    return pts::Matrix{T}
end

"""
$(TYPEDEF)

Euclidean hyperball ``\\lVert (x-c) \\rVert_2 \\leq r`` with center
`c::Vector{T}` and positive radius `r::T`.
"""
mutable struct BallDomain{T <: Real} <: Domain{T}
    c::Vector{T}
    r::T
    function BallDomain{T}(c::Vector{T}, r::T) where {T <: Real}
        @assert r > 0
        dom = new{T}()
        dom.c = c
        dom.r = r
        return dom
    end
end

dimension(dom::BallDomain) = length(dom.c)

function sample(dom::BallDomain{T}, npts::Int) where {T <: Real}
    pts = ball_sample(dom, npts)
    pts .*= dom.r # scale
    pts .+= dom.c' # shift
    return pts::Matrix{T}
end

degree(::BallDomain) = 2

function weights(dom::BallDomain{T}, pts::AbstractMatrix{T}) where {T <: Real}
    g = [abs2(dom.r) - sum(abs2, pts[j, :] - dom.c) for j in 1:size(pts, 1)]
    @assert all(g .>= 0)
    return [g]::Vector{Vector{T}}
end

"""
$(TYPEDEF)

Hyperellipse ``(x-c)' Q (x-c) \\leq 1`` with center `c::Vector{T}` and symmetric
positive definite scaling/rotation matrix `Q::AbstractMatrix{T}`.
"""
mutable struct EllipsoidDomain{T <: Real} <: Domain{T}
    c::Vector{T}
    Q::AbstractMatrix{T}
    QU::UpperTriangular{T}
    function EllipsoidDomain{T}(
        c::Vector{T},
        Q::AbstractMatrix{T},
        ) where {T <: Real}
        @assert length(c) == size(Q, 1)
        @assert issymmetric(Q)
        F = cholesky(Q, check = false)
        @assert isposdef(F)
        dom = new{T}()
        dom.c = c
        dom.Q = Q
        dom.QU = F.U
        return dom
    end
end

dimension(dom::EllipsoidDomain) = length(dom.c)

function sample(dom::EllipsoidDomain{T}, npts::Int) where {T <: Real}
    pts = ball_sample(dom, npts)
    rdiv!(pts, dom.QU') # scale/rotate
    pts .+= dom.c' # shift
    return pts::Matrix{T}
end

degree(::EllipsoidDomain) = 2

function weights(
    dom::EllipsoidDomain{T},
    pts::AbstractMatrix{T},
    ) where {T <: Real}
    g = [1 - sum(abs2, dom.QU * (pts[j, :] - dom.c)) for j in 1:size(pts, 1)]
    @assert all(g .>= 0)
    return [g]::Vector{Vector{T}}
end
