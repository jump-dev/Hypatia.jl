#=
definitions of domains for polynomials
=#

abstract type Domain{T <: Real} end


# all reals
mutable struct FreeDomain{T <: Real} <: Domain{T}
    n::Int
end

get_dimension(dom::FreeDomain) = dom.n
get_degree(::FreeDomain) = 0

interp_sample(dom::FreeDomain{T}, npts::Int) where {T <: Real} =
    interp_sample(BoxDomain{T}(-ones(T, dom.n), ones(T, dom.n)), npts)

get_weights(::FreeDomain{T}, ::AbstractMatrix{T}) where {T <: Real} = Vector{T}[]


# hyperrectangle/box
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

get_dimension(dom::BoxDomain) = length(dom.l)
get_degree(::BoxDomain) = 2

function interp_sample(dom::BoxDomain{T}, npts::Int) where {T <: Real}
    dim = get_dimension(dom)
    pts = rand(T, npts, dim) .- T(0.5)
    shift = (dom.u + dom.l) .* T(0.5)
    for i in 1:npts
        @views pts[i, :] = pts[i, :] .* (dom.u - dom.l) + shift
    end
    return pts
end

function get_weights(dom::BoxDomain{T}, pts::AbstractMatrix{T}) where {T <: Real}
    @views g = [(pts[:, i] .- dom.l[i]) .* (dom.u[i] .- pts[:, i]) for
        i in 1:size(pts, 2)]
    @assert all(all(gi .>= 0) for gi in g)
    return g
end


# for hyperball and hyperellipse
function ball_sample(dom::Domain{T}, npts::Int) where {T <: Real}
    dim = get_dimension(dom)
    pts = T.(randn(npts, dim)) # randn doesn't work with all real types
    norms = sum(abs2, pts, dims = 2)
    pts ./= sqrt.(norms) # scale
    norms ./= 2
    gammainv = [gamma_inc(ai, dim / 2)[2] ^ inv(dim) for ai in norms]
    pts .*= gammainv
    return pts
end


# Euclidean hyperball
mutable struct BallDomain{T <: Real} <: Domain{T}
    c::Vector{T}
    r::T
    function BallDomain{T}(c::Vector{T}, r::T) where {T <: Real}
        dom = new{T}()
        dom.c = c
        dom.r = r
        return dom
    end
end

get_dimension(dom::BallDomain) = length(dom.c)
get_degree(::BallDomain) = 2

function interp_sample(dom::BallDomain{T}, npts::Int) where {T <: Real}
    pts = ball_sample(dom, npts)
    pts .*= dom.r # scale
    pts .+= dom.c' # shift
    return pts
end

function get_weights(dom::BallDomain{T}, pts::AbstractMatrix{T}) where {T <: Real}
    g = [abs2(dom.r) - sum(abs2, pts[j, :] - dom.c) for j in 1:size(pts, 1)]
    @assert all(g .>= 0)
    return [g]
end


# hyperellipse: (x-c)'Q(x-c) <= 1
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

get_dimension(dom::EllipsoidDomain) = length(dom.c)
get_degree(::EllipsoidDomain) = 2

function interp_sample(dom::EllipsoidDomain{T}, npts::Int) where {T <: Real}
    pts = ball_sample(dom, npts)
    rdiv!(pts, dom.QU') # scale/rotate
    pts .+= dom.c' # shift
    return pts
end

function get_weights(
    dom::EllipsoidDomain{T},
    pts::AbstractMatrix{T},
    ) where {T <: Real}
    g = [1 - sum(abs2, dom.QU * (pts[j, :] - dom.c)) for j in 1:size(pts, 1)]
    @assert all(g .>= 0)
    return [g]
end
