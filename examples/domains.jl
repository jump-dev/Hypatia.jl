#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using Hypatia
using MathOptInterface
MOI = MathOptInterface
using MultivariatePolynomials
using DynamicPolynomials
using SemialgebraicSets
using JuMP
using PolyJuMP
using SumOfSquares
using LinearAlgebra
using GSL
using Distributions
using ApproxFun
using Test

import Combinatorics

abstract type InterpDomain end

mutable struct Box <: InterpDomain
    l::Vector{Float64}
    u::Vector{Float64}
    function Box(l::Vector{Float64}, u::Vector{Float64})
        @assert length(l) == length(u)
        d = new()
        d.l = l
        d.u = u
        return d
    end
end
function Box(l::Vector{T}, u::Vector{T}) where T <: Real
    return Box(Float64.(l), Float64.(u))
end


# (x-c)'Q(x-c) \leq 1
# struct Ellipsoid <: InterpDomain
#     c::Vector{Float64}
#     Q::Matrix{Float64}
#     function Ellipsoid(c::Vector{Float64}, Q::Matrix{Float64})
#         @assert isposdef(Q)
#         @assert length(c) == size(Q, 1)
#         d = new()
#         d.c = c
#         d.Q = Q
#         return d
#     end
# end
# function Ball(c::Vector{Float64})
#     dim = length(c)
#     return Ellipsoid(c, Matrix{Float64}(I, dim, dim))
# end

# should be an ellipsoid
mutable struct Ball <: InterpDomain
    c::Vector{Float64}
    r::Float64
    function Ball(c::Vector{Float64}, r::Float64)
        d = new()
        d.c = c
        d.r = r
        return d
    end
end

dimension(d::Box) = length(d.l)
dimension(d::Ball) = length(d.c)

function interp_sample(d::Box, npts::Int)
    dim = dimension(d)
    pts = rand(dim, npts)
    pts = (d.u + d.l)/2.0 .+ (d.u - d.l) .* (pts .- 0.5)
    return pts'
end
# will be replaced with proper sampling function
function interp_sample(d::Ball, npts::Int, strategy::Int=3)
    dim = dimension(d)

    if strategy == 1
        # generate uniformly distributed points in a ball, doesn't work well numerically
        pts = randn(npts, dim)
        norms = sum(pts.^2, dims=2)
        pts .*= d.r ./ sqrt.(norms)
        # sf_gamma_inc_Q is the normalized incomplete gamma function
        pts .*= sf_gamma_inc_Q.(norms/2, dim/2).^(1/dim)

    elseif strategy == 2
        # heuristic with 1-radius being truncated exponential
        pts_on_sphere = randn(npts, dim)
        norms = sqrt.(sum(pts_on_sphere.^2, dims=2))
        pts_on_sphere ./= norms
        if any(norm.(pts_on_sphere) .> 1) || any(norm.(pts_on_sphere) .< 0)
            error()
        end
        lambda = dim * 10.0
        rdist = Truncated(Exponential(lambda), 0, 1)
        radii = (1.0 .- rand(rdist, npts)) .* d.r
        pts = pts_on_sphere .* radii

    elseif strategy == 3
        # sample from box and project
        pts = 0.5 .- rand(npts, dim)
        for i in 1:npts
            if norm(pts[i, :]) > 0.5
                pts[i, :] .*= 0.4999 / norm(pts[i, :]) #* sqrt(n))
            end
            pts[i, :] .= pts[i, :] * 2 * d.r + d.c
        end

    end

    return pts
end

# struct BoxSurf <: InterpDomain
#     l::Vector{Float64}
#     u::Vector{Float64}
# end
#
# dimension(d::Box) = length(d.l)
#
#
# struct BallSurf <: InterpDomain
#     c::Vector{Float64}
#     r::Float64
# end
#
# struct Ellipse <: InterpDomain
#     c::Vector{Float64}
#     Q::Matrix{Float64}
# end
#
# struct EllipseSurf <: InterpDomain
#     c::Vector{Float64}
#     Q::Matrix{Float64}
# end


get_bss(dom::InterpDomain, x) = error("")
function get_bss(dom::Box, x)
    bss = BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}()
    for i in 1:dimension(dom)
        addinequality!(bss, (-x[i] + dom.u[i]) * (x[i] - dom.l[i]))
    end
    return bss
end
function get_bss(dom::Ball, x)
    return @set sum((x - dom.c).^2) <= dom.r^2
end


# function get_InterpDomain(dom::Ellipsoid, x)
#     bss = BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}()
#     lhs = (x - dom.c)' * dom.Q * (x - dom.c)
#     addinequality!(bss, 1 - lhs)
#     return bss
# end

function get_weights(::Box, bss::BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}, pts)
    m = length(bss.p)
    U = size(pts, 1)
    g = Vector{Vector{Float64}}(undef, m)
    for i in 1:m
        g[i] = bss.p[i].(pts[:,i])
        @assert all(g[i] .> -1e-6)
    end
    return g
end
function get_weights(::Ball, bss::BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}, pts)
    U = size(pts, 1)
    @assert length(bss.p) == 1
    # sub_func(j) = dom.r^2 - sum((dom.c - pts[j, :]).^2)
    sub_func(j) = bss.p[1](pts[j, :])
    g = [sub_func(j) for j in 1:U]
    return [g]
end

function get_P(ipts, d::Int, U::Int)
    (npts, n) = size(ipts)
    u = Hypatia.calc_u(n, 2d, ipts)
    m = Vector{Float64}(undef, U)
    m[1] = 2^n
    M = Matrix{Float64}(undef, npts, U)
    M[:,1] .= 1.0

    col = 1
    for t in 1:2d
        for xp in Combinatorics.multiexponents(n, t)
            col += 1
            if any(isodd, xp)
                m[col] = 0.0
            else
                m[col] = m[1]/prod(1.0 - abs2(xp[j]) for j in 1:n)
            end
            @. @views M[:,col] = u[1][:,xp[1]+1]
            for j in 2:n
                @. @views M[:,col] *= u[j][:,xp[j]+1]
            end
        end
    end
    return M
end
