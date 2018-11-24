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
using Distributions
using Test

import Combinatorics
import GSL: sf_gamma_inc_Q

abstract type InterpDomain end

mutable struct Box <: InterpDomain
    l::Vector{Float64}
    u::Vector{Float64}
    function Box(l::Vector{T}, u::Vector{T}) where {T<:Real}
        @assert length(l) == length(u)
        d = new()
        d.l = Float64.(l)
        d.u = Float64.(u)
        return d
    end
end

mutable struct Ball <: InterpDomain
    c::Vector{Float64}
    r::Float64
    function Ball(c::Vector{T}, r::S) where {T<:Real, S<:Real}
        d = new()
        d.c = Float64.(c)
        d.r = Float64(r)
        return d
    end
end

# (x-c)'Q(x-c) \leq 1
mutable struct Ellipsoid <: InterpDomain
    c::Vector{Float64}
    Q::AbstractMatrix{Float64}
    function Ellipsoid(c::Vector{T}, Q::AbstractMatrix{S}) where {T<:Real, S<:Real}
        @assert isposdef(Q)
        @assert length(c) == size(Q, 1)
        d = new()
        d.c = Float64.(c)
        d.Q = Float64.(Q)
        return d
    end
end

dimension(d::Box) = length(d.l)
dimension(d::Ball) = length(d.c)
dimension(d::Ellipsoid) = length(d.c)

function interp_sample(d::Box, npts::Int)
    dim = dimension(d)
    pts = rand(npts, dim) .- 0.5
    shift = (d.u + d.l)/2.0
    for i in 1:npts
        pts[i,:] = pts[i,:] .* (d.u - d.l) + shift
    end
    return pts
end
function interp_sample(d::Ball, npts::Int)
    dim = dimension(d)
    pts = randn(npts, dim)
    norms = sum(pts.^2, dims=2)
    pts .*= d.r ./ sqrt.(norms)
    # sf_gamma_inc_Q is the normalized incomplete gamma function
    pts .*= sf_gamma_inc_Q.(norms/2, dim/2).^(1/dim)
    for i in 1:dim
        pts[:, i] .+= d.c[i]
    end
    return pts
end
function interp_sample(d::Ellipsoid, npts::Int)
    dim = dimension(d)
    fchol = cholesky(inv(d.Q))
    pts = randn(npts, dim)
    norms = sum(pts.^2, dims=2)
    for i in 1:npts
        pts[i,:] ./= sqrt(norms[i])
    end
    # sf_gamma_inc_Q is the normalized incomplete gamma function
    pts .*= sf_gamma_inc_Q.(norms/2, dim/2).^(1/dim)
    # rotate/scale
    for i in 1:npts
        @assert norm(pts[i,:]) < 1.0
        pts[i,:] = fchol.L * pts[i,:]
    end
    # shift
    for i in 1:dim
        pts[:, i] .+= d.c[i]
    end
    return pts
end

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
function get_bss(dom::Ellipsoid, x)
    return @set (x - dom.c)' * dom.Q * (x - dom.c) <= 1
end

function get_weights(
    dom::Box,
    pts::AbstractArray{Float64},
    idxs::UnitRange{Int}=1:size(pts, 2)
    )

    m = length(idxs)
    @assert m == length(dom.l) == length(dom.u)
    U = size(pts, 1)
    g = Vector{Vector{Float64}}(undef, m)
    for i in 1:m
        g[i] = (pts[:,i] .- dom.l[i]) .* (dom.u[i] .- pts[:,i])
    end
    return g
end
function get_weights(
    dom::Ball,
    pts::AbstractArray{Float64},
    idxs::UnitRange{Int}=1:size(pts, 2),
    )

    U = size(pts, 1)
    @assert length(idxs) == length(dom.c)
    sub_func(j) = dom.r^2 - sum((pts[j, idxs] - dom.c).^2)
    g = [sub_func(j) for j in 1:U]
    return [g]
end
function get_weights(
    dom::Ellipsoid,
    pts::AbstractArray{Float64},
    idxs::UnitRange{Int}=1:size(pts, 2),
    )

    U = size(pts, 1)
    @assert length(idxs) == length(dom.c)
    sub_func(j) = 1.0 - (pts[j, idxs] - dom.c)' * dom.Q * (pts[j, idxs] - dom.c)
    g = [sub_func(j) for j in 1:U]
    return [g]
end

function get_large_P(ipts, d::Int, U::Int)
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
