#=
Copyright 2018, Chris Coey and contributors

see description in examples/namedpoly/native.jl
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
using Test

import Combinatorics

abstract type Domain end

mutable struct Box <: Domain
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
# struct Ellipsoid <: Domain
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
struct Ball <: Domain
    c::Vector{Float64}
    r::Float64
    function Ball(c::Vector{Float64}, r::Float64)
        @assert length(c) == length(r)
        d = new()
        d.c = c
        d.r = r
        return d
    end
end

dimension(d::Box) = length(d.l)
dimension(d::Ball) = length(d.c)

function sample(d::Box, npts::Int)
    dim = dimension(d)
    pts = rand(dim, npts)
    return (d.u + d.l)/2.0 .+ (d.u - d.l) .* (pts .- 0.5)
end

# function sample(d::Ellipsoid, npts::Int)
#     dim = dimension(d)
#     random_radii = rand(dim, npts) * d.r
#     random_thetas = rand(dim, npts) * 2pi
#     pts = random_radii * cos.(random_thetas)
#     return (d.u + d.l)/2.0 .+ (d.u - d.l) .* (pts .- 0.5)
# end

function sample(d::Ball, npts::Int)
    dim = dimension(d)
    pts = rand(dim, npts)
    for i in 1:npts
        pts[:, i] .= pts[:, i] / norm(pts) * d.r + d.c
    end
    return pts
end

# struct BoxSurf <: Domain
#     l::Vector{Float64}
#     u::Vector{Float64}
# end
#
# dimension(d::Box) = length(d.l)
#
#
# struct BallSurf <: Domain
#     c::Vector{Float64}
#     r::Float64
# end
#
# struct Ellipse <: Domain
#     c::Vector{Float64}
#     Q::Matrix{Float64}
# end
#
# struct EllipseSurf <: Domain
#     c::Vector{Float64}
#     Q::Matrix{Float64}
# end


get_bss(dom::Domain, x) = error("")

function get_bss(dom::Box, x)
    bss = BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}()
    for i in 1:dimension(dom)
        addinequality!(bss, -x[i] + dom.u[i])
        addinequality!(bss, x[i] - dom.l[i])
    end
    return bss
end

# function get_domain(dom::Ellipsoid, x)
#     bss = BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}()
#     lhs = (x - dom.c)' * dom.Q * (x - dom.c)
#     addinequality!(bss, 1 - lhs)
#     return bss
# end

function get_bss(dom::Ball, x)
    return @set abs2(x - dom.c) <= dom.r^2
end

# assumes we built the box ourselves for now, with known order of equations
function get_weights(::Box, bss::BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}, pts)
    m = length(bss.p)
    U = size(pts, 2)
    @assert mod(m, 2) == 0
    gpolys = Vector{Polynomial{true,Float64}}(undef, div(m, 2))
    for i in eachindex(gpolys)
        gpolys[i] = bss.p[2i-1] * bss.p[2i]
    end
    g = Vector{Vector{Float64}}(undef, length(gpolys))
    for i in eachindex(gpolys)
        g[i] = gpolys[i].(pts[i,:])
        @assert all(g[i] .> -1e-6)
    end
    return g
end

function get_P(ipts, d::Int, U::Int)
    (n, npts) = size(ipts)
    L = binomial(n+d,n)
    u = Hypatia.calc_u(n, 2d, ipts')
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
    P0 = M[:,1:L]
    P = Array(qr(P0).Q)
    return (P0, P)
end

function build_JuMP_namedpoly_SDP(
    x,
    f::DynamicPolynomials.Polynomial, #{true,Float64},
    dom::Domain,
    d::Int = div(length(x), 2),
    )

    # build domain BasicSemialgebraicSet representation
    bss = get_bss(dom, x)

    # build JuMP model
    model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, a)
    @objective(model, Max, a)
    @constraint(model, fnn, f >= a, domain=bss)

    return model
end

function build_JuMP_namedpoly_WSOS(
    x,
    f::DynamicPolynomials.Polynomial, #{true,Float64},
    dom::Domain,
    d::Int = div(length(x), 2),
    )

    n = nvariables(f) # number of polyvars
    U = binomial(n+2d, n)

    # toggle between sampling and current Hypatia method
    sample_pts = true

    if sample_pts
        pts = sample(dom, U)
        (P0, P) = get_P(pts, d, U)
    else
        L = binomial(n+d, n)
        (L, U, pts, P0, P, w) = Hypatia.interpolate(n, d, calc_w=false)
        pts = pts'
        pts .*= 2.0
    end

    P0sub = view(P0, :, 1:binomial(n+d-1, n))

    bss = get_bss(dom, x)

    g = get_weights(dom, bss, pts)
    PWts = [sqrt.(g[i]) .* P0sub for i in 1:n]

    wsos_cone = WSOSPolyInterpCone(U, [P, PWts...])

    # build JuMP model
    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variables(model, begin
        a
        q[1:U]
    end)
    @objective(model, Max, a)
    @constraints(model, begin
        q - a .* ones(U) in wsos_cone
        [i in 1:U], f(pts[:,i]) == q[i]
    end)

    return model
end

function run_JuMP_namedpoly()
    # select the named polynomial to minimize
    polyname =
        # :butcher
        # :caprasse
        :goldsteinprice
        # :heart
        # :lotkavolterra
        # :magnetism7
        # :motzkin
        # :reactiondiffusion
        # :robinson
        # :rosenbrock
        # :schwefel

    # get data for named polynomial
    (x, f, dom, truemin, d) = getpolydata(polyname)

    println("solving model with PSD cones")
    model = build_JuMP_namedpoly_SDP(x, f, dom, d)
    JuMP.optimize!(model)

    println("done")
    term_status = JuMP.termination_status(model)
    pobj = JuMP.objective_value(model)
    dobj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4

    println("solving model with WSOS interpolation cones")
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d)
    JuMP.optimize!(model)

    println("done")
    term_status = JuMP.termination_status(model)
    pobj = JuMP.objective_value(model)
    dobj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4

    return nothing
end








function getpolydata(polyname::Symbol)
    if polyname == :butcher
        @polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        dom = Box([-1,-0.1,-0.1,-1,-0.1,-0.1], [0,0.9,0.5,-0.1,-0.05,-0.03])
        truemin = -1.4393333333
        d = 2
    elseif polyname == :caprasse
        @polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = Box(fill(-0.5, 4), fill(0.5, 4))
        truemin = -3.1800966258
        d = 4
    elseif polyname == :goldsteinprice
        @polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = Box(fill(-2, 2), fill(2, 2))
        truemin = 3
        d = 7
    elseif polyname == :heart
        @polyvar x[1:8]
        f = x[1]*x[6]^3-3x[1]*x[6]*x[7]^2+x[3]*x[7]^3-3x[3]*x[7]*x[6]^2+x[2]*x[5]^3-3*x[2]*x[5]*x[8]^2+x[4]*x[8]^3-3x[4]*x[8]*x[5]^2
        dom = Box([-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], [0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3])
        truemin = -1.36775
        d = 2
    elseif polyname == :lotkavolterra
        @polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = Box(fill(-2, 4), fill(2, 4))
        truemin = -20.8
        d = 3
    elseif polyname == :magnetism7
        @polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = Box(fill(-1, 7), fill(1, 7))
        truemin = -0.25
        d = 2
    elseif polyname == :motzkin
        @polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = Box(fill(-1, 2), fill(1, 2))
        truemin = 0
        d = 7
    elseif polyname == :reactiondiffusion
        @polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = Box(fill(-5, 3), fill(5, 3))
        truemin = -36.71269068
        d = 3
    elseif polyname == :robinson
        @polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = Box(fill(-1, 2), fill(1, 2))
        truemin = 0.814814
        d = 8
    elseif polyname == :rosenbrock
        @polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = Box(fill(-5, 2), fill(10, 2))
        truemin = 0
        d = 4
    elseif polyname == :schwefel
        @polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = Box(fill(-10, 3), fill(10, 3))
        truemin = 0
        d = 3
    else
        error("poly $polyname not recognized")
    end

    return (x, f, dom, truemin, d)
end



run_JuMP_namedpoly()
