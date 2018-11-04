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

function build_JuMP_namedpoly_PSD(
    x,
    f::DynamicPolynomials.Polynomial{true,Float64},
    dom::Box,
    )
    @show typeof(x)

    # build domain BasicSemialgebraicSet representation
    bss = BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}()
    for i in 1:dimension(dom)
        addinequality!(bss, -x[i] + dom.u[i])
        addinequality!(bss, x[i] - dom.l[i])
    end

    # build JuMP model
    model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, a)
    @objective(model, Max, a)
    @constraint(model, fnn, f >= a, domain=bss)

    return model
end

function build_JuMP_namedpoly_interp(
    x,
    f::DynamicPolynomials.Polynomial{true,Float64},
    dom::Domain,
    )
    # generate interpolation
    n = nvariables(f) # number of polyvars
    d = degree(f) # let degree of interpolating polynomials match degree of given named polynomial
    L = binomial(n+d, n)
    U = binomial(n+2d, n)
    pts = sample(dom, U)







    # build JuMP model
    model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, a)
    @objective(model, Max, a)
    @constraint(model, fnn, f >= a, domain=dom)

    return model
end

function run_JuMP_namedpoly()
    # select the named polynomial to minimize
    polyname =
        # :butcher
        # :caprasse
        # :goldsteinprice
        # :heart
        # :lotkavolterra
        # :magnetism7
        # :motzkin
        :reactiondiffusion
        # :robinson
        # :rosenbrock
        # :schwefel

    # get data for named polynomial
    (f, dom, truemin) = getpolydata(polyname)

    println("solving model with PSD cones")
    model = build_JuMP_namedpoly_PSD(f, dom)
    JuMP.optimize!(model)

    println("solving model with WSOS interpolation cones")
    model = build_JuMP_namedpoly_PSD(f, dom)
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


abstract type Domain end

struct Box <: Domain
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

dimension(d::Box) = length(d.l)

function sample(d::Box, npts::Int)
    dim = dimension(d)
    pts = rand(dim, npts)
    return (d.u + d.l)/2.0 .+ (d.u - d.l) .* (pts .- 0.5)
end

# struct BoxSurf <: Domain
#     l::Vector{Float64}
#     u::Vector{Float64}
# end
#
# dimension(d::Box) = length(d.l)
#
# struct Ball <: Domain
#     c::Vector{Float64}
#     r::Float64
# end
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






function getpolydata(polyname::Symbol)
    if polyname == :butcher
        @polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        dom = Box([-1,-0.1,-0.1,-1,-0.1,-0.1], [0,0.9,0.5,-0.1,-0.05,-0.03])
        truemin = -1.4393333333
    elseif polyname == :caprasse
        @polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = Box(fill(-0.5, 4), fill(0.5, 4))
        truemin = -3.1800966258
    elseif polyname == :goldsteinprice
        @polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = Box(fill(-2, 2), fill(2, 2))
        truemin = 3
    elseif polyname == :heart
        @polyvar x[1:8]
        f = x[1]*x[6]^3-3x[1]*x[6]*x[7]^2+x[3]*x[7]^3-3x[3]*x[7]*x[6]^2+x[2]*x[5]^3-3*x[2]*x[5]*x[8]^2+x[4]*x[8]^3-3x[4]*x[8]*x[5]^2
        dom = Box([-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], [0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3])
        truemin = -1.36775
    elseif polyname == :lotkavolterra
        @polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = Box(fill(-2, 4), fill(2, 4))
        truemin = -20.8
    elseif polyname == :magnetism7
        @polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = Box(fill(-1, 7), fill(1, 7))
        truemin = -0.25
    elseif polyname == :motzkin
        @polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = Box(fill(-1, 2), fill(1, 2))
        truemin = 0
    elseif polyname == :reactiondiffusion
        @polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = Box(fill(-5, 3), fill(5, 3))
        truemin = -36.71269068
    elseif polyname == :robinson
        @polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = Box(fill(-1, 2), fill(1, 2))
        truemin = 0.814814
    elseif polyname == :rosenbrock
        @polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = Box(fill(-5, 2), fill(10, 2))
        truemin = 0
    elseif polyname == :schwefel
        @polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = Box(fill(-10, 3), fill(10, 3))
        truemin = 0
    else
        error("poly $polyname not recognized")
    end

    return (f, dom, truemin)
end



run_JuMP_namedpoly()
