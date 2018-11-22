#=
Copyright 2018, Chris Coey and contributors

see description in examples/namedpoly/native.jl
=#
using Random
include(joinpath(dirname(@__DIR__()), "domains.jl"))

Random.seed!(1234)

function build_JuMP_namedpoly_SDP(
    x,
    f::DynamicPolynomials.Polynomial, #{true,Float64},
    dom::InterpDomain;
    d::Int = div(maxdegree(f) + 1, 2),
    )

    # build domain BasicSemialgebraicSet representation
    bss = get_bss(dom, x)

    # build JuMP model
    model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, a)
    @objective(model, Max, a)
    @constraint(model, fnn, f >= a, domain=bss, maxdegree=2d)

    return model
end

function build_JuMP_namedpoly_WSOS(
    x,
    f::DynamicPolynomials.Polynomial, #{true,Float64},
    dom::InterpDomain;
    d::Int = div(maxdegree(f) + 1, 2),
    pts_factor = nvariables(f),
    )

    n = nvariables(f) # number of polyvars
    L = binomial(n+d,n)
    U = binomial(n+2d, n)

    # toggle between new sampling and current Hypatia method, this is just for debugging purposes
    sample_pts = true

    if sample_pts
        candidate_pts = interp_sample(dom, U * pts_factor)
        M = get_P(candidate_pts, d, U)

        Mp = Array(M')
        F = qr!(Mp, Val(true))
        keep_pnt = F.p[1:U]

        pts = candidate_pts[keep_pnt,:] # subset of points indexed with the support of w
        P0 = M[keep_pnt,1:L] # subset of polynomial evaluations up to total degree d
        P = Array(qr(P0).Q)

    # this is just for debugging purposes
    else
        L = binomial(n+d, n)
        (L, U, pts, P0, P, w) = Hypatia.interpolate(n, d, calc_w=false)
        pts .*= 2.0
    end

    P0sub = view(P0, :, 1:binomial(n+d-1, n))

    bss = get_bss(dom, x)

    g = get_weights(dom, bss, pts)
    @assert length(g) == length(bss.p)
    PWts = [sqrt.(gi) .* P0sub for gi in g]

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
        [i in 1:U], f(pts[i,:]) == q[i]
    end)

    return model
end

function run_JuMP_namedpoly()
    # select the named polynomial to minimize
    polynames = [
        :butcher
        :butcher_ball
        :butcher_ellipsoid
        :caprasse
        :caprasse_ball
        :goldsteinprice
        :goldsteinprice_ball
        :goldsteinprice_ellipsoid # it is just a ball
        :heart
        :heart_ellipsoid #TODO tighter ellispoid
        :lotkavolterra
        :lotkavolterra_ball
        :magnetism7
        :magnetism7_ball
        :motzkin
        :motzkin_ball1
        # :motzkin_ball2  # runs into issues
        :reactiondiffusion
        :reactiondiffusion_ball
        :robinson
        :robinson_ball
        :rosenbrock
        :rosenbrock_ball
        :schwefel
        :schwefel_ball
    ]

    # some polynomials run into predictor/corrector failusres with SDP but not WSOS formulation
    sdp_skip = [:butcher_ball, :goldsteinprice_ball, :goldsteinprice_ellipsoid, :heart, :heart_ellipsoid]

    for polyname in polynames

        println(polyname)

        # get data for named polynomial
        (x, f, dom, truemin) = getpolydata(polyname)

        # some polynomials run into predictor/corrector failusres with SDP but not WSOS formulation, skipping these with SDP only
        if !(polyname in sdp_skip)
            println("solving model with PSD cones")

            if polyname in [:butcher, :butcher_ball, :caprasse_ball, :lotkavolterra, :lotkavolterra_ball]
                d = default_degrees[polyname]
                model = build_JuMP_namedpoly_SDP(x, f, dom, d=d)
            else
                model = build_JuMP_namedpoly_SDP(x, f, dom)
            end
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
        end

        println("solving model with WSOS interpolation cones")
        d = default_degrees[polyname]
        if polyname == :heart
            model = build_JuMP_namedpoly_WSOS(x, f, dom, d=d, pts_factor=3)
        else
            model = build_JuMP_namedpoly_WSOS(x, f, dom, d=d, pts_factor=4*length(x))
        end
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
    end

    return nothing
end



default_degrees = Dict(
    :butcher => 2,
    :butcher_ball => 2,
    :butcher_ellipsoid => 2,
    :caprasse => 4,
    :caprasse_ball => 4,
    :goldsteinprice => 7,
    :goldsteinprice_ball => 7,
    :goldsteinprice_ellipsoid => 7,
    :heart => 2,
    :heart_ellipsoid => 2,
    :lotkavolterra => 3,
    :lotkavolterra_ball => 3,
    :magnetism7 => 2,
    :magnetism7_ball => 2,
    :motzkin => 7,
    :motzkin_ball1 => 7,
    :reactiondiffusion => 3,
    :reactiondiffusion_ball => 3,
    :robinson => 8,
    :robinson_ball => 8,
    :rosenbrock => 4,
    :rosenbrock_ball => 4,
    :schwefel => 3,
    :schwefel_ball => 3,
)



function getpolydata(polyname::Symbol)
    if polyname == :butcher
        @polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        dom = Box([-1,-0.1,-0.1,-1,-0.1,-0.1], [0,0.9,0.5,-0.1,-0.05,-0.03])
        truemin = -1.4393333333
    elseif polyname == :butcher_ball
        @polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        radius = maximum(0.5 * (-[-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])) * sqrt(6)
        centers = 0.5 * ([-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])
        dom = Ball(centers, radius)
        truemin = -4.10380
    elseif polyname == :butcher_ellipsoid
        @polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        # heuristically attained enclosing ellipsoid
        semiradii_sqrd = 6 * (0.5 * (-[-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])).^2
        centers = 0.5 * ([-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])
        Q = Diagonal(semiradii_sqrd)
        dom = Ellipsoid(centers, Q)
        truemin = -16.7378208
    elseif polyname == :caprasse
        @polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = Box(fill(-0.5, 4), fill(0.5, 4))
        truemin = -3.1800966258
    elseif polyname == :caprasse_ball
        @polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        # 0.5*sqrt(4) = 1
        dom = Ball(fill(0.0, 4), 1.0)
        truemin = -9.47843346
    elseif polyname == :goldsteinprice
        @polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = Box(fill(-2, 2), fill(2, 2))
        truemin = 3
    elseif polyname == :goldsteinprice_ball
        @polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = Ball(fill(0.0, 2), 2.0 * sqrt(2))
        truemin = 3
    elseif polyname == :goldsteinprice_ellipsoid
        @polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        centers = fill(0, 2)
        semiradii = fill(0.25, 2)
        Q = Diagonal(semiradii)
        dom = Ellipsoid(centers, Q)
        truemin = 3
    elseif polyname == :heart
        @polyvar x[1:8]
        f = x[1]*x[6]^3-3x[1]*x[6]*x[7]^2+x[3]*x[7]^3-3x[3]*x[7]*x[6]^2+x[2]*x[5]^3-3*x[2]*x[5]*x[8]^2+x[4]*x[8]^3-3x[4]*x[8]*x[5]^2+0.9563453
        dom = Box([-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], [0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3])
        truemin = -1.36775
    elseif polyname == :heart_ellipsoid
        @polyvar x[1:8]
        f = x[1]*x[6]^3-3x[1]*x[6]*x[7]^2+x[3]*x[7]^3-3x[3]*x[7]*x[6]^2+x[2]*x[5]^3-3*x[2]*x[5]*x[8]^2+x[4]*x[8]^3-3x[4]*x[8]*x[5]^2+0.9563453
        # heuristically attained enclosing ellipsoid
        semiradii_sqrd = 8.0 * (0.5 * (-[-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1] + [0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3])).^2
        centers = 0.5 * ([-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1] + [0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3])
        Q = Diagonal(semiradii_sqrd)
        dom = Ellipsoid(centers, Q)
        truemin = -306.46395
    elseif polyname == :lotkavolterra
        @polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = Box(fill(-2, 4), fill(2, 4))
        truemin = -20.8
    elseif polyname == :lotkavolterra_ball
        @polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = Ball(fill(0.0, 4), 4.0)
        truemin = -21.13744
    elseif polyname == :magnetism7
        @polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = Box(fill(-1, 7), fill(1, 7))
        truemin = -0.25
    elseif polyname == :magnetism7_ball
        @polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = Ball(fill(0.0, 7), sqrt(7))
        truemin = -0.25
    elseif polyname == :motzkin
        @polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = Box(fill(-1, 2), fill(1, 2))
        truemin = 0
    elseif polyname == :motzkin_ball1
        @polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = Ball([0.0; 0.0], sqrt(2.0))
        truemin = 0
    elseif polyname == :motzkin_ball2
        @polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = Ball([0.0; 0.0], 0.5*sqrt(0.5))
        truemin = 0.84375
    elseif polyname == :reactiondiffusion
        @polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = Box(fill(-5, 3), fill(5, 3))
        truemin = -36.71269068
    elseif polyname == :reactiondiffusion_ball
        @polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = Ball(fill(0.0, 3), 5*sqrt(3))
        truemin = -73.31
    elseif polyname == :robinson
        @polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = Box(fill(-1, 2), fill(1, 2))
        truemin = 0.814814
    elseif polyname == :robinson_ball
        @polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = Ball(fill(0.0, 2), sqrt(2.0))
        truemin = 0.814814
    elseif polyname == :rosenbrock
        @polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = Box(fill(-5, 2), fill(10, 2))
        truemin = 0
    elseif polyname == :rosenbrock_ball
        @polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = Ball(fill(2.5, 2), 7.5*sqrt(2.0))
        truemin = 0
    elseif polyname == :schwefel
        @polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = Box(fill(-10, 3), fill(10, 3))
        truemin = 0
    elseif polyname == :schwefel_ball
        @polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = Ball(fill(0.0, 3), 10.0 * sqrt(3))
        truemin = 0
    else
        error("poly $polyname not recognized")
    end

    return (x, f, dom, truemin)
end



run_JuMP_namedpoly()
