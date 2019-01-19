#=
Copyright 2018, Chris Coey and contributors

list of predefined polynomials from various applications
see https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html

TODO this data should not be duplicated
=#

polys = Dict{Symbol,NamedTuple}(
    :butcher => (n=6, lbs=[-1.0,-0.1,-0.1,-1.0,-0.1,-0.1], ubs=[0.0,0.9,0.5,-0.1,-0.05,-0.03], deg=3,
        fn=((u,v,w,x,y,z) -> z*v^2+y*w^2-u*x^2+x^3+x^2-(1/3)*u+(4/3)*x)
        ),
    :caprasse => (n=4, lbs=-0.5*ones(4), ubs=0.5*ones(4), deg=8,
        fn=((w,x,y,z) -> -w*y^3+4x*y^2*z+4w*y*z^2+2x*z^3+4w*y+4y^2-10x*z-10z^2+2)
        ),
    :goldsteinprice => (n=2, lbs=-2.0*ones(2), ubs=2.0*ones(2), deg=8,
        fn=((x,y) -> (1+(x+y+1)^2*(19-14x+3x^2-14y+6x*y+3y^2))*(30+(2x-3y)^2*(18-32x+12x^2+48y-36x*y+27y^2)))
        ),
    :heart => (n=8, lbs=[-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], ubs=[0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3], deg=4,
        fn=((s,t,u,v,w,x,y,z) -> s*x^3-3s*x*y^2+u*y^3-3u*y*x^2+t*w^3-3*t*w*z^2+v*z^3-3v*z*w^2+0.9563453)
        ),
    :lotkavolterra => (n=4, lbs=-2.0*ones(4), ubs=2.0*ones(4), deg=3,
        fn=((w,x,y,z) -> w*(x^2+y^2+z^2-1.1)+1)
        ),
    :magnetism7 => (n=7, lbs=-ones(7), ubs=ones(7), deg=2,
        fn=((t,u,v,w,x,y,z) -> t^2+2u^2+2v^2+2w^2+2x^2+2y^2+2z^2-t)
        ),
    :motzkin => (n=2, lbs=-ones(2), ubs=ones(2), deg=6,
        fn=((x,y) -> 1-48x^2*y^2+64x^2*y^4+64x^4*y^2)
        ),
    :reactiondiffusion => (n=3, lbs=-5.0*ones(3), ubs=5.0*ones(3), deg=2,
        fn=((x,y,z) -> -x+2y-z-0.835634534y*(1+y))
        ),
    :robinson => (n=2, lbs=-ones(2), ubs=ones(2), deg=6,
        fn=((x,y) -> 1+x^6+y^6-x^4*y^2+x^4-x^2*y^4+y^4-x^2+y^2+3x^2*y^2)
        ),
    :rosenbrock => (n=2, lbs=-5.0*ones(2), ubs=10.0*ones(2), deg=4,
        fn=((x,y) -> (1-x)^2+100*(x^2-y)^2)
        ),
    :schwefel => (n=3, lbs=-10.0*ones(3), ubs=10.0*ones(3), deg=4,
        fn=((x,y,z) -> (x-y^2)^2+(y-1)^2+(x-z^2)^2+(z-1)^2)
        ),
)

function getpolydata(polyname::Symbol)
    if polyname == :butcher
        DynamicPolynomials.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        dom = MU.Box([-1,-0.1,-0.1,-1,-0.1,-0.1], [0,0.9,0.5,-0.1,-0.05,-0.03])
        truemin = -1.4393333333
    elseif polyname == :butcher_ball
        DynamicPolynomials.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        axes = 0.5 * ([0,0.9,0.5,-0.1,-0.05,-0.03] - [-1,-0.1,-0.1,-1,-0.1,-0.1])
        centers = 0.5 * ([-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])
        dom = MU.Ball(centers, sqrt(6) * maximum(axes))
        truemin = -4.10380
    elseif polyname == :butcher_ellipsoid
        DynamicPolynomials.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        # heuristically-obtained enclosing ellipsoid
        centers = 0.5 * ([-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])
        Q = Diagonal(6 * abs2.(centers))
        dom = MU.Ellipsoid(centers, Q)
        truemin = -16.7378208
    elseif polyname == :caprasse
        DynamicPolynomials.@polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = MU.Box(-0.5*ones(4), 0.5*ones(4))
        truemin = -3.1800966258
    elseif polyname == :caprasse_ball
        DynamicPolynomials.@polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = MU.Ball(zeros(4), 1.0)
        truemin = -9.47843346
    elseif polyname == :goldsteinprice
        DynamicPolynomials.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = MU.Box(-2*ones(2), 2*ones(2))
        truemin = 3
    elseif polyname == :goldsteinprice_ball
        DynamicPolynomials.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = MU.Ball(zeros(2), 2*sqrt(2))
        truemin = 3
    elseif polyname == :goldsteinprice_ellipsoid
        DynamicPolynomials.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        centers = zeros(2)
        Q = Diagonal(0.25*ones(2))
        dom = MU.Ellipsoid(centers, Q)
        truemin = 3
    elseif polyname == :heart
        DynamicPolynomials.@polyvar x[1:8]
        f = x[1]*x[6]^3-3x[1]*x[6]*x[7]^2+x[3]*x[7]^3-3x[3]*x[7]*x[6]^2+x[2]*x[5]^3-3*x[2]*x[5]*x[8]^2+x[4]*x[8]^3-3x[4]*x[8]*x[5]^2+0.9563453
        dom = MU.Box([-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], [0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3])
        truemin = -1.36775
    elseif polyname == :lotkavolterra
        DynamicPolynomials.@polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = MU.Box(-2*ones(4), 2*ones(4))
        truemin = -20.8
    elseif polyname == :lotkavolterra_ball
        DynamicPolynomials.@polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = MU.Ball(zeros(4), 4.0)
        truemin = -21.13744
    elseif polyname == :magnetism7
        DynamicPolynomials.@polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = MU.Box(-ones(7), ones(7))
        truemin = -0.25
    elseif polyname == :magnetism7_ball
        DynamicPolynomials.@polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = MU.Ball(zeros(7), sqrt(7))
        truemin = -0.25
    elseif polyname == :motzkin
        DynamicPolynomials.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = MU.Box(-ones(2), ones(2))
        truemin = 0
    elseif polyname == :motzkin_ball
        DynamicPolynomials.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = MU.Ball(zeros(2), sqrt(2))
        truemin = 0
    elseif polyname == :motzkin_ellipsoid
        # ellipsoid contains two local minima in opposite orthants
        DynamicPolynomials.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        Q = [1 1; 1 -1]
        D = [1 0; 0 0.1]
        S = Q * D * Q
        dom = MU.Ellipsoid(zeros(2), S)
        truemin = 0
    elseif polyname == :reactiondiffusion
        DynamicPolynomials.@polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = MU.Box(-5*ones(3), 5*ones(3))
        truemin = -36.71269068
    elseif polyname == :reactiondiffusion_ball
        DynamicPolynomials.@polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = MU.Ball(zeros(3), 5*sqrt(3))
        truemin = -73.31
    elseif polyname == :robinson
        DynamicPolynomials.@polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = MU.Box(-ones(2), ones(2))
        truemin = 0.814814
    elseif polyname == :robinson_ball
        DynamicPolynomials.@polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = MU.Ball(zeros(2), sqrt(2))
        truemin = 0.814814
    elseif polyname == :rosenbrock
        DynamicPolynomials.@polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = MU.Box(-5*ones(2), 10*ones(2))
        truemin = 0
    elseif polyname == :rosenbrock_ball
        DynamicPolynomials.@polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = MU.Ball(2.5*ones(2), 7.5*sqrt(2))
        truemin = 0
    elseif polyname == :schwefel
        DynamicPolynomials.@polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = MU.Box(-10*ones(3), 10*ones(3))
        truemin = 0
    elseif polyname == :schwefel_ball
        DynamicPolynomials.@polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = MU.Ball(zeros(3), 10*sqrt(3))
        truemin = 0
    else
        error("poly $polyname not recognized")
    end

    return (x, f, dom, truemin)
end
