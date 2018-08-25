#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/polyOpt.m
formulates and solves the polynomial optimization problem for a given polynomial, described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Alfonso
using LinearAlgebra

# list of currently available named polynomials, see https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
const polys = Dict{Symbol,NamedTuple}(
    :butcher => (n=6, lbs=[-1,-0.1,-0.1,-1,-0.1,-0.1], ubs=[0,0.9,0.5,-0.1,-0.05,-0.03], deg=3,
        fn=((u,v,w,x,y,z) -> z*v^2+y*w^2-u*x^2+x^3+x^2-(1/3)*u+(4/3)*x)
        ),
    :caprasse => (n=4, lbs=fill(-1/2,4), ubs=fill(1/2,4), deg=8,
        fn=((w,x,y,z) -> -w*y^3+4x*y^2*z+4w*y*z^2+2x*z^3+4w*y+4y^2-10x*z-10z^2+2)
        ),
    :goldsteinprice => (n=2, lbs=fill(-2,2), ubs=fill(2,2), deg=8,
        fn=((x,y) -> (1+(x+y+1)^2*(19-14x+3x^2-14y+6x*y+3y^2))*(30+(2x-3y)^2*(18-32x+12x^2+48y-36x*y+27y^2)))
        ),
    :heart => (n=8, lbs=[-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], ubs=[0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3], deg=4,
        fn=((s,t,u,v,w,x,y,z) -> z*v^2+y*w^2-u*x^2+x^3+x^2-(1/3)*u+(4/3)*x)
        ),
    :lotkavolterra => (n=4, lbs=fill(-2,4), ubs=fill(2,4), deg=3,
        fn=((w,x,y,z) -> w*(x^2+y^2+z^2-1.1)+1)
        ),
    :magnetism7 => (n=7, lbs=fill(-1,7), ubs=fill(1,7), deg=2,
        fn=((t,u,v,w,x,y,z) -> t^2+2u^2+2v^2+2w^2+2x^2+2y^2+2z^2-t)
        ),
    :motzkin => (n=2, lbs=fill(-1,2), ubs=fill(1,2), deg=6,
        fn=((x,y) -> 1-48x^2*y^2+64x^2*y^4+64x^4*y^2)
        ),
    :reactiondiffusion => (n=3, lbs=fill(-5,3), ubs=fill(5,3), deg=2,
        fn=((x,y,z) -> -x+2y-z-0.835634534y*(1+y))
        ),
    :robinson => (n=2, lbs=fill(-1,2), ubs=fill(1,2), deg=6,
        fn=((x,y) -> 1+x^6+y^6-x^4*y^2+x^4-x^2*y^4+y^4-x^2+y^2+3x^2*y^2)
        ),
    :rosenbrock => (n=2, lbs=fill(-5,2), ubs=fill(10,2), deg=4,
        fn=((x,y) -> (1-x)^2+100*(x^2-y)^2)
        ),
    :schwefel => (n=3, lbs=fill(-10,3), ubs=fill(10,3), deg=4,
        fn=((x,y,z) -> (x-y^2)^2+(y-1)^2+(x-z^2)^2+(z-1)^2)
        ),
)

function build_namedpoly!(alf::Alfonso.AlfonsoOpt, polyname::Symbol, d::Int)
    # get data for named polynomial
    (n, lbs, ubs, deg, fn) = polys[polyname]
    if d < ceil(Int, deg/2)
        error("requires d >= $(ceil(Int, deg/2))")
    end

    # generate interpolation
    (L, U, pts, P0, P, w) = Alfonso.interpolate(n, d, calc_w=false)

    # transform points to fit the box domain
    pts .*= (ubs - lbs)'/2
    pts .+= (ubs + lbs)'/2
    wtVals = (pts .- lbs') .* (ubs' .- pts)
    LWts = fill(binomial(n+d-1, n), n)
    PWts = [Diagonal(sqrt.(wtVals[:,j]))*P0[:,1:LWts[j]] for j in 1:n]

    # set up problem data
    A = ones(1, U)
    b = [1.0,]
    c = [fn(pts[j,:]...) for j in 1:U]
    cone = Alfonso.Cone([Alfonso.SumOfSquaresCone(U, [P0, PWts...]),], AbstractUnitRange[1:U,])

    return Alfonso.load_data!(alf, A, b, c, cone)
end

# alf = Alfonso.AlfonsoOpt(maxiter=100, verbose=false)

# select the named polynomial to minimize and the SOS degree (to be squared)
# build_namedpoly!(alf, :butcher, 2)
# build_namedpoly!(alf, :caprasse, 4)
# build_namedpoly!(alf, :goldsteinprice, 7)
# build_namedpoly!(alf, :heart, 2)
# build_namedpoly!(alf, :lotkavolterra, 3)
# build_namedpoly!(alf, :magnetism7, 2)
# build_namedpoly!(alf, :motzkin, 7)
# build_namedpoly!(alf, :reactiondiffusion, 4)
# build_namedpoly!(alf, :robinson, 8)
# build_namedpoly!(alf, :rosenbrock, 4)
# build_namedpoly!(alf, :schwefel, 3)

# solve it
# @time Alfonso.solve!(alf)
