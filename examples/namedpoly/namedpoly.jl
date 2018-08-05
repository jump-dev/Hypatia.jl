#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/polyOpt.m
formulates and solves the polynomial optimization problem for a given polynomial, described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Alfonso
import MathOptInterface
const MOI = MathOptInterface
using SparseArrays
using LinearAlgebra

# list of currently available named polynomials
const polys = Dict{Symbol,NamedTuple}(
    :caprasse => (n=4, lbs=fill(-1/2, 4), ubs=fill(1/2, 4), deg=8,
        feval=((w,x,y,z) -> -w*y^3+4x*y^2*z+4w*y*z^2+2x*y^3+4w*y+4y^2-10x*z-10z^2+2)
        ),
    :goldsteinprice => (n=2, lbs=fill(-2, 2), ubs=fill(2, 2), deg=8,
        feval=((x,y) -> (1+(x+y+1)^2*(19-14x+3x^2-14y+6x*y+3y^2))*(30+(2x-3y)^2*(18-32x+12x^2+48y-36x*y+27y^2)))
        ),
    :lotkavolterra => (n=4, lbs=fill(-2, 4), ubs=fill(2, 4), deg=3,
        feval=((w,x,y,z) -> w*(x^2+y^2+z^2-1.1)+1)
        ),
    :motzkin => (n=2, lbs=fill(-1, 2), ubs=fill(1, 2), deg=6,
        feval=((x,y) -> 1-48x^2*y^2+64x^2*y^4+64x^4*y^2)
        ),
    :reactiondiffusion => (n=3, lbs=fill(-5, 3), ubs=fill(5, 3), deg=2,
        feval=((x,y,z) -> -x+2y-z-0.835634534y*(1+y))
        ),
    :robinson => (n=2, lbs=fill(-1, 2), ubs=fill(1, 2), deg=6,
        feval=((x,y) -> 1+x^6+y^6-x^4*y^2+x^4-x^2*y^4+y^4-x^2+y^2+3x^2*y^2)
        ),
    :rosenbrock => (n=2, lbs=fill(-10, 2), ubs=fill(1, 2), deg=4,
        feval=((x,y) -> 1-2x+x^2+100x^4-200x^2*y+100y^2)
        ),
    :schwefel => (n=3, lbs=fill(-10, 3), ubs=fill(10, 3), deg=4,
        feval=((x,y,z) -> (x-y^2)^2+(y-1)^2+(x-z^2)^2+(z-1)^2)
        ),
)


# case 'butcher'
#     if n ~= 6; error('butcher requires 6 arguments.'); end;
#     polyDeg = 3;
#     lb = [-1; -0.1; -0.1; -1; -0.1; -0.1];
#     ub = [0; 0.9; 0.5; -0.1; -0.05; -0.03];
#
#     vals = pts(:,6).*(pts(:,2).^2) + pts(:,5).*(pts(:,3).^2) - pts(:,1).*(pts(:,4).^2) +...
# pts(:,4).^3 + pts(:,4).^2 - (1/3)*pts(:,1) + (4/3)*pts(:,4);
# case 'magnetism7'
#     if n ~= 7; error('magnetism7 requires 7 arguments.'); end;
#     polyDeg = 2;
#     lb = repmat(-1, n, 1);
#     ub = repmat(1, n, 1);
# case 'heart'
#     if n ~= 8; error('heart requires 8 arguments.'); end;
#     polyDeg = 4;
#     lb = [-0.1; 0.4; -0.7; -0.7; 0.1; -0.1; -0.3; -1.1];
#     ub = [0.4; 1; -0.4; 0.4; 0.2; 0.2; 1.1; -0.3];




function build_namedpoly(polyname, d; native=true)
    (n, lbs, ubs, deg, feval) = polys[polyname]
    if d < ceil(Int, deg/2)
        error("requires d >= $(ceil(Int, deg/2))")
    end

    # generate interpolation
    if n == 1
        (L, U, pts, w, P0, P) = Alfonso.cheb2_data(d)
    elseif n == 2
        (L, U, pts, w, P0, P) = Alfonso.padua_data(d)
        # (L, U, pts, w, P0, P) = Alfonso.approxfekete_data(n, d)
    elseif n > 2
        (L, U, pts, w, P0, P) = Alfonso.approxfekete_data(n, d)
    end

    # transform points to fit the box domain
    pts .*= (ubs - lbs)'/2
    pts .+= (ubs + lbs)'/2
    wtVals = (pts .- lbs') .* (ubs' .- pts)
    LWts = fill(binomial(n+d-1, n), n)
    PWts = [Diagonal(sqrt.(wtVals[:,j]))*P0[:,1:LWts[j]] for j in 1:n]

    # set up MOI problem data
    A = ones(1, U)
    b = [1.0,]
    c = [feval(pts[j,:]...) for j in 1:U]
    cones = [Alfonso.SumOfSqrData(U, P0, PWts),]
    coneidxs = [1:U,]

    if native
        # use native interface
        alf = Alfonso.AlfonsoOpt(maxiter=100, verbose=true)
        Alfonso.load_data!(alf, A, b, c, cones, coneidxs)
        return alf
    else
        error("MOI tests not implemented")
        # opt = Alfonso.Optimizer()
        # return opt
    end
end

# select the named polynomial to minimize and the SOS degree (to be squared)
# alf =
#     build_namedpoly(:robinson, 8)
    # build_namedpoly(:goldsteinprice, 7)
    # build_namedpoly(:lotkavolterra, 3)

# solve it
# @time Alfonso.solve!(alf)
