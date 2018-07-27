#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Alfonso
import MathOptInterface
const MOI = MathOptInterface
using SparseArrays
using LinearAlgebra


# select the SOS degree and named polynomial to minimize
d = 10
polyname = :goldsteinprice


# TODO add more named polys from https://github.com/dpapp-github/alfonso/blob/master/polyOpt.m
# TODO use nicer way to eval the pts, maybe using MultivariatePolynomials
polys = Dict{Symbol,NamedTuple}(
    :robinson => (n=2, lbs=[-1.0,-1.0], ubs=[1.0,1.0], deg=6,
        feval=((x,y) -> 1+x^6+y^6-x^4*y^2+x^4-x^2*y^4+y^4-x^2+y^2+3x^2*y^2)
        ),
    :goldsteinprice => (n=2, lbs=[-2.0,-2.0], ubs=[2.0,2.0], deg=8,
        feval=((x,y) -> (1+(x+y+1)^2*(19-14*x+3*x^2-14*y+6*x*y+3*y^2))*(30+(2*x-3*y)^2*(18-32*x+12*x^2+48*y-36*x*y+27*y^2)))
        ),
)

(n, lbs, ubs, polyDeg, feval) = polys[polyname]
if d < ceil(Int, polyDeg/2)
    error("requires d >= $(ceil(Int, polyDeg/2))")
end

if n == 2
    (L, U, pts, w, P0, P) = padua_data(d)
else
    error("only bivariate implemented currently")
end

# TODO % transforms points to fit the domain
# scale   = (ub-lb)/2;
# shift   = (lb+ub)/2;
# pts     = bsxfun(@plus,bsxfun(@times,pts,scale'),shift');
# wtVals  = bsxfun(@minus,pts,lb').*bsxfun(@minus,ub',pts);
wtVals = 1.0 .- pts.^2 # TODO works for this example [-1,1] box
LWts = fill(binomial(n+d-1, n), n)
PWts = [Diagonal(sqrt.(wtVals[:,j]))*P0[:,1:LWts[j]] for j in 1:n]

A = ones(1, U)
b = [1.0,]
c = [feval(pts[j,1], pts[j,2]) for j in 1:U]

cones = ConeData[SumOfSqrData(U, P0, PWts),]
coneidxs = AbstractUnitRange[1:U]


# load into optimizer and solve
opt = AlfonsoOptimizer(maxiter=100)
Alfonso.loaddata!(opt, A, b, c, cones, coneidxs)
# @show opt
@time MOI.optimize!(opt)
