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


# select the SOS degree and named polynomial to minimize
# let d > deg/2 where deg is the degree of the named polynomial
d = 3
polyname = :lotkavolterra


# list of currently available named polynomials
polys = Dict{Symbol,NamedTuple}(
    :robinson => (n=2, lbs=fill(-1.0, 2), ubs=fill(1.0, 2), deg=6,
        feval=((x,y) -> 1+x^6+y^6-x^4*y^2+x^4-x^2*y^4+y^4-x^2+y^2+3x^2*y^2)
        ),
    :goldsteinprice => (n=2, lbs=fill(-2.0, 2), ubs=fill(2.0, 2), deg=8,
        feval=((x,y) -> (1+(x+y+1)^2*(19-14*x+3*x^2-14*y+6*x*y+3*y^2))*(30+(2*x-3*y)^2*(18-32*x+12*x^2+48*y-36*x*y+27*y^2)))
        ),
    :lotkavolterra => (n=4, lbs=fill(-2.0, 4), ubs=fill(2.0, 4), deg=3,
        feval=((w,x,y,z) -> w*(x^2+y^2+z^2-1.1)+1)
        ),
)

(n, lbs, ubs, deg, feval) = polys[polyname]
if d < ceil(Int, deg/2)
    error("requires d >= $(ceil(Int, deg/2))")
end

# generate interpolation
if n == 1
    (L, U, pts, w, P0, P) = cheb2_data(d)
elseif n == 2
    (L, U, pts, w, P0, P) = padua_data(d)
    # (L, U, pts, w, P0, P) = approxfekete_data(n, d)
elseif n > 2
    (L, U, pts, w, P0, P) = approxfekete_data(n, d)
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
cones = ConeData[SumOfSqrData(U, P0, PWts),]
coneidxs = AbstractUnitRange[1:U]


# load into optimizer and solve
opt = AlfonsoOptimizer(maxiter=100, verbose=true)
Alfonso.loaddata!(opt, A, b, c, cones, coneidxs)
# @show opt
@time MOI.optimize!(opt)
