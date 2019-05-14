#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

list of predefined polynomials and domains from various applications
see https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
=#

import DynamicPolynomials
const DP = DynamicPolynomials
import Hypatia
const MU = Hypatia.ModelUtilities

# real polynomials
function getpolydata(polyname::Symbol)
    if polyname == :butcher
        DP.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        dom = MU.Box([-1,-0.1,-0.1,-1,-0.1,-0.1], [0,0.9,0.5,-0.1,-0.05,-0.03])
        true_obj = -1.4393333333
    elseif polyname == :butcher_ball
        DP.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        axes = 0.5 * ([0,0.9,0.5,-0.1,-0.05,-0.03] - [-1,-0.1,-0.1,-1,-0.1,-0.1])
        centers = 0.5 * ([-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])
        dom = MU.Ball(centers, sqrt(6) * maximum(axes))
        true_obj = -4.10380
    elseif polyname == :butcher_ellipsoid
        DP.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        # heuristically-obtained enclosing ellipsoid
        centers = 0.5 * ([-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])
        Q = Diagonal(6 * abs2.(centers))
        dom = MU.Ellipsoid(centers, Q)
        true_obj = -16.7378208
    elseif polyname == :caprasse
        DP.@polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = MU.Box(-0.5 * ones(4), 0.5 * ones(4))
        true_obj = -3.1800966258
    elseif polyname == :caprasse_ball
        DP.@polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = MU.Ball(zeros(4), 1.0)
        true_obj = -9.47843346
    elseif polyname == :goldsteinprice
        DP.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = MU.Box(-2 * ones(2), 2 * ones(2))
        true_obj = 3
    elseif polyname == :goldsteinprice_ball
        DP.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = MU.Ball(zeros(2), 2*sqrt(2))
        true_obj = 3
    elseif polyname == :goldsteinprice_ellipsoid
        DP.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        centers = zeros(2)
        Q = Diagonal(0.25 * ones(2))
        dom = MU.Ellipsoid(centers, Q)
        true_obj = 3
    elseif polyname == :heart
        DP.@polyvar x[1:8]
        f = x[1]*x[6]^3-3x[1]*x[6]*x[7]^2+x[3]*x[7]^3-3x[3]*x[7]*x[6]^2+x[2]*x[5]^3-3*x[2]*x[5]*x[8]^2+x[4]*x[8]^3-3x[4]*x[8]*x[5]^2+0.9563453
        dom = MU.Box([-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], [0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3])
        true_obj = -1.36775
    elseif polyname == :lotkavolterra
        DP.@polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = MU.Box(-2 * ones(4), 2 * ones(4))
        true_obj = -20.8
    elseif polyname == :lotkavolterra_ball
        DP.@polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = MU.Ball(zeros(4), 4.0)
        true_obj = -21.13744
    elseif polyname == :magnetism7
        DP.@polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = MU.Box(-ones(7), ones(7))
        true_obj = -0.25
    elseif polyname == :magnetism7_ball
        DP.@polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = MU.Ball(zeros(7), sqrt(7))
        true_obj = -0.25
    elseif polyname == :motzkin
        DP.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = MU.Box(-ones(2), ones(2))
        true_obj = 0
    elseif polyname == :motzkin_ball
        DP.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = MU.Ball(zeros(2), sqrt(2))
        true_obj = 0
    elseif polyname == :motzkin_ellipsoid
        # ellipsoid contains two local minima in opposite orthants
        DP.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        Q = [1 1; 1 -1]
        D = [1 0; 0 0.1]
        S = Q * D * Q
        dom = MU.Ellipsoid(zeros(2), S)
        true_obj = 0
    elseif polyname == :reactiondiffusion
        DP.@polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = MU.Box(-5 * ones(3), 5 * ones(3))
        true_obj = -36.71269068
    elseif polyname == :reactiondiffusion_ball
        DP.@polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = MU.Ball(zeros(3), 5*sqrt(3))
        true_obj = -73.31
    elseif polyname == :robinson
        DP.@polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = MU.Box(-ones(2), ones(2))
        true_obj = 0.814814
    elseif polyname == :robinson_ball
        DP.@polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = MU.Ball(zeros(2), sqrt(2))
        true_obj = 0.814814
    elseif polyname == :rosenbrock
        DP.@polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = MU.Box(-5 * ones(2), 10 * ones(2))
        true_obj = 0
    elseif polyname == :rosenbrock_ball
        DP.@polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = MU.Ball(2.5 * ones(2), 7.5*sqrt(2))
        true_obj = 0
    elseif polyname == :schwefel
        DP.@polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = MU.Box(-10 * ones(3), 10 * ones(3))
        true_obj = 0
    elseif polyname == :schwefel_ball
        DP.@polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = MU.Ball(zeros(3), 10*sqrt(3))
        true_obj = 0
    else
        error("poly $polyname not recognized")
    end

    return (x, f, dom, true_obj)
end

# merge with real polys dictionary when complex polyvars are allowed in DynamicPolynomials: https://github.com/JuliaAlgebra/MultivariatePolynomials.jl/issues/11
# real-valued complex polynomials
complexpolys = Dict{Symbol, NamedTuple}(
    :abs1d => (n = 1, deg = 1,
        f = (z -> 1 + sum(abs2, z)),
        gs = [],
        gdegs = [],
        truemin = 1,
        ),
    :absunit1d => (n = 1, deg = 1,
        f = (z -> 1 + sum(abs2, z)),
        gs = [z -> 1 - sum(abs2, z)],
        gdegs = [1],
        truemin = 1,
        ),
    :negabsunit1d => (n = 1, deg = 1,
        f = (z -> -sum(abs2, z)),
        gs = [z -> 1 - sum(abs2, z)],
        gdegs = [1],
        truemin = -1,
        ),
    :absball2d => (n = 2, deg = 1,
        f = (z -> 1 + sum(abs2, z)),
        gs = [z -> 1 - sum(abs2, z)],
        gdegs = [1],
        truemin = 1,
        ),
    :absbox2d => (n = 2, deg = 1,
        f = (z -> 1 + sum(abs2, z)),
        gs = [z -> 1 - abs2(z[1]), z -> 1 - abs2(z[2])],
        gdegs = [1, 1],
        truemin = 1,
        ),
    :negabsbox2d => (n = 2, deg = 1,
        f = (z -> -sum(abs2, z)),
        gs = [z -> 1 - abs2(z[1]), z -> 1 - abs2(z[2])],
        gdegs = [1, 1],
        truemin = -2,
        ),
    :denseunit1d => (n = 1, deg = 2,
        f = (z -> 1 + 2real(z[1]) + abs(z[1])^2 + 2real(z[1]^2) + 2real(z[1]^2 * conj(z[1])) + abs(z[1])^4),
        gs = [z -> 1 - abs2(z[1])],
        gdegs = [1],
        truemin = 0,
    )
)
