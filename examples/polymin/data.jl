#=
list of predefined polynomials and domains from various applications
see https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
=#

import DynamicPolynomials
const DP = DynamicPolynomials
import Hypatia

# get interpolation for a predefined real poly
function get_interp_data(
    ::Type{T},
    poly_name::Symbol,
    halfdeg::Int,
    ) where {T <: Real}
    (x, fn, dom, true_min) = real_poly_data(poly_name, T)
    (U, pts, Ps) = ModelUtilities.interpolate(dom, halfdeg)
    interp_vals = T[fn(pts[j, :]...) for j in 1:U]
    return (interp_vals, Ps, true_min)
end

# get interpolation for a predefined complex poly
function get_interp_data(
    R::Type{Complex{T}},
    poly_name::Symbol,
    halfdeg::Int,
    ) where {T <: Real}
    (n, f, gs, g_halfdegs, true_min) = complex_poly_data[poly_name]
    (points, Ps) = ModelUtilities.interpolate(R, halfdeg, n, gs, g_halfdegs)
    interp_vals = f.(points)
    return (interp_vals, Ps, true_min)
end

# get interpolation for a random real poly in n variables of half degree halfdeg and use a box domain
function random_interp_data(
    ::Type{T},
    n::Int,
    halfdeg::Int,
    dom = ModelUtilities.Box{T}(-ones(T, n), ones(T, n)),
    ) where {T <: Real}
    (U, pts, Ps) = ModelUtilities.interpolate(dom, halfdeg)
    interp_vals = randn(T, U)
    true_min = T(NaN) # TODO could get an upper bound by evaluating at random points in domain
    return (interp_vals, Ps, true_min)
end

# real polynomials
function real_poly_data(polyname::Symbol, T::Type{<:Real} = Float64)
    if polyname == :butcher
        DP.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        dom = ModelUtilities.Box{T}(T[-1,-0.1,-0.1,-1,-0.1,-0.1], T[0,0.9,0.5,-0.1,-0.05,-0.03])
        true_obj = -1.4393333333
    elseif polyname == :caprasse
        DP.@polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = ModelUtilities.Box{T}(-T(0.5) * ones(T, 4), T(0.5) * ones(T, 4))
        true_obj = -3.1800966258
    elseif polyname == :goldsteinprice
        DP.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = ModelUtilities.Box{T}(-2 * ones(T, 2), 2 * ones(T, 2))
        true_obj = 3
    # elseif polyname == :goldsteinprice_ball
    #     DP.@polyvar x[1:2]
    #     f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
    #     dom = ModelUtilities.Ball{T}(zeros(T, 2), 2*sqrt(T(2)))
    #     true_obj = 3 # small neighborhood around box
    # elseif polyname == :goldsteinprice_ellipsoid
    #     DP.@polyvar x[1:2]
    #     f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
    #     centers = zeros(T, 2)
    #     Q = Diagonal(T(0.25) * ones(T, 2))
    #     dom = ModelUtilities.Ellipsoid{T}(centers, Q)
    #     true_obj = 3 # small neighborhood around box
    elseif polyname == :heart
        DP.@polyvar x[1:8]
        f = x[1]*x[6]^3-3x[1]*x[6]*x[7]^2+x[3]*x[7]^3-3x[3]*x[7]*x[6]^2+x[2]*x[5]^3-3*x[2]*x[5]*x[8]^2+x[4]*x[8]^3-3x[4]*x[8]*x[5]^2+0.9563453
        dom = ModelUtilities.Box{T}(T[-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], T[0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3])
        true_obj = -1.36775
    elseif polyname == :lotkavolterra
        DP.@polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = ModelUtilities.Box{T}(-2 * ones(T, 4), 2 * ones(T, 4))
        true_obj = -20.8
    elseif polyname == :magnetism7
        DP.@polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = ModelUtilities.Box{T}(-ones(T, 7), ones(T, 7))
        true_obj = -0.25
    # elseif polyname == :magnetism7_ball
    #     DP.@polyvar x[1:7]
    #     f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
    #     dom = ModelUtilities.Ball{T}(zeros(T, 7), sqrt(T(7)))
    #     true_obj = -0.25 # small neighborhood around box
    elseif polyname == :motzkin
        DP.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = ModelUtilities.Box{T}(-ones(T, 2), ones(T, 2))
        true_obj = 0
    # elseif polyname == :motzkin_ball
    #     DP.@polyvar x[1:2]
    #     f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
    #     dom = ModelUtilities.Ball{T}(zeros(T, 2), sqrt(T(2)))
    #     true_obj = 0 # small neighborhood around box
    # elseif polyname == :motzkin_ellipsoid
    #     # ellipsoid contains two local minima in opposite orthants
    #     DP.@polyvar x[1:2]
    #     f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
    #     Q = T[1 1; 1 -1]
    #     D = T[1 0; 0 0.1]
    #     S = Q * D * Q
    #     dom = ModelUtilities.Ellipsoid{T}(zeros(T, 2), S)
    #     true_obj = 0 # small neighborhood around box
    elseif polyname == :reactiondiffusion
        DP.@polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = ModelUtilities.Box{T}(-5 * ones(T, 3), 5 * ones(T, 3))
        true_obj = -36.71269068
    elseif polyname == :robinson
        DP.@polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = ModelUtilities.Box{T}(-ones(T, 2), ones(T, 2))
        true_obj = 0.814814
    # elseif polyname == :robinson_ball
    #     DP.@polyvar x[1:2]
    #     f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
    #     dom = ModelUtilities.Ball{T}(zeros(T, 2), sqrt(T(2)))
    #     true_obj = 0.814814 # small neighborhood around box
    elseif polyname == :rosenbrock
        DP.@polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = ModelUtilities.Box{T}(-5 * ones(T, 2), 10 * ones(T, 2))
        true_obj = 0
    # elseif polyname == :rosenbrock_ball
    #     DP.@polyvar x[1:2]
    #     f = (1-x[1])^2+100*(x[1]^2-x[2])^2
    #     dom = ModelUtilities.Ball{T}(T(2.5) * ones(T, 2), T(7.5) * sqrt(T(2)))
    #     true_obj = 0 # small neighborhood around box
    elseif polyname == :schwefel
        DP.@polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = ModelUtilities.Box{T}(-10 * ones(T, 3), 10 * ones(T, 3))
        true_obj = 0
    # elseif polyname == :schwefel_ball
    #     DP.@polyvar x[1:3]
    #     f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
    #     dom = ModelUtilities.Ball{T}(zeros(T, 3), 10 * sqrt(T(3)))
    #     true_obj = 0 # small neighborhood around box
    else
        error("poly $polyname not recognized")
    end

    return (x, f, dom, true_obj)
end

# merge with real polys when complex polyvars are allowed in DynamicPolynomials: https://github.com/JuliaAlgebra/MultivariatePolynomials.jl/issues/11
# real-valued complex polynomials
complex_poly_data = Dict{Symbol, NamedTuple}(
    :abs1d => (n = 1,
        f = (z -> 1 + sum(abs2, z)),
        gs = Function[],
        g_halfdegs = Int[],
        true_min = 1,
        ),
    :absunit1d => (n = 1,
        f = (z -> 1 + sum(abs2, z)),
        gs = [z -> 1 - sum(abs2, z)],
        g_halfdegs = [1],
        true_min = 1,
        ),
    :negabsunit1d => (n = 1,
        f = (z -> -sum(abs2, z)),
        gs = [z -> 1 - sum(abs2, z)],
        g_halfdegs = [1],
        true_min = -1,
        ),
    :absball2d => (n = 2,
        f = (z -> 1 + sum(abs2, z)),
        gs = [z -> 1 - sum(abs2, z)],
        g_halfdegs = [1],
        true_min = 1,
        ),
    :absbox2d => (n = 2,
        f = (z -> 1 + sum(abs2, z)),
        gs = [z -> 1 - abs2(z[1]), z -> 1 - abs2(z[2])],
        g_halfdegs = [1, 1],
        true_min = 1,
        ),
    :negabsbox2d => (n = 2,
        f = (z -> -sum(abs2, z)),
        gs = [z -> 1 - abs2(z[1]), z -> 1 - abs2(z[2])],
        g_halfdegs = [1, 1],
        true_min = -2,
        ),
    :denseunit1d => (n = 1,
        f = (z -> 1 + 2real(z[1]) + abs(z[1])^2 + 2real(z[1]^2) + 2real(z[1]^2 * conj(z[1])) + abs(z[1])^4),
        gs = [z -> 1 - abs2(z[1])],
        g_halfdegs = [1],
        true_min = 0,
        ),
    )
