#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

polyminreal: formulates and solves the real polynomial optimization problem for a given polynomial; see:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming.

polymincomplex: minimizes a real-valued complex polynomial over a domain defined by real-valued complex polynomials
=#

import Random
using LinearAlgebra
import Combinatorics
using Test
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers

include(joinpath(@__DIR__, "data.jl"))

function polyminreal(
    polyname::Symbol,
    halfdeg::Int;
    primal_wsos::Bool = true,
    )
    (x, fn, dom, true_obj) = getpolydata(polyname)
    sample = (length(x) >= 5) || !isa(dom, MU.Box)
    (U, pts, P0, PWts, _) = MU.interpolate(dom, halfdeg, sample = sample)

    # set up problem data
    interp_vals = [fn(pts[j, :]...) for j in 1:U]
    if primal_wsos
        c = [-1.0]
        A = zeros(0, 1)
        b = Float64[]
        G = ones(U, 1)
        h = interp_vals
        true_obj *= -1
    else
        c = interp_vals
        A = ones(1, U) # TODO eliminate constraint and first variable
        b = [1.0]
        G = Diagonal(-1.0I, U) # TODO use UniformScaling
        h = zeros(U)
    end
    cones = [CO.WSOSPolyInterp(U, [P0, PWts...], !primal_wsos)]
    cone_idxs = [1:U]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs, true_obj = true_obj)
end


polyminreal1() = polyminreal(:heart, 2)
polyminreal2() = polyminreal(:schwefel, 2)
polyminreal3() = polyminreal(:magnetism7_ball, 2)
polyminreal4() = polyminreal(:motzkin_ellipsoid, 4)
polyminreal5() = polyminreal(:caprasse, 4)
polyminreal6() = polyminreal(:goldsteinprice, 7)
polyminreal7() = polyminreal(:lotkavolterra, 3)
polyminreal8() = polyminreal(:robinson, 8)
polyminJuMP9() = polyminJuMP(:robinson_ball, 8)
polyminreal10() = polyminreal(:rosenbrock, 5)
polyminreal11() = polyminreal(:butcher, 2)
polyminJuMP12() = polyminJuMP(:goldsteinprice_ellipsoid, 7)
polyminJuMP13() = polyminJuMP(:goldsteinprice_ball, 7)
polyminreal14() = polyminreal(:motzkin, 3, primal_wsos = false)
polyminreal15() = polyminreal(:motzkin, 3)
polyminreal16() = polyminreal(:reactiondiffusion, 4, primal_wsos = false)
polyminreal17() = polyminreal(:lotkavolterra, 3, primal_wsos = false)

function polymincomplex(
    polyname::Symbol,
    halfdeg::Int;
    primal_wsos = true,
    sample_factor::Int = 100,
    )
    (n, deg, f, gs, g_halfdegs, true_obj) = complexpolys[polyname]

    # generate interpolation
    L = binomial(n + halfdeg, n)
    U = L^2
    L_basis = [a for t in 0:halfdeg for a in Combinatorics.multiexponents(n, t)]
    mon_pow(z, ex) = prod(z[i]^ex[i] for i in eachindex(ex))
    V_basis = [z -> mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for l in eachindex(L_basis) for k in eachindex(L_basis)]
    @assert length(V_basis) == U

    # sample from cartesian product of n complex unit balls
    # TODO expo vs unif give different distributions it seems
    # TODO look up theory of ideal interpolation points analogous to Fekete points in real case
    num_samp = sample_factor * U
    X = randn(num_samp, 2n)
    # Y = randexp(num_samp)
    # all_points_real = [X[p, :] ./ sqrt(Y[p] + sum(abs2, X)) for p in 1:num_samp]
    Y = rand(num_samp).^inv(2n)
    all_points_real = [Y[p] * X[p, :] ./ norm(X[p, :]) for p in 1:num_samp]
    all_points = [[p[i] + im * p[i+1] for i in 1:2:2n] for p in all_points_real]

    # select subset of points to maximize |det(V)| in heuristic QR-based procedure (analogous to real case)
    V = [b(z) for z in all_points, b in V_basis]
    @test rank(V) == U
    VF = qr(Matrix(transpose(V)), Val(true))
    keep = VF.p[1:U]
    points = all_points[keep]
    V = V[keep, :]
    @test rank(V) == U

    # setup P matrices
    g_data = [ones(U)]
    P_data = [V[:, 1:L]]
    for i in eachindex(gs)
        push!(g_data, gs[i].(points))
        push!(P_data, V[:, 1:binomial(n + halfdeg - g_halfdegs[i], n)])
    end

    # setup problem data
    if primal_wsos
        c = [-1.0]
        A = zeros(0, 1)
        b = Float64[]
        G = ones(U, 1)
        h = f.(points)
        true_obj *= -1
    else
        c = f.(points)
        A = ones(1, U) # TODO can eliminate equality and a variable
        b = [1.0]
        G = Diagonal(-1.0I, U)
        h = zeros(U)
    end
    cones = [CO.WSOSPolyInterp_2(U, P_data, g_data, !primal_wsos)] # TODO use old WSOS cone definition
    cone_idxs = [1:U]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs, true_obj = true_obj)
end

polymincomplex1() = polymincomplex(:abs1d, 1)
polymincomplex2() = polymincomplex(:absunit1d, 1)
polymincomplex3() = polymincomplex(:negabsunit1d, 2)
polymincomplex4() = polymincomplex(:absball2d, 1)
polymincomplex5() = polymincomplex(:absbox2d, 2)
polymincomplex6() = polymincomplex(:negabsbox2d, 1)
polymincomplex7() = polymincomplex(:denseunit1d, 2)
polymincomplex8() = polymincomplex(:abs1d, 1, primal_wsos = false)
polymincomplex9() = polymincomplex(:absunit1d, 1, primal_wsos = false)
polymincomplex10() = polymincomplex(:negabsunit1d, 2, primal_wsos = false)
polymincomplex11() = polymincomplex(:absball2d, 1, primal_wsos = false)
polymincomplex12() = polymincomplex(:absbox2d, 2, primal_wsos = false)
polymincomplex13() = polymincomplex(:negabsbox2d, 1, primal_wsos = false)
polymincomplex14() = polymincomplex(:denseunit1d, 2, primal_wsos = false)

function test_polymin(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    model = MO.PreprocessedLinearModel(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    @test r.primal_obj ≈ d.true_obj atol = 1e-4 rtol = 1e-4
    return
end

test_polymin(; options...) = test_polymin.([
    polyminreal1,
    polyminreal2,
    polyminreal3,
    polyminreal4,
    polyminreal5,
    polyminreal6,
    polyminreal7,
    polyminreal8,
    polyminreal9,
    polyminreal10,
    polyminreal11,
    polyminreal12,
    polyminreal13,
    polyminreal14,
    polyminreal15,
    polyminreal16,
    polyminreal17,
    polymincomplex1,
    polymincomplex2,
    polymincomplex3,
    polymincomplex4,
    polymincomplex5,
    polymincomplex6,
    polymincomplex7,
    polymincomplex8,
    polymincomplex9,
    polymincomplex10,
    polymincomplex11,
    polymincomplex12,
    polymincomplex13,
    polymincomplex14,
    ], options = options)

test_polymin_quick(; options...) = test_polyminJuMP.([
    polyminJuMP2,
    polyminJuMP3,
    polyminJuMP6,
    polyminJuMP14,
    polyminJuMP15,
    polymincomplex1,
    polymincomplex3,
    polymincomplex8,
    polymincomplex10,
    ], options = options)
