#=
Copyright 2018, Chris Coey and contributors

minimizes a real-valued complex polynomial over a domain defined by real-valued complex polynomials
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers

using LinearAlgebra
import Combinatorics
using Test

include("data.jl")

mon_pow(z, ex) = prod(z[i]^ex[i] for i in eachindex(ex))

function polymincomplex(polyname::Symbol, d::Int; primal_wsos = true, sample_factor::Int = 100)
    (n, deg, f, gs, gdegs, true_obj) = complexpolys[polyname]
    # generate interpolation
    L = binomial(n + d, n)
    U = L^2
    num_samp = sample_factor * U

    L_basis = [a for t in 0:d for a in Combinatorics.multiexponents(n, t)]
    V_basis = [z -> mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for l in eachindex(L_basis) for k in eachindex(L_basis)]
    @assert length(V_basis) == U

    # sample from cartesian product of n complex unit balls
    # TODO expo vs unif give different distributions it seems
    # TODO look up theory of ideal interpolation points analogous to Fekete points in real case
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
        push!(P_data, V[:, 1:binomial(n + d - gdegs[i], n)])
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
    cones = [CO.WSOSPolyInterp_2(U, P_data, g_data, !primal_wsos)]
    cone_idxs = [1:U]

    return (model = (c, A, b, G, h, cones, cone_idxs), true_obj = true_obj)
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

function test_polymincomplex(builder::Function)
    ((c, A, b, G, h, cones, cone_idxs), true_obj) = builder()
    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = true)
    SO.solve(solver)
    r = SO.test_certificates(solver, model, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    @test r.primal_obj ≈ true_obj atol = 1e-4 rtol = 1e-4

    return
end

test_polymincomplexs() = test_polymincomplex.([
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
    ])
