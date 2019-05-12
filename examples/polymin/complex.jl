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

function build_complexpolymin(
    n::Int,
    d::Int,
    f::Function,
    gs::Vector,
    gdegs::Vector,
    primal_wsos::Bool;
    sample_factor::Int = 100,
    )
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
    else
        c = f.(points)
        A = ones(1, U) # TODO can eliminate equality and a variable
        b = [1.0]
        G = Diagonal(-1.0I, U)
        h = zeros(U)
    end
    cones = [CO.WSOSPolyInterp_2(U, P_data, g_data, !primal_wsos)]
    cone_idxs = [1:U]

    return (c, A, b, G, h, cones, cone_idxs)
end

function complexpolymin(polyname::Symbol, d::Int; primal_wsos = true)
    (n, deg, f, gs, gdegs, truemin) = complexpolys[polyname]
    return build_complexpolymin(n, d, f, gs, gdegs, primal_wsos)
end

complexpolymin1() = complexpolymin(:abs1d, 1)
complexpolymin2() = complexpolymin(:absunit1d, 1)
complexpolymin3() = complexpolymin(:negabsunit1d, 2)
complexpolymin4() = complexpolymin(:absball2d, 1)
complexpolymin5() = complexpolymin(:absbox2d, 2)
complexpolymin6() = complexpolymin(:negabsbox2d, 1)
complexpolymin7() = complexpolymin(:denseunit1d, 2)
complexpolymin8() = complexpolymin(:abs1d, 1, primal_wsos = false)
complexpolymin9() = complexpolymin(:absunit1d, 1, primal_wsos = false)
complexpolymin10() = complexpolymin(:negabsunit1d, 2, primal_wsos = false)
complexpolymin11() = complexpolymin(:absball2d, 1, primal_wsos = false)
complexpolymin12() = complexpolymin(:absbox2d, 2, primal_wsos = false)
complexpolymin13() = complexpolymin(:negabsbox2d, 1, primal_wsos = false)
complexpolymin14() = complexpolymin(:denseunit1d, 2, primal_wsos = false)



function run_complexpolymin(polyname::Symbol, d::Int; primal_wsos::Bool = false)
    # get data for polynomial and domain
    (n, deg, f, gs, gdegs, truemin) = complexpolys[polyname]
    @assert d >= deg
    @assert all(d .>= gdegs)

    (c, A, b, G, h, cones, cone_idxs) = build_complexpolymin(n, d, f, gs, gdegs, primal_wsos)

    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = true, tol_feas = 1e-6, tol_rel_opt = 1e-7, tol_abs_opt = 1e-7)
    SO.solve(solver)

    @test SO.get_status(solver) == :Optimal
    @test SO.get_primal_obj(solver) ≈ (primal_wsos ? -truemin : truemin) atol = 1e-4 rtol = 1e-4

    return
end

run_complexpolymin_primal() = run_complexpolymin(:denseunit1d, 2, primal_wsos = true)
run_complexpolymin_dual() = run_complexpolymin(:denseunit1d, 2, primal_wsos = false)
