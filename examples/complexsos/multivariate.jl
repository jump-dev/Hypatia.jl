#=
Copyright 2018, Chris Coey and contributors
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers
# const MU = HYP.ModelUtilities

using LinearAlgebra
using Random
import Combinatorics
using Test


# primal_wsos = false
primal_wsos = true

d = 3
n = 3

L = binomial(n + d, n)
U = L^2

mon_pow(z, ex) = prod(z[i]^ex[i] for i in eachindex(ex))

L_basis = [a for t in 0:d for a in Combinatorics.multiexponents(n, t)]
V_basis = [z -> mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for l in eachindex(L_basis) for k in eachindex(L_basis)]
@test length(V_basis) == U


# # inf 1 - 4/3 * |z₁|² + 7/18 * |z₁|⁴ : 1 - |z₁|² - |z₂|² = 0
# # f - γ - σg is Hermitian-SOS, σ is an unconstrained poly
# # optimal value 1/18
# # at order 2 of hierarchy
# but with equality changed to equality, only achieves -1/3 at all orders (Josz & Molzahn)
# n = 2
# f(z) = 1 - 4/3 * abs2(z[1]) + 7/18 * abs2(abs2(z[1]))
# g1(z) = 1 - abs2(z[1]) - abs2(z[2])


# random
Fh = randn(ComplexF64, L, L)
F = (rand() > 0.5) ? Fh * Fh' : Fh
F ./= norm(F)
F = Hermitian(F)
@show isposdef(F)
f(z) = real(sum(F[k, l] * mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for l in eachindex(L_basis) for k in eachindex(L_basis)))

g1(z) = 1 - sum(abs2, z)


sample_factor = 10
num_reruns = 5
rerun_objvals = Float64[]

for rerun in 1:num_reruns
    # # roots of unity do not seem to be unisolvent
    # nrts1 = 10
    # rts1 = ComplexF64[cospi(2k / nrts1) + sinpi(2k / nrts1) * im for k = 0:(nrts1 - 1)]
    # points = Matrix{ComplexF64}(undef, nrts1^2, 2)
    # idx = 1
    # for i in 1:nrts1, j in 1:nrts1
    #     points[idx, 1] = rts1[i]
    #     points[idx, 2] = rts1[j]
    #     global idx += 1
    # end

    # sample
    num_samp = sample_factor * U
    X = randn(num_samp, 2n)
    # Y = randexp(num_samp)
    # all_points_real = [X[p, :] ./ sqrt(Y[p] + sum(abs2, X)) for p in 1:num_samp]
    Y = rand(num_samp).^inv(2n)
    all_points_real = [Y[p] * X[p, :] ./ norm(X[p, :]) for p in 1:num_samp]
    all_points = [[p[i] + im * p[i+1] for i in 1:2:2n] for p in all_points_real]
    # @show all_points[1:10]
    # @show [g1(z) for z in all_points]

    V = [b(z) for z in all_points, b in V_basis]
    @test rank(V) == U
    VF = qr(Matrix(transpose(V)), Val(true))
    keep = VF.p[1:U]
    points = all_points[keep]
    V = V[keep, :]

    @test rank(V) == U
    # @show eigvals(V)

    @show U
    @show length(points)

    # setup P0
    L0 = binomial(n + d, n)
    v0 = ones(U)
    P0 = V[:, 1:L0]
    # P0 = [prod(p[i]^a[i] for i in eachindex(a)) for p in eachrow(points), t in 0:d for a in Combinatorics.multiexponents(n, t)]
    # P0 = Matrix(qr(P0).Q)

    # setup P1
    L1 = binomial(n + d - 1, n)
    v1 = [g1(z) for z in points]
    # @show v1
    P1 = V[:, 1:L1]
    # P1 = [prod(p[i]^a[i] for i in eachindex(a)) for p in eachrow(points), t in 0:d-1 for a in Combinatorics.multiexponents(n, t)]
    # P1 = Matrix(qr(P1).Q)

    # setup problem data
    if primal_wsos
        c = [-1.0]
        A = zeros(0, 1)
        b = Float64[]
        G = ones(U, 1)
        h = [f(z) for z in points]
    else
        c = [f(z) for z in points]
        A = ones(1, U) # TODO can manually eliminate this equality
        b = [1.0]
        G = Diagonal(-1.0I, U)
        h = zeros(U)
    end
    # cones = [CO.WSOSPolyInterp_Complex(U, [P0], [v0], !primal_wsos)]
    cones = [CO.WSOSPolyInterp_Complex(U, [P0, P1], [v0, v1], !primal_wsos)]
    cone_idxs = [1:U]


    # solve
    system_solvers = [
        SO.NaiveCombinedHSDSystemSolver,
        # SO.QRCholCombinedHSDSystemSolver,
        ]
    linear_models = [
        MO.RawLinearModel,
        # MO.PreprocessedLinearModel,
        ]
    for s in system_solvers, m in linear_models
        if s == SO.QRCholCombinedHSDSystemSolver && m == MO.RawLinearModel
            continue # QRChol linear system solver needs preprocessed model
        end
        println()
        println(s)
        println(m)
        println()

        model = m(c, A, b, G, h, cones, cone_idxs)

        solver = SO.HSDSolver(model,
            verbose = true,
            max_iters = 100,
            # tol_rel_opt = 1e-9,
            # tol_abs_opt = 1e-9,
            # tol_feas = 1e-9,
            stepper = SO.CombinedHSDStepper(model, system_solver = s(model)),
            )

        SO.solve(solver)

        @show isposdef(F)
        testmin = minimum(f(z) for z in all_points)
        @show testmin
        obj = primal_wsos ? -SO.get_primal_obj(solver) : SO.get_primal_obj(solver)
        @show obj
        @test testmin > obj
        if isposdef(F)
            @test obj > -1e-5
            @test testmin > 0.0
        end

        push!(rerun_objvals, obj)
    end
end

@show rerun_objvals
