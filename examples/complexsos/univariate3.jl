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
import Random
using Test

# Random.seed!(1)

primal_wsos = false
# primal_wsos = true

d = 3

# inf -1 + |z|² + ... + |z|ᵈ
# optimal value -1, for z = 0
# f(z) = -1 + 4*real(z)# + abs(z)^2 + real(z^2) + abs(z)^3 # + sum(abs(z)^i for i in 2:2:d))
# f(z) = 1 + real(z) + abs(z)^2 + real(z^2) + real(z^2 * conj(z)) + abs(z)^4

# f(z) = 1 + 2real(z) + abs(z)^2 + 2real(z^2) + 2real(z^2 * conj(z)) + abs(z)^4
# f(z) = 1 + real(z)# + 3abs(z)^2
# f(z) = -1 + 2.5abs(z)^2 + 0.5abs(z)^4
# mathematica:
# Minimize[1+2x+x^2+y^2+2(x^2-y^2)+2(x^3+x*y^2)+(x^2+y^2)^2,{x,y}]
# min is 0

# random
Fh = randn(ComplexF64, d + 1, d + 1)
if rand() > 0.5
    F = Hermitian(Fh)
else
    F = Hermitian(Fh * Fh')
end
# @show isposdef(F)

f(z) = real(sum(F[i+1, j+1] * z^i * conj(z)^j for i in 0:d, j in 0:d))


sample_factor = 10
num_reruns = 5
rerun_objvals = Float64[]

for rerun in 1:num_reruns
    U = (d + 1)^2
    V_basis = [z -> z^i * conj(z)^j for j in 0:d for i in 0:d] # TODO columns are dependent if not doing j in 0:i
    # U = div((d+1)*(d+2), 2)
    # V_basis = [z -> z^i * conj(z)^j for j in 0:d for i in 0:j]

    # # roots of unity do not seem to be unisolvent
    # points = [cospi(2k / U) + sinpi(2k / U) * im for k = 0:(U - 1)]
    # # @show points
    # V = [b(p) for p in points, b in V_basis]

    # sample
    # all_points = rand(ComplexF64, sample_factor * U) .- 0.5 * (1 + im)
    # for i in eachindex(all_points)
    #     if abs(all_points[i]) >= 1
    #         all_points[i] /= abs(all_points[i])
    #     end
    # end
    radii = sqrt.(rand(sample_factor * U))
    angles = rand(sample_factor * U) * 2pi
    all_points = radii .* (cos.(angles) .+ (sin.(angles) .* im))
    # @show all_points[1:10]

    V = [b(p) for p in all_points, b in V_basis]
    # @test rank(V) == U
    VF = qr(Matrix(transpose(V)), Val(true))
    keep = VF.p[1:U]
    points = all_points[keep]
    V = V[keep, :]
    # @test rank(V) == U
    # @show eigvals(V)


    # setup P0
    # L0 = d + 1
    v0 = ones(U)
    # P0 = V[:, 1:L0]
    P0 = [p^i for p in points, i in 0:d]
    # P0 = Matrix(qr(P0).Q)

    # # setup P1
    # L1 = d
    g1(z) = 1 - abs2(z)
    v1 = [g1(p) for p in points]
    # # @show v1
    P1 = [p^i for p in points, i in 0:d-1]
    # # P1 = Matrix(qr(P1).Q)

    # setup problem data
    if primal_wsos
        c = [-1.0]
        A = zeros(0, 1)
        b = Float64[]
        G = ones(U, 1)
        h = [f(points[j]) for j in 1:U]
    else
        c = [f(points[j]) for j in 1:U]
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
        # SO.NaiveCombinedHSDSystemSolver,
        SO.QRCholCombinedHSDSystemSolver,
        ]
    linear_models = [
        # MO.RawLinearModel,
        MO.PreprocessedLinearModel,
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
            tol_rel_opt = 1e-9,
            tol_abs_opt = 1e-9,
            tol_feas = 1e-9,
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
            @test obj > 0.0
            @test testmin > 0.0
        end

        push!(rerun_objvals, obj)
    end
end

@show rerun_objvals
