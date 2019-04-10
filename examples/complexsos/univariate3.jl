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


primal_wsos = false
# primal_wsos = true

d = 2

# inf -1 + |z|² + ... + |z|ᵈ
# optimal value -1, for z = 0
# f(z) = -1 + 4*real(z) + abs(z)^2 + real(z^2) + abs(z)^3 # + sum(abs(z)^i for i in 2:2:d))
# f(z) = 1 + real(z) + abs(z)^2 + real(z^2) + real(z^2 * conj(z)) + 8abs(z)^4

f(z) = 1 + 0.2real(z) + abs(z)^2 + 0.2real(z^2) + 0.2real(z^2 * conj(z)) + abs(z)^4
# mathematica:
# Minimize[1+2x+x^2+y^2+2(x^2-y^2)+2(x^3+x*y^2)+(x^2+y^2)^2,{x,y}]
# min is 0


# # inf random f
# Fh = randn(ComplexF64, d + 1, d + 1)
# if rand() > 0.5
#     F = Hermitian(Fh)
# else
#     F = Hermitian(Fh * Fh')
# end
# @show isposdef(F)
#
# f(z) = real(sum(F[i+1, j+1] * z^i * conj(z)^j for i in 0:d, j in 0:d))


# sample
# U = (d + 1)^2
U = div((d+1)*(d+2), 2)
V_basis = [z -> z^i * conj(z)^j for j in 0:d for i in 0:j] # TODO columns are dependent if not doing j in 0:i

# # roots of unity do not seem to be unisolvent
# points = [cospi(2k / U) + sinpi(2k / U) * im for k = 0:(U - 1)]
# # @show points
# V = [b(p) for p in points, b in V_basis]

# sample
sample_factor = 1000
points = 2 * rand(ComplexF64, sample_factor * U) .- (1 + im)
V = [b(p) for p in points, b in V_basis]
@test rank(V) == U
VF = qr(Matrix(transpose(V)), Val(true))
keep = VF.p[1:U]
points = points[keep]
V = V[keep, :]

@test rank(V) == U
# @show eigvals(V)

# setup P0
L0 = d + 1
v0 = ones(U)
P0 = V[:, 1:L0]
# P0 = [p^i for p in points, i in 0:d]

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
cones = [CO.WSOSPolyInterp_Complex(U, [P0], [v0], !primal_wsos)]
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
        # tol_rel_opt = 1e-5,
        # tol_abs_opt = 1e-5,
        # tol_feas = 1e-5,
        stepper = SO.CombinedHSDStepper(model, system_solver = s(model)),
        )

    SO.solve(solver)

    # @test SO.get_status(solver) == :Optimal
    # @show SO.get_primal_obj(solver)
    # @show isposdef(F)
    # @show F

    # testmin = minimum(f(z) for z in randn(ComplexF64, 1000))
    # @show testmin
    obj = primal_wsos ? -SO.get_primal_obj(solver) : SO.get_primal_obj(solver)
    @show obj

    # if isposdef(F)
        @test SO.get_status(solver) == :Optimal
        # @test obj > 0.0
        # @test testmin > 0.0
        # @test testmin > obj
        # @test obj ≈ -1 atol=1e-3
    # else
    #     stat = primal_wsos ? :PrimalInfeasible : :DualInfeasible
    #     @test SO.get_status(solver) == stat
    # end
end
