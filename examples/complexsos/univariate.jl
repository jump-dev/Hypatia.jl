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


# primal_wsos = false
primal_wsos = true

# inf z + ̄z : |z|² ≤ 1
# ≡ inf f(z) = 2Re(z) : g₁(z) = 1 - |z|² ≥ 0
# optimal value -2, for z = -1 + 0im
# z + ̄z - (-2) = |1 + z|² + |1|² × (1 - |z|²)

f(z) = 2 * real(z)
g1(z) = 1 - abs2(z)
deg = 2

# setup interpolation
U = deg + 1
basis = [x -> x^j for j in 0:deg]

# # points are the roots of unity
# points = [cospi(2k / U) + sinpi(2k / U) * im for k = 0:(U - 1)]
# @show points
# V = [b(p) for p in points, b in basis]
# @test rank(V) == U

# sample
sample_factor = 100
points = rand(ComplexF64, sample_factor * U) .- 0.5 * (1 + im)
@assert all(abs2.(points) .<= 1)
V = [b(p) for p in points, b in basis]
@test rank(V) == U
F = qr(Matrix(V'), Val(true))
keep = F.p[1:U]
points = points[keep]
V = V[keep, :]

# setup P0
L0 = 2
v0 = ones(U)
P0 = V[:, 1:L0]
# setup P1
L1 = 1
v1 = [g1(p) for p in points]
P1 = V[:, 1:L1]

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
cones = [CO.WSOSPolyInterp_Complex(U, [P0, P1], [v0, v1], !primal_wsos)]
cone_idxs = [1:U]

# solve
system_solvers = [
    SO.NaiveCombinedHSDSystemSolver,
    SO.QRCholCombinedHSDSystemSolver,
    ]
linear_models = [
    MO.RawLinearModel,
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
        tol_rel_opt = 1e-10,
        tol_abs_opt = 1e-10,
        tol_feas = 1e-10,
        stepper = SO.CombinedHSDStepper(model, system_solver = s(model)),
        )

    SO.solve(solver)

    @test SO.get_status(solver) == :Optimal
    @show SO.get_primal_obj(solver)
end
