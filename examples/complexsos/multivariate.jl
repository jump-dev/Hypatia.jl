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
import Combinatorics
using Test


# inf 1 - 4/3 * |z₁|² + 7/18 * |z₁|⁴ : 1 - |z₁|² - |z₂|² = 0
# f - γ - σg is Hermitian-SOS, σ is an unconstrained poly
# optimal value 1/18
# at order 2 of hierarchy

f(z1, z2) = 1 - 4/3 * abs2(z1) + 7/18 * abs2(abs2(z1))
# f(z1, z2) = 1 - 4/3 * abs2(z1) + 7/18 * abs2(abs2(z1)) - (17/18 - 7/18 * abs2(z1) + 2/3 * abs2(z2)) * (1 - abs2(z1) - abs2(z2))
g(z1, z2) = 1 - abs2(z1) - abs2(z2)
n = 2
deg = 4

# setup interpolation
U = binomial(n + deg, n)
basis = [(p -> prod(p[i]^a[i] for i in eachindex(a))) for t in 0:deg for a in Combinatorics.multiexponents(n, t)]
@test length(basis) == U

# # points are sampled from Cartesian product of roots of unity
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
sample_factor = 100
points = 2 * rand(ComplexF64, sample_factor * U, 2) .- (1 + im)

V = ComplexF64[b(p) for p in eachrow(points), b in basis]
@test rank(V) == U
F = qr(Matrix(V'), Val(true))
keep = F.p[1:U]
points = points[keep, :]
V = V[keep, :]
@test rank(V) == U

# setup P0
L0 = binomial(n + 2, n)
v0 = ones(U)
P0 = V[:, 1:L0]

# setup problem data
# f - γ - σg is Hermitian-SOS, σ is an unconstrained poly
c = Float64[-1, 0, 0, 0] # γ, σ
# c = [-1.0]
A = zeros(0, 4)
# A = zeros(3, 4); b = zeros(3); A[1, 2] = 1; b[1] = 17/18; A[2, 3] = 1; b[2] = -7/18; A[3, 4] = 1; b[3] = 2/3;
# A = zeros(0, 1)
b = Float64[]

g1 = [g(p[1], p[2]) for p in eachrow(points)]
gz1 = [g(p[1], p[2]) * abs2(p[1]) for p in eachrow(points)]
gz2 = [g(p[1], p[2]) * abs2(p[2]) for p in eachrow(points)]
G = [ones(U) g1 gz1 gz2]
# G = ones(U, 1)

h = [f(p[1], p[2]) for p in eachrow(points)]

cones = [CO.WSOSPolyInterp_Complex(U, [P0], [v0], false)]
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
        tol_rel_opt = 1e-10,
        tol_abs_opt = 1e-10,
        tol_feas = 1e-10,
        stepper = SO.CombinedHSDStepper(model, system_solver = s(model)),
        )

    SO.solve(solver)

    @test SO.get_status(solver) == :Optimal
    @show SO.get_primal_obj(solver)
end
