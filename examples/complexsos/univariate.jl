#=
Copyright 2018, Chris Coey and contributors
=#

import Hypatia
const HYP = Hypatia
using JuMP
using LinearAlgebra
using Test

# TODO tidy up 2 formulations (herm PSD and interp-herm-wsos). maybe refac into 2 functions
# use given or random f (requires automatic construction of M0, M1 - see old code in univarite4.jl)
# check objective values of 2 formulations match


# inf z + ̄z : |z|² ≤ 1
# ≡ inf f(z) = 2Re(z) : g₁(z) = 1 - |z|² ≥ 0
# optimal value -2, for z = -1 + 0im
# z + ̄z - (-2) = |1 + z|² + |1|² × (1 - |z|²)
deg = 2

# dual formulation
model = Model(with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-8, tol_rel_opt = 1e-7, tol_abs_opt = 1e-8))

@variable(model, yr[i in 0:deg, j in 0:i; i > 0 || j > 0]) # real part lower triangle, with yr[0, 0] = 1 below
@variable(model, yi[i in 0:deg, j in 0:(i - 1)]) # complex part lower triangle no diagonal

@objective(model, Min, 2 * yr[1, 0])

# Md0(y) constraint
# lower triangle of real PSD formulation
K = 2 * (deg + 1)
Md0 = zeros(AffExpr, K, K)
for i in 0:deg
    Md0[i + 1, i + 1] = (i == 0) ? 1 : yr[i, i]
    Md0[i + deg + 2, i + 1] = 0
    for j in 0:(i - 1)
        Md0[i + 1, j + 1] = Md0[i + deg + 2, j + deg + 2] = yr[i, j]
        Md0[i + deg + 2, j + 1] = yi[i, j]
    end
end
Md0[(deg + 2):K, (deg + 2):K] = Md0[1:(deg + 1), 1:(deg + 1)]
Md0[(deg + 2):K, 1:(deg + 1)] += -Md0[(deg + 2):K, 1:(deg + 1)]'
# @show Symmetric(Md0, :L)
@constraint(model, Symmetric(Md0, :L) in PSDCone())

# Md1(g*y) constraint
# lower triangle of real PSD formulation
K = 4
Md1 = zeros(AffExpr, K, K)
Md1[1, 1] = 1 - yr[1, 1]
Md1[2, 1] = yr[1, 0] - yr[2, 1]
Md1[2, 2] = yr[1, 1] - yr[2, 2]
Md1[3:4, 3:4] = Md1[1:2, 1:2]
Md1[4, 1] = yi[1, 0] - yi[2, 1]
Md1[3, 2] = -Md1[4, 1]
@show Symmetric(Md1, :L)
@constraint(model, Symmetric(Md1, :L) in PSDCone())

# solve
optimize!(model)

term_status = termination_status(model)
primal_obj = objective_value(model)
dual_obj = objective_bound(model)
pr_status = primal_status(model)
du_status = dual_status(model)
;



# f(z) = -1 + 4*real(z)# + abs(z)^2 + real(z^2) + abs(z)^3 # + sum(abs(z)^i for i in 2:2:d))
# f(z) = 1 + real(z) + abs(z)^2 + real(z^2) + real(z^2 * conj(z)) + abs(z)^4
# f(z) = 1 + real(z)# + 3abs(z)^2
# f(z) = -1 + 2.5abs(z)^2 + 0.5abs(z)^4

f(z) = 1 + 2real(z) + abs(z)^2 + 2real(z^2) + 2real(z^2 * conj(z)) + abs(z)^4
# mathematica: min is 0
# Minimize[1+2x+x^2+y^2+2(x^2-y^2)+2(x^3+x*y^2)+(x^2+y^2)^2,{x,y}]


# # random
# Fh = randn(ComplexF64, d + 1, d + 1)
# F = Hermitian((rand() > 0.5) ? Fh * Fh' : Fh)
# @show isposdef(F)
# f(z) = real(sum(F[i+1, j+1] * z^i * conj(z)^j for i in 0:d, j in 0:d))


sample_factor = 1000

U = (d + 1)^2
V_basis = [z -> z^i * conj(z)^j for j in 0:d for i in 0:d] # TODO columns are dependent if not doing j in 0:i

# sample
radii = sqrt.(rand(sample_factor * U))
angles = rand(sample_factor * U) .* 2pi
all_points = radii .* (cos.(angles) .+ (sin.(angles) .* im))

V = [b(p) for p in all_points, b in V_basis]
@test rank(V) == U
VF = qr(Matrix(transpose(V)), Val(true))
keep = VF.p[1:U]
points = all_points[keep]
V = V[keep, :]
@test rank(V) == U

# setup P0
v0 = ones(U)
# P0 = V[:, 1:L0]
P0 = [p^i for p in points, i in 0:d]
# P0 = Matrix(qr(P0).Q)

# setup P1
g1(z) = 1 - abs2(z)
v1 = [g1(p) for p in points]
P1 = [p^i for p in points, i in 0:d-1]
# P1 = Matrix(qr(P1).Q)

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
cones = [CO.WSOSPolyInterp_2(U, [P0, P1], [v0, v1], !primal_wsos)]
cone_idxs = [1:U]
