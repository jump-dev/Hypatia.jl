#=
Copyright 2018, Chris Coey and contributors
=#

import Hypatia
const HYP = Hypatia
using JuMP
using LinearAlgebra
using Combinatorics


# inf 1 - 4/3 * |z₁|² + 7/18 * |z₁|⁴ : 1 - |z₁|² - |z₂|² = 0
# f - γ - σg is Hermitian-SOS, σ is an unconstrained poly
# optimal value 1/18
# at order 2 of hierarchy
n = 2
deg = 4
αs = [α for t in 0:deg for α in Combinatorics.multiexponents(n, t)]
T = binomial(n + deg, n)
@assert length(αs) == T

@show αs
# f(z1, z2) = 1 - 4/3 * abs2(z1) + 7/18 * abs2(abs2(z1))
# g(z1, z2) = 1 - abs2(z1) - abs2(z2)


# dual formulation
model = Model(with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-8, tol_rel_opt = 1e-7, tol_abs_opt = 1e-8))

# TODO make matrix of AffExpr for yr and yi?
@variable(model, yr[a in 1:T, b in 1:a; a > 1 || b > 1]) # real part lower triangle
@variable(model, yi[a in 0:T, b in 0:(a - 1)]) # complex part lower triangle no diagonal

@objective(model, Min, 1 - 4/3 * )

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
