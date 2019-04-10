#=
Copyright 2018, Chris Coey and contributors
=#

import Hypatia
const HYP = Hypatia
using JuMP
using LinearAlgebra
using Test


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
# @show Symmetric(Md1, :L)
@constraint(model, Symmetric(Md1, :L) in PSDCone())

# solve
optimize!(model)

term_status = termination_status(model)
primal_obj = objective_value(model)
dual_obj = objective_bound(model)
pr_status = primal_status(model)
du_status = dual_status(model)
;
