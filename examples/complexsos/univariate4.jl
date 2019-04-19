#=
Copyright 2018, Chris Coey and contributors
=#

import Hypatia
const HYP = Hypatia
using JuMP
using MathOptInterface
using LinearAlgebra
using Test


# inf z + ̄z : |z|² ≤ 1
# ≡ inf f(z) = 2Re(z) : g₁(z) = 1 - |z|² ≥ 0
# optimal value -2, for z = -1 + 0im
# z + ̄z - (-2) = |1 + z|² + |1|² × (1 - |z|²)
deg = 2

model = Model(with_optimizer(HYP.Optimizer, verbose = true))#, tol_feas = 1e-8, tol_rel_opt = 1e-7, tol_abs_opt = 1e-8))

@variable(model, yr[i in 0:deg, j in 0:i; i > 0 || j > 0]) # real part lower triangle, with yr[0, 0] = 1 below
@variable(model, yi[i in 0:deg, j in 0:(i - 1)]) # complex part lower triangle no diagonal

@objective(model, Min, 2 * yr[1, 0])
# @objective(model, Min, 2 * yr[1, 1])
# @objective(model, Min, 2yr[1, 0] - yr[2, 2])

sidedims = Int[]
Mfuncs = Function[]

lr = length(yr)
li = length(yi)

initpoint = zeros(1 + lr + li)
kr = 0
for i in 1:deg+1
    global kr += i
    initpoint[kr] = deg + 2 - i
end
@show initpoint

# Md0(y) constraint
push!(sidedims, deg + 1)
function M0(x::AbstractVector)
    reals = x[1:lr+1]
    imags = x[lr+2:end]
    M = similar([reals[1] + im * imags[1]], deg + 1, deg + 1)
    kr = 1
    ki = 1
    for i in 1:deg+1
        for j in 1:i-1
            M[i, j] = reals[kr] + im * imags[ki]
            kr += 1
            ki += 1
        end
        M[i, i] = reals[kr]
        kr += 1
    end
    M = Hermitian(M, :L)
    return M
end
push!(Mfuncs, M0)

# Md1(g1*y) constraint
# g1 = [1.0 0.0; 0.0 -1.0]
g1 = [(0, 0, 1.0), (1, 1, -1.0)]

push!(sidedims, deg)
function M1(x::AbstractVector)
    Mz = M0(x)
    M = similar(Mz, deg, deg)
    for i in 1:deg, j in 1:i
        M[i, j] = sum(v * Mz[i+k, j+l] for (k, l, v) in g1)
    end
    M = Hermitian(M, :L)
    return M
end
push!(Mfuncs, M1)

@show M1(initpoint)

cone = Hypatia.WSOSComplexCone(1 + lr + li, sidedims, initpoint, Mfuncs, true)
@constraint(model, AffExpr[1.0, yr..., yi...] in cone)

# solve
optimize!(model)

term_status = termination_status(model)
primal_obj = objective_value(model)
dual_obj = objective_bound(model)
pr_status = primal_status(model)
du_status = dual_status(model)

@show primal_obj
;
