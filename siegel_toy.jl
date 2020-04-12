#
using JuMP
using Random
using Hypatia
MU = Hypatia.ModelUtilities
CO = Hypatia.Cones
using LinearAlgebra

n = 5
m = 15
A = randn(n, n)
A = Symmetric(A * A')
B = randn(n, m)
P = -A
C = randn(n, n)
# U_data = -P * A - A * P - C' * C / 100
U_data = randn(n, n)
U_data = U_data * U_data'
# W = randn(n, m)

model = JuMP.Model()
@variable(model, W[1:n, 1:m])
@variable(model, t)
U_vec = zeros(CO.svec_length(n))
CO.smat_to_svec!(U_vec, U_data, sqrt(2))


@constraint(model, vcat(U_vec, t / 2, vec(W)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(n, m))

# @constraint(model, Symmetric([t .* Matrix(I, m, m) W'; W U_data]) in JuMP.PSDCone())

@objective(model, Min, t)

set_optimizer(model, Hypatia.Optimizer)
@time optimize!(model)
