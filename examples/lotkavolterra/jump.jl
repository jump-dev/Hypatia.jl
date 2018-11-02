#=
Copyright 2018, Chris Coey and contributors

TODO reference paper for model
TODO options to use standard PSD cone formulation vs interpolation-based WSOS cone formulation
=#

using Hypatia
using MathOptInterface
MOI = MathOptInterface
using MultivariatePolynomials
using DynamicPolynomials
using SemialgebraicSets
using JuMP
using PolyJuMP
using SumOfSquares
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using Test




function int_X_mon(mon, n)
    as = exponents(mon)
    # as = exponents(mon * prod(x_h)^0)
    @assert length(as) == n
    if any(isodd, as)
        return 0
    else
        bs = (as .+ 1) ./ 2
        return 2*prod(gamma.(bs))/gamma(sum(bs))/(sum(as) + n)
    end
end

int_X(p, n) = sum(coefficient(t)*int_X_mon(t, n) for t in terms(p))


println("initializing data")

# Problem parameters
d = 4 # Polynomials degree
n = 4 # Number of species (x)
m = 2*n # Number of control inputs (u)
Q = 0.475 # Transformation parameters: x_h = (x-q)/Q
q = 0.525
l_x = 1 # Cost function on rho
l_u = [-1, 0.5, 0.6, 0.8, 1.1, 2, 4, 6] # Cost function on sigmas
r = [1, 0.6, 0.4, 0.2] # Growth rate of each species

# Initialize polynomial variables. Make transformation implicitly x_h = (x - q)/Q. Work only with x_h through the optimization
@polyvar x_h[1:n]
x_mon = monomials(x_h, 0:d)
x_o = x_h*Q .+ q

A = [1 0.3 0.4 0.2; -0.2 1 0.4 -0.1; -0.1 -0.2 1 0.3; -0.1 -0.2 -0.3 1] # Influence between species
M = sum(abs, l_u)/2 + sum(l_u)/2 + l_x
M *= 0.01 # Upper bound on the total cost

f = r .* x_o .* (1 .- A*x_o)
f_u = hcat(SparseMatrixCSC(-1.0I, n, n), SparseMatrixCSC(1.0I, n, n)) # TODO better to change in formulation directly maybe
brho = 1  # Beta from model
u_bar = 1 # Upper bound on u

# Non-extintion constraint. Do a transformation to have a buffer on extinction
X = @set x_h'*x_h <= 1
delX = @set x_h'*x_h == 1

println("creating model")

model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))

# @variable(model, rho, Poly(x_mon))
# @variable(model, rho_0, Poly(x_mon))
# @variable(model, rho_T, Poly(x_mon))
# @variable(model, sigma[1:m], Poly(x_mon))
#
# @objective(model, Min, -int_X(l_x*rho, n) +
#     sum(int_X(sigma[i]*l_u[i], n) for i in 1:m) +
#     M*int_X(rho_T, n))
#
# @constraint(model, 0 == rho_T - rho_0 + brho*rho +
#     sum(differentiate(rho*f[i], x_h[i])/Q for i in 1:n) +
#     sum(sum(differentiate(sigma[j]*f_u[i,j], x_h[i])/Q for i in 1:n) for j in 1:m))
# @constraint(model, rho <= 0, domain=delX)
# @constraint(model, rho_0 >= 1, domain=X)
# @constraint(model, [i in 1:m], u_bar*rho >= sigma[i], domain=X)
# @constraint(model, rho_T >= 0, domain=X)
# @constraint(model, [i in 1:m], sigma[i] >= 0, domain=X)

# without rho_0

@variable(model, rho, Poly(x_mon))
@variable(model, rho_T, Poly(x_mon))
@variable(model, sigma[1:m], Poly(x_mon))

@objective(model, Min, -int_X(l_x*rho, n) +
    sum(int_X(sigma[i]*l_u[i], n) for i in 1:m) +
    M*int_X(rho_T, n))

@constraint(model, rho <= 0, domain=delX)
@constraint(model, rho_T + brho*rho +
    sum(differentiate(rho*f[i], x_h[i])/Q for i in 1:n) +
    sum(sum(differentiate(sigma[j]*f_u[i,j], x_h[i])/Q for i in 1:n) for j in 1:m)
    >= 1, domain=X)
@constraint(model, [i in 1:m], u_bar*rho >= sigma[i], domain=X)
@constraint(model, rho_T >= 0, domain=X)
@constraint(model, [i in 1:m], sigma[i] >= 0, domain=X)


println("optimizing model")

JuMP.optimize!(model)

term_status = JuMP.termination_status(model)
pobj = JuMP.objective_value(model)
dobj = JuMP.objective_bound(model)
pr_status = JuMP.primal_status(model)
du_status = JuMP.dual_status(model)

@test term_status == MOI.Success
@test pr_status == MOI.FeasiblePoint
@test du_status == MOI.FeasiblePoint
@test pobj ≈ dobj atol=1e-4 rtol=1e-4


#
# function build_JuMP_lotkavolterra(
#     )
#     # build JuMP model

# end
#
# function run_JuMP_lotkavolterra()
#
#
#
#
#     model = build_JuMP_lotkavolterra(...)
#     JuMP.optimize!(model)
#
#     term_status = JuMP.termination_status(model)
#     pobj = JuMP.objective_value(model)
#     dobj = JuMP.objective_bound(model)
#     pr_status = JuMP.primal_status(model)
#     du_status = JuMP.dual_status(model)
#
#     @test term_status == MOI.Success
#     @test pr_status == MOI.FeasiblePoint
#     @test du_status == MOI.FeasiblePoint
#     @test pobj ≈ dobj atol=1e-4 rtol=1e-4
#
#     return nothing
# end
