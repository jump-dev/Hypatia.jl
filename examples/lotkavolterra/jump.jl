#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

TODO reference paper for model
TODO options to use standard PSD cone formulation vs interpolation-based WSOS cone formulation
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const SO = HYP.Solvers
const MO = HYP.Models
const MU = HYP.ModelUtilities

import MathOptInterface
const MOI = MathOptInterface
import JuMP
import MultivariatePolynomials
import DynamicPolynomials
import SemialgebraicSets
import SumOfSquares
import PolyJuMP
using LinearAlgebra
import GSL: sf_gamma
using Test

function integrate_ball_monomial(mon, n)
    as = DynamicPolynomials.exponents(mon)
    @assert length(as) == n
    if any(isodd, as)
        return 0.0
    else
        bs = (as .+ 1) ./ 2.0
        return 2.0 * prod(sf_gamma.(bs)) / (sf_gamma(sum(bs)) * (sum(as) + n))
    end
end

integrate_ball(p, n) = sum(DynamicPolynomials.coefficient(t) * integrate_ball_monomial(t, n) for t in DynamicPolynomials.terms(p))

function build_JuMP_lotkavolterra_PSD(model)
    # parameters
    d = 4 # degree
    n = 4 # number of species
    m = 2 * n # number of control inputs (u)
    Q = 0.475
    q = 0.525
    l_x = 1.0 # cost of rho
    l_u = [-1.0, 0.5, 0.6, 0.8, 1.1, 2.0, 4.0, 6.0] # cost of sigmas
    r = [1.0, 0.6, 0.4, 0.2] # growth rate of species

    DynamicPolynomials.@polyvar x_h[1:n]
    x_mon = DynamicPolynomials.monomials(x_h, 0:d)
    x_o = x_h * Q .+ q
    A = [1.0 0.3 0.4 0.2; -0.2 1.0 0.4 -0.1; -0.1 -0.2 1.0 0.3; -0.1 -0.2 -0.3 1.0]
    M = (sum(abs, l_u) + sum(l_u)) / 2.0 + l_x
    M *= 0.01 # upper bound on the total cost
    f = r .* x_o .* (1.0 .- A * x_o)
    f_u = hcat(Matrix(-1.0I, n, n), Matrix(1.0I, n, n))
    brho = 1.0
    u_bar = 1.0 # upper bound on u
    X = SemialgebraicSets.@set x_h' * x_h <= 1.0 # non-extinction domain
    delta_X = SemialgebraicSets.@set x_h' * x_h == 1.0

    JuMP.@variable(model, rho, PolyJuMP.Poly(x_mon))
    JuMP.@variable(model, rho_T, PolyJuMP.Poly(x_mon))
    JuMP.@variable(model, sigma[1:m], PolyJuMP.Poly(x_mon))

    JuMP.@objective(model, Min, integrate_ball(l_x * rho, n) +
        sum(integrate_ball(sigma[i] * l_u[i], n) for i in 1:m) +
        M * integrate_ball(rho_T, n))

    JuMP.@constraint(model, rho <= 0, domain = delta_X)
    JuMP.@constraint(model, rho_T + brho * rho +
        sum(DynamicPolynomials.differentiate(rho * f[i], x_h[i]) / Q for i in 1:n) +
        sum(sum(DynamicPolynomials.differentiate(sigma[j] * f_u[i, j], x_h[i]) / Q for i in 1:n) for j in 1:m)
        >= 1, domain = X)
    JuMP.@constraint(model, [i in 1:m], u_bar * rho >= sigma[i], domain = X)
    JuMP.@constraint(model, rho_T >= 0, domain = X)
    JuMP.@constraint(model, [i in 1:m], sigma[i] >= 0, domain = X)

    return (sigma, rho)
end

function JuMP_lotkavolterra()
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer,
        use_dense = true,
        verbose = true,
        system_solver = SO.QRCholCombinedHSDSystemSolver,
        linear_model = MO.PreprocessedLinearModel,
        max_iters = 1000,
        time_limit = 3.6e4,
        tol_rel_opt = 1e-5,
        tol_abs_opt = 1e-6,
        tol_feas = 1e-6,
        ))
    build_JuMP_lotkavolterra_PSD(model)
    return model
end

function run_JuMP_lotkavolterra()
    model = JuMP_lotkavolterra()
    JuMP.optimize!(model)

    term_status = JuMP.termination_status(model)
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT
    @test primal_obj ≈ dual_obj atol = 1e-4 rtol = 1e-4

    return
end

# run_JuMP_lotkavolterra()
