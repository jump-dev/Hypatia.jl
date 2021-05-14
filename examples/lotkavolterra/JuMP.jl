#=
control of Lotka-Volterra population dynamics
adapted from
"Moment-sum-of-squares hierarchies for set approximation and optimal control"
by Milan Korda (2016)

TODO
- add options to use standard PSD cone formulation vs WSOS cone formulation
- add random data generation
=#

import SpecialFunctions: gamma
import DynamicPolynomials
const DP = DynamicPolynomials
import SemialgebraicSets
const SAS = SemialgebraicSets
import SumOfSquares
import PolyJuMP

struct LotkaVolterraJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    deg::Int # polynomial degrees
end

function build(inst::LotkaVolterraJuMP{T}) where {T <: Float64}
    # parameters
    n = 4 # number of species
    m = 2 * n # number of control inputs (u)
    Q = 0.475
    q = 0.525
    l_x = 1.0 # cost of rho
    l_u = [-1.0, 0.5, 0.6, 0.8, 1.1, 2.0, 4.0, 6.0] # cost of sigmas
    r = [1.0, 0.6, 0.4, 0.2] # growth rate of species

    DP.@polyvar x_h[1:n]
    x_mon = DP.monomials(x_h, 0:inst.deg)
    x_o = x_h * Q .+ q
    A = [1.0 0.3 0.4 0.2; -0.2 1.0 0.4 -0.1;
        -0.1 -0.2 1.0 0.3; -0.1 -0.2 -0.3 1.0]
    M = (sum(abs, l_u) + sum(l_u)) / 2.0 + l_x
    M *= 0.01 # upper bound on the total cost
    f = r .* x_o .* (1.0 .- A * x_o)
    f_u = hcat(Matrix(-1.0I, n, n), Matrix(1.0I, n, n))
    brho = 1.0
    u_bar = 1.0 # upper bound on u
    X = SAS.@set x_h' * x_h <= 1.0 # non-extinction domain
    delta_X = SAS.@set x_h' * x_h == 1.0

    function integrate_ball_monomial(mon, n)
        as = DP.exponents(mon)
        @assert length(as) == n
        if any(isodd, as)
            return 0.0
        else
            bs = (as .+ 1) ./ 2.0
            return 2.0 * prod(gamma.(bs)) / (gamma(sum(bs)) * (sum(as) + n))
        end
    end

    integrate_ball(p, n) = sum(DP.coefficient(t) * integrate_ball_monomial(t, n)
        for t in DP.terms(p))

    model = SumOfSquares.SOSModel()
    JuMP.@variable(model, rho, PolyJuMP.Poly(x_mon))
    JuMP.@variable(model, rho_T, PolyJuMP.Poly(x_mon))
    JuMP.@variable(model, sigma[1:m], PolyJuMP.Poly(x_mon))

    JuMP.@objective(model, Min, integrate_ball(l_x * rho, n) +
        sum(integrate_ball(sigma[i] * l_u[i], n) for i in 1:m) +
        M * integrate_ball(rho_T, n))

    JuMP.@constraint(model, rho <= 0, domain = delta_X)
    JuMP.@constraint(model, rho_T + brho * rho +
        sum(DP.differentiate(rho * f[i], x_h[i]) / Q for i in 1:n) +
        sum(sum(DP.differentiate(sigma[j] * f_u[i, j], x_h[i]) / Q
            for i in 1:n) for j in 1:m) >= 1, domain = X)
    JuMP.@constraint(model, [i in 1:m], u_bar * rho >= sigma[i], domain = X)
    JuMP.@constraint(model, rho_T >= 0, domain = X)
    JuMP.@constraint(model, [i in 1:m], sigma[i] >= 0, domain = X)

    return model
end
