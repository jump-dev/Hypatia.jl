#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

given a sequence of observations X_1,...,X_n with each Xᵢ in Rᵈ,
find a density function f maximizing the log likelihood of the observations
    min -∑ᵢ zᵢ
    -zᵢ + log(f(Xᵢ)) ≥ 0 ∀ i = 1,...,n
    ∫f = 1
    f ≥ 0
==#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const SO = HYP.Solvers
const MO = HYP.Models
const MU = HYP.ModelUtilities

import JuMP
import MathOptInterface
const MOI = MathOptInterface
import PolyJuMP
import MultivariatePolynomials
import DynamicPolynomials
using LinearAlgebra
import Random
import Distributions
using Test

function densityest_JuMP(
    nobs::Int,
    n::Int,
    deg::Int;
    rseed::Int = 1,
    sample_factor::Int = 100,
    use_monomials::Bool = false,
    )
    Random.seed!(rseed)
    X = rand(Distributions.Uniform(-1, 1), nobs, n)
    (nobs, dim) = size(X)
    domain = MU.Box(-ones(n), ones(n))
    d = div(deg + 1, 2)

    model = JuMP.Model()
    (U, pts, P0, PWts, w) = MU.interpolate(domain, d, sample = true, calc_w = true, sample_factor = sample_factor)
    JuMP.@variable(model, z[1:nobs])
    JuMP.@objective(model, Max, sum(z))

    if use_monomials
        lagrange_polys = []
        DynamicPolynomials.@polyvar x[1:dim]
        PX = DynamicPolynomials.monomials(x, 0:(2 * d))
        JuMP.@variable(model, f, PolyJuMP.Poly(PX))
        JuMP.@constraints(model, begin
            sum(w[i] * f(pts[i, :]) for i in 1:U) == 1.0 # integrate to 1
            [f(pts[i, :]) for i in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
            [i in 1:nobs], vcat(z[i], 1.0, f(X[i, :])) in MOI.ExponentialCone() # hypograph of log
        end)
    else
        lagrange_polys = MU.recover_lagrange_polys(pts, 2 * d)
        basis_evals = Matrix{Float64}(undef, nobs, U)
        for i in 1:nobs, j in 1:U
            basis_evals[i, j] = lagrange_polys[j](X[i, :])
        end
        JuMP.@variable(model, f[1:U])
        JuMP.@constraints(model, begin
            dot(w, f) == 1.0 # integrate to 1
            f in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
            [i in 1:nobs], vcat(z[i], 1.0, dot(f, basis_evals[i, :])) in MOI.ExponentialCone() # hypograph of log
        end)
    end
    return (model = model,)
end

densityest1_JuMP() = densityest_JuMP(200, 1, 4, use_monomials = false)
densityest2_JuMP() = densityest_JuMP(200, 1, 4, use_monomials = true)

function test_densityest_JuMP(instance::Function; options)
    data = instance()
    JuMP.optimize!(data.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(data.model) == MOI.OPTIMAL
    return
end

test_densityest_JuMP(; options...) = test_densityest_JuMP.([
    densityest1_JuMP,
    densityest2_JuMP,
    ], options = options)
