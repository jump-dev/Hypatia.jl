#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

given a sequence of observations X_1,...,X_n with each Xᵢ in Rᵈ,
find a density function f maximizing the log likelihood of the observations
    min -∑ᵢ zᵢ
    -zᵢ + log(f(Xᵢ)) ≥ 0 ∀ i = 1,...,n
    ∫f = 1
    f ≥ 0
==#

using LinearAlgebra
import Random
import Distributions
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
import PolyJuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

function densityestJuMP(
    nobs::Int,
    n::Int,
    deg::Int;
    sample_factor::Int = 100,
    use_monomials::Bool = false,
    )
    X = rand(Distributions.Uniform(-1, 1), nobs, n)
    (nobs, dim) = size(X)
    domain = MU.Box(-ones(n), ones(n))
    d = div(deg + 1, 2)
    (U, pts, P0, PWts, w) = MU.interpolate(domain, d, sample = true, calc_w = true, sample_factor = sample_factor)

    model = JuMP.Model()
    JuMP.@variable(model, z[1:nobs])
    JuMP.@objective(model, Max, sum(z))

    if use_monomials
        lagrange_polys = []
        DynamicPolynomials.@polyvar x[1:dim]
        PX = DynamicPolynomials.monomials(x, 0:2d)

        JuMP.@variable(model, f, PolyJuMP.Poly(PX))
        JuMP.@constraints(model, begin
            sum(w[i] * f(pts[i, :]) for i in 1:U) == 1.0 # integrate to 1
            [f(pts[i, :]) for i in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
            [i in 1:nobs], vcat(z[i], 1.0, f(X[i, :])) in MOI.ExponentialCone() # hypograph of log
        end)
    else
        lagrange_polys = MU.recover_lagrange_polys(pts, 2d)
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

densityestJuMP1() = densityestJuMP(200, 1, 4, use_monomials = false)
densityestJuMP2() = densityestJuMP(200, 1, 4, use_monomials = true)

function test_densityestJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    data = instance()
    JuMP.optimize!(data.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(data.model) == MOI.OPTIMAL
    return
end

test_densityestJuMP(; options...) = test_densityestJuMP.([
    densityestJuMP1,
    densityestJuMP2,
    ], options = options)
