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
using Test
import DataFrames
import CSV
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
import PolyJuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

function densityestJuMP(
    X::AbstractMatrix{Float64},
    deg::Int;
    sample_factor::Int = 100,
    use_monomials::Bool = false,
    )
    (nobs, dim) = size(X)

    domain = MU.Box(-ones(dim), ones(dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, P0, PWts, w) = MU.interpolate(domain, halfdeg, sample = true, calc_w = true, sample_factor = sample_factor)

    model = JuMP.Model()
    JuMP.@variable(model, z[1:nobs])
    JuMP.@objective(model, Max, sum(z))

    if use_monomials
        lagrange_polys = []
        DynamicPolynomials.@polyvar x[1:dim]
        PX = DynamicPolynomials.monomials(x, 0:(2 * halfdeg))

        JuMP.@variable(model, f, PolyJuMP.Poly(PX))
        JuMP.@constraints(model, begin
            sum(w[i] * f(pts[i, :]) for i in 1:U) == 1.0 # integrate to 1
            [f(pts[i, :]) for i in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
            [i in 1:nobs], vcat(z[i], 1.0, f(X[i, :])) in MOI.ExponentialCone() # hypograph of log
        end)
    else
        lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)
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

densityestJuMP(nobs::Int, n::Int, deg::Int, use_monomials::Bool) = densityestJuMP(randn(nobs, n), deg, use_monomials = use_monomials)

# iris dataset
function iris_data()
    df = CSV.read(joinpath(@__DIR__, "data", "iris.csv"))
    # only use setosa species
    dfsub = df[df.species .== "setosa", [:sepal_length, :sepal_width, :petal_length, :petal_width]] # n = 4
    X = convert(Matrix{Float64}, dfsub)
    return X
end

# lung cancer dataset from https://github.com/therneau/survival (cancer.rda)
# description at https://github.com/therneau/survival/blob/master/man/lung.Rd
function cancer_data()
    df = CSV.read(joinpath(@__DIR__, "data", "cancer.csv"), missingstring = "NA", copycols = true)
    DataFrames.dropmissing!(df, disallowmissing = true)
    # only use males with status 2
    dfsub = df[df.status .== 2, :]
    dfsub = dfsub[dfsub.sex .== 1, [:time, :age, :ph_ecog, :ph_karno, :pat_karno, :meal_cal, :wt_loss]] # n = 7
    X = convert(Matrix{Float64}, dfsub)
    return X
end

densityestJuMP1() = densityestJuMP(iris_data(), 4)
densityestJuMP2() = densityestJuMP(iris_data(), 6)
densityestJuMP3() = densityestJuMP(cancer_data(), 4)
densityestJuMP4() = densityestJuMP(cancer_data(), 6)
densityestJuMP5() = densityestJuMP(200, 1, 4, false)
densityestJuMP6() = densityestJuMP(200, 1, 4, true)

function test_densityestJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_densityestJuMP(; options...) = test_densityestJuMP.([
    densityestJuMP1,
    # densityestJuMP2,
    densityestJuMP3,
    # densityestJuMP4,
    densityestJuMP5,
    densityestJuMP6,
    ], options = options)
