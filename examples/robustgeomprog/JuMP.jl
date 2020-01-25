#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

robust geometric programming problem
given a convex set C in R_+^k (described by conic constraints) and a matrix B in R^{k, n}, calculate
    f(C, B) = sup_{c in C} (inf_{x in R^n, z in R_+^k} c'*z : B_i*x <= log(z_i), i = 1..k)
note the inner problem is an unconstrained GP, and C specifies coefficient uncertainty
for more details, see section 4.4 of:
"Relative entropy optimization and its applications" (2017) by Chandrasekaran & Shah
the authors show that:
    f(C, B) = -inf_{c in C, v in R_+^k} (d(v, e*c) : B_j'*v = 0, j = 1..n)
where e = exp(1) and d(a, b) = sum_i a_i*log(a_i/b_i) is the relative entropy of a and b
=#

using LinearAlgebra
import JuMP
const MOI = JuMP.MOI
import Hypatia
import Random

function robustgeomprogJuMP(
    n::Int,
    k::Int;
    B::Matrix{Float64} = randn(k, n), # linear constraint matrix
    alphas::Vector{Float64} = rand(k) .+ 1, # for geomean constraint for set C
    betas::Vector{Float64} = rand(k) .+ 1, # for entropy constraint for set C
    )
    @assert n < k # want some degrees of freedom for v
    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)

    JuMP.@variable(model, c[1:k])
    JuMP.@variable(model, v[1:k])

    JuMP.@constraint(model, vcat(t, ℯ * c, v) in Hypatia.EpiSumPerEntropyCone{Float64}(1 + 2k))

    JuMP.@constraint(model, B' * v .== 0)

    # use bounded convex set C of R_+^k excluding origin (note that the entropy constraint already forces c >= 0)
    # satisfy a geomean constraint with powers vector alphas (note c = ones(k) is feasible)
    @assert all(alphas .> 1e-6)
    alphas /= sum(alphas)
    JuMP.@constraint(model, vcat(1, c) in Hypatia.HypoGeomeanCone{Float64}(alphas))
    # satisfy an entropy constraint with perspective vector betas (note c = ones(k) is feasible)
    @assert all(betas .> 1e-6)
    betas /= sum(betas)
    epi_value = -sum(log, betas)
    JuMP.@constraint(model, vcat(epi_value, betas, c) in Hypatia.EpiSumPerEntropyCone{Float64}(1 + 2k))

    return (model = model,)
end

robustgeomprogJuMP1() = robustgeomprogJuMP(3, 4)
robustgeomprogJuMP2() = robustgeomprogJuMP(3, 10)
robustgeomprogJuMP3() = robustgeomprogJuMP(6, 10)
robustgeomprogJuMP4() = robustgeomprogJuMP(6, 20)
robustgeomprogJuMP5() = robustgeomprogJuMP(15, 40)
robustgeomprogJuMP6() = robustgeomprogJuMP(40, 60)
robustgeomprogJuMP7() = robustgeomprogJuMP(50, 100)

function test_robustgeomprogJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_robustgeomprogJuMP_all(; options...) = test_robustgeomprogJuMP.([
    robustgeomprogJuMP1,
    robustgeomprogJuMP2,
    robustgeomprogJuMP3,
    robustgeomprogJuMP4,
    robustgeomprogJuMP5,
    robustgeomprogJuMP6,
    robustgeomprogJuMP7,
    ], options = options)

test_robustgeomprogJuMP(; options...) = test_robustgeomprogJuMP.([
    robustgeomprogJuMP1,
    robustgeomprogJuMP2,
    robustgeomprogJuMP3,
    robustgeomprogJuMP4,
    robustgeomprogJuMP5,
    ], options = options)
