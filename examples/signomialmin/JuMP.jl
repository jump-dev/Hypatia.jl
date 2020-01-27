#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

signomial minimization problem
see "Signomial and Polynomial Optimization via Relative Entropy and Partial Dualization" (2019) by Murray, Chandrasekaran, & Wierman

a signomial f = Sig(A, c) of variables
    x in R^n
with parameters
    c in R^m
    A in R^{m, n}
takes values
    f(x) = sum_{i in [m]} c_i exp(A_{i, :} * x)

the "unconstrained" signomial minimization SAGE relaxation problem is
    max_{g in R} g :
    c - (g, 0, ..., 0) in C_SAGE(A)

a vector d in R^m belongs to SAGE cone C_SAGE(A) if
    exists C in R^{m, m} :
    d = sum_{k in [m]} C_{k, :}
    C_{k, :} in C_AGE(A, k)
which is equivalent to the feasibility problem over C in R^{m, m} and V in R^{m, m-1}
    d = sum_{k in [m]} C_{k, :}
    for k in [m]:
        (A_{\k, :} - [1,...,1] * A_{k, :}')' * V_{k, :} == [0,...,0]
        [C_{k, k} + sum(V_{k, :}), C_{k, \k}, V_{k, :}] in RelEntr(1 + 2(m - 1))
=#

using Test
using LinearAlgebra
import JuMP
const MOI = JuMP.MOI
import Hypatia
import Random

include(joinpath(@__DIR__, "data.jl"))

function signomialminJuMP(
    signomialname::Symbol,
    )
    (c, A, true_obj) = signomials[signomialname]
    (m, n) = size(A)
    @assert length(c) == m

    model = JuMP.Model()
    JuMP.@variable(model, g)
    JuMP.@objective(model, Max, g)
    d = c - vcat(g, zeros(m - 1))

    # setup SAGE constraints
    notk = [[l for l in 1:m if l != k] for k in 1:m]
    JuMP.@variable(model, C[1:m, 1:m])
    JuMP.@variable(model, V[1:m, 1:(m - 1)])
    JuMP.@constraint(model, [k in 1:m], d[k] == sum(C[:, k]))
    JuMP.@constraint(model, [k in 1:m, i in 1:n], dot(A[notk[k], i] .- A[k, i], V[k, :]) == 0)
    JuMP.@constraint(model, [k in 1:m], vcat(C[k, k] + sum(V[k, :]), C[k, notk[k]], V[k, :]) in MOI.RelativeEntropyCone(2m - 1))

    return (model = model, true_obj = true_obj)
end

signomialminJuMP1() = signomialminJuMP(:motzkin)

function test_signomialminJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.objective_value(d.model) ≈ d.true_obj atol = 1e-4 rtol = 1e-4
    return
end

test_signomialminJuMP_all(; options...) = test_signomialminJuMP.([
    signomialminJuMP1,
    # signomialminJuMP2,
    # signomialminJuMP3,
    # signomialminJuMP4,
    # signomialminJuMP5,
    # signomialminJuMP6,
    # signomialminJuMP7,
    ], options = options)

test_signomialminJuMP(; options...) = test_signomialminJuMP.([
    signomialminJuMP1,
    # signomialminJuMP2,
    # signomialminJuMP3,
    # signomialminJuMP4,
    # signomialminJuMP5,
    ], options = options)
