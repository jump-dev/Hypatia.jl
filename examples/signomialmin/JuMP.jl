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
    c::Vector{<:Real},
    A::Matrix{<:Real};
    x::Vector = [],
    obj_ub = NaN,
    )
    (m, n) = size(A)
    @assert length(c) == m
    if isnan(obj_ub)
        @assert !isempty(x)
        obj_ub = eval_signomial(c, A, x)
    end

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

    return (model = model, obj_ub = obj_ub)
end

function signomialminJuMP(signomial_name::Symbol)
    (c, A, x, obj_ub) = signomials[signomial_name]
    return signomialminJuMP(c, A; x = x, obj_ub = obj_ub)
end

function signomialminJuMP(m::Int, n::Int)
    (c, A, obj_ub) = random_signomial(m, n)
    return signomialminJuMP(c, A; obj_ub = obj_ub)
end

signomialminJuMP1() = signomialminJuMP(:motzkin2)
signomialminJuMP2() = signomialminJuMP(:motzkin3)
signomialminJuMP3() = signomialminJuMP(:CS16ex8_13)
signomialminJuMP4() = signomialminJuMP(:CS16ex8_14)
signomialminJuMP5() = signomialminJuMP(:CS16ex18)
signomialminJuMP6() = signomialminJuMP(3, 2)
signomialminJuMP7() = signomialminJuMP(4, 4)
signomialminJuMP8() = signomialminJuMP(8, 4)
signomialminJuMP9() = signomialminJuMP(12, 6)

function test_signomialminJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.objective_value(d.model) <= d.obj_ub
    @show JuMP.objective_value(d.model)
    @show d.obj_ub
    return
end

test_signomialminJuMP_all(; options...) = test_signomialminJuMP.([
    signomialminJuMP1,
    signomialminJuMP2,
    signomialminJuMP3,
    signomialminJuMP4,
    signomialminJuMP5,
    signomialminJuMP6,
    signomialminJuMP7,
    signomialminJuMP8,
    signomialminJuMP9,
    ], options = options)

test_signomialminJuMP(; options...) = test_signomialminJuMP.([
    # signomialminJuMP1,
    # signomialminJuMP2,
    # signomialminJuMP3,
    # signomialminJuMP4,
    # signomialminJuMP5,
    signomialminJuMP6,
    signomialminJuMP7,
    signomialminJuMP8,
    signomialminJuMP9,
    ], options = options)
