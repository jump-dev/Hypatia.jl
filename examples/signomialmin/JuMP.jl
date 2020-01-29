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

for signomials f, (g_p)_{p in 1:q}, the signomial minimization problem is
    inf_{x in R^n} f(x) : g_p(x) >= 0, p in 1:q

the convex SAGE relaxation problem is
    max_{γ in R, μ in R_+^q} γ : c - γ - sum_{p in 1:q} μ_p * g_p in C_SAGE(A)

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
    fc::Vector{<:Real},
    fA::Matrix{<:Real},
    gc::Vector,
    gA::Vector;
    x::Vector = [],
    obj_ub = NaN,
    )
    (mc, n) = size(fA)
    @assert length(fc) == mc
    if isnan(obj_ub)
        @assert !isempty(x)
        for (gc_p, gA_p) in zip(gc, gA)
            @show eval_signomial(gc_p, gA_p, x)
        end
        @assert all(eval_signomial(gc_p, gA_p, x) >= 0 for (gc_p, gA_p) in zip(gc, gA))
        obj_ub = eval_signomial(fc, fA, x)
    end
    q = length(gc)
    @assert length(gA) == q
    @assert all(size(gA_p, 2) == n for gA_p in gA)
    @assert all(size(gA_p, 1) == length(gc_p) for (gc_p, gA_p) in zip(gc, gA))

    # let c and A contain all terms
    # currently assuming terms in each g_p are unique among all terms in f and gs
    # TODO merge like-terms
    A = vcat(fA, gA...)
    m = size(A, 1)

    model = JuMP.Model()
    JuMP.@variable(model, γ)
    JuMP.@objective(model, Max, γ)
    JuMP.@variable(model, μ[1:q] >= 0)
    d = vcat(fc, zeros(m - mc)) - vcat(γ, zeros(m - 1)) - vcat(zeros(mc), vcat((μ .* gc)...))

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
    (fc, fA, gc, gA, x, obj_ub) = signomials[signomial_name]
    return signomialminJuMP(fc, fA, gc, gA; x = x, obj_ub = obj_ub)
end

function signomialminJuMP(m::Int, n::Int)
    (fc, fA, gc, gA, obj_ub) = random_signomial(m, n)
    return signomialminJuMP(fc, fA, gc, gA; obj_ub = obj_ub)
end

signomialminJuMP1() = signomialminJuMP(:motzkin2)
signomialminJuMP2() = signomialminJuMP(:motzkin3)
signomialminJuMP3() = signomialminJuMP(:CS16ex8_13)
signomialminJuMP4() = signomialminJuMP(:CS16ex8_14)
signomialminJuMP5() = signomialminJuMP(:CS16ex18)
signomialminJuMP6() = signomialminJuMP(:CS16ex12)
signomialminJuMP7() = signomialminJuMP(:CS16ex13)
# signomialminJuMP6() = signomialminJuMP(3, 2)
# signomialminJuMP7() = signomialminJuMP(4, 4)
# signomialminJuMP8() = signomialminJuMP(8, 4)
# signomialminJuMP9() = signomialminJuMP(12, 6)

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
    # signomialminJuMP6,
    signomialminJuMP7,
    # signomialminJuMP8,
    # signomialminJuMP9,
    ], options = options)
