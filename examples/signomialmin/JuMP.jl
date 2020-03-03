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
import Random
import JuMP
const MOI = JuMP.MOI
import Hypatia

include(joinpath(@__DIR__, "data.jl"))

function signomialmin_JuMP(
    ::Type{T},
    fc::Vector,
    fA::AbstractMatrix,
    gc::Vector,
    gA::Vector,
    obj_ub::Real,
    ) where {T <: Float64} # TODO support generic reals
    (fm, n) = size(fA)
    @assert length(fc) == fm
    q = length(gc)
    @assert length(gA) == q
    @assert all(size(gA_p, 2) == n for gA_p in gA)
    @assert all(size(gA_p, 1) == length(gc_p) for (gc_p, gA_p) in zip(gc, gA))

    # find unique terms
    unique_terms = Dict{Vector, Tuple{Vector{Int}, Vector{Tuple{Int, Int}}}}()
    for k in 1:size(fA, 1)
        row_k = fA[k, :]
        if row_k in keys(unique_terms)
            push!(unique_terms[row_k][1], k)
        else
            unique_terms[row_k] = (Int[k], Tuple{Int, Int}[])
        end
    end
    for (p, gA_p) in enumerate(gA)
        for k in 1:size(gA_p, 1)
            p_row_k = gA_p[k, :]
            if p_row_k in keys(unique_terms)
                push!(unique_terms[p_row_k][2], (p, k))
            else
                unique_terms[p_row_k] = (Int[], Tuple{Int, Int}[(p, k)])
            end
        end
    end
    A = vcat((row_k' for row_k in keys(unique_terms))...)
    m = size(A, 1)

    model = JuMP.Model()
    JuMP.@variable(model, γ)
    JuMP.@objective(model, Max, γ)
    JuMP.@variable(model, μ[1:q] >= 0)

    d = zeros(JuMP.AffExpr, m)
    const_found = false
    for k in 1:m
        row_k = A[k, :]
        (ks, pks) = unique_terms[row_k]
        for k2 in ks
            d[k] += fc[k2]
        end
        for (p, k2) in pks
            d[k] -= μ[p] * gc[p][k2]
        end
        if iszero(norm(row_k))
            # row is a constant term
            d[k] -= γ
            @assert !const_found
            const_found = true
        end
    end
    if !const_found
        A = vcat(A, zeros(A, 1, n))
        d = vcat(d, -γ)
    end

    # setup SAGE constraints
    notk = [[l for l in 1:m if l != k] for k in 1:m]
    JuMP.@variable(model, C[1:m, 1:m])
    JuMP.@variable(model, V[1:m, 1:(m - 1)])
    JuMP.@constraint(model, [k in 1:m], d[k] == sum(C[:, k]))
    JuMP.@constraint(model, [k in 1:m, i in 1:n], dot(A[notk[k], i] .- A[k, i], V[k, :]) == 0)
    JuMP.@constraint(model, [k in 1:m], vcat(C[k, k] + sum(V[k, :]), C[k, notk[k]], V[k, :]) in MOI.RelativeEntropyCone(2m - 1))

    return (model = model, obj_ub = obj_ub)
end

signomialmin_JuMP(
    ::Type{T},
    sig::Symbol,
    ) where {T <: Float64} = signomialmin_JuMP(T, signomialmin_data[sig]...)

signomialmin_JuMP(
    ::Type{T},
    m::Int,
    n::Int,
    ) where {T <: Float64} = signomialmin_JuMP(T, signomialmin_random(m, n)...)

function test_signomialmin_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = signomialmin_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    if !isnan(d.obj_ub)
        @test JuMP.objective_value(d.model) <= d.obj_ub + 1e-4
    end
    return d.model.moi_backend.optimizer.model.optimizer.result
end

signomialmin_JuMP_fast = [
    (:motzkin2,),
    (:motzkin3,),
    (:CS16ex8_13,),
    (:CS16ex8_14,),
    (:CS16ex18,),
    (:CS16ex12,),
    (:CS16ex13,),
    (:MCW19ex1_mod,),
    (:MCW19ex8,),
    (3, 2),
    (3, 3),
    (4, 3),
    (6, 2),
    (6, 4),
    (6, 6),
    (8, 4),
    ]
signomialmin_JuMP_slow = [
    # TODO
    ]
