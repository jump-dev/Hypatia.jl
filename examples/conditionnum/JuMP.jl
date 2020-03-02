#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

minimize the condition number of positive definite matrix M(x) = M_0 + sum_i x_i*M_i
subject to F(x) = F_0 + sum_i x_i*F_i in S_+

from section 3.2 "Linear Matrix Inequalities in System and Control Theory" by
S. Boyd, L. El Ghaoui, E. Feron, and V. Balakrishnan

original formulation:
min gamma
mu >= 0
F(x) in S_+
M(x) - mu*I in S_+
mu*gamma*I - M(x) in S_+

introduce nu = inv(mu), y = x/mu:
min gamma
nu >= 0
nu*F_0 + sum_i y_i*F_i in S_+
nu*M_0 + sum_i y_i*M_i - I in S_+
gamma*I - nu*M_0 - sum_i y_i*M_i in S_+

we make F_0 and M_0 positive definite to ensure existence of a feasible solution
=#

using LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia

function conditionnum_JuMP(
    T::Type{Float64}, # TODO support generic reals
    side::Int,
    len_y::Int,
    use_linmatrixineq::Bool, # use linmatrixineq cone, else PSD formulation
    )
    Mi = [zeros(side, side) for i in 1:len_y]
    for i in eachindex(Mi)
        Mi_half = randn(side)
        Mi[i] = Symmetric(Mi_half * Mi_half')
    end
    M0 = randn(side, side)
    M0 = Symmetric(M0 * M0')
    Fi = [Symmetric(randn(side, side)) for i in 1:len_y]
    F0 = randn(side, side)
    F0 = Symmetric(F0 * F0')
    # choose to make some F_i matrices pd so several feasible solutions exists
    pd_idxs = rand(1:len_y, max(1, div(len_y, 5)))
    for i in pd_idxs
        Fi[i] = Symmetric(Fi[i] * Fi[i]')
    end

    model = JuMP.Model()
    JuMP.@variables(model, begin
        gamma
        nu >= 0
        y[1:len_y]
    end)
    JuMP.@objective(model, Min, gamma)

    if use_linmatrixineq
        JuMP.@constraints(model, begin
            vcat(nu, y) in Hypatia.LinMatrixIneqCone{Float64}([F0, Fi...])
            vcat(-1, nu, y) in Hypatia.LinMatrixIneqCone{Float64}([I, M0, Mi...])
            vcat(gamma, -nu, -y) in Hypatia.LinMatrixIneqCone{Float64}([I, M0, Mi...])
        end)
    else
        JuMP.@constraints(model, begin
            Symmetric(nu .* F0 + sum(y[i] .* Fi[i] for i in eachindex(y))) in JuMP.PSDCone()
            Symmetric(nu .* M0 + sum(y[i] .* Mi[i] for i in eachindex(y)) - I) in JuMP.PSDCone()
            Symmetric(gamma .* Matrix(I, side, side) - nu .* M0 - sum(y[i] .* Mi[i] for i in eachindex(y))) in JuMP.PSDCone()
        end)
    end

    return (model = model,)
end

function test_conditionnum_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = conditionnum_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

conditionnum_JuMP_fast = [
    (2, 3, true),
    (2, 3, false),
    (3, 2, true),
    (3, 2, false),
    (10, 8, true),
    (10, 8, false),
    (50, 15, true),
    (50, 15, false),
    (100, 10, true),
    (100, 10, false),
    ]
conditionnum_JuMP_slow = [
    # TODO
    ]
