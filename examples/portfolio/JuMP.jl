400#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

using LinearAlgebra
import JuMP
const MOI = JuMP.MOI
import Random
using Test
import Hypatia
const CO = Hypatia.Cones

function portfolio_JuMP(
    T::Type{Float64}, # TODO support generic reals
    num_stocks::Int,
    epipernormeucl_constr::Bool, # add an L2 ball constraint, else don't add
    epinorminf_constrs::Bool, # add L1 and Linfty ball constraints, elsle don't add
    )
    returns = rand(num_stocks)
    sigma_half = randn(num_stocks, num_stocks)
    x = randn(num_stocks)
    x ./= norm(x)
    gamma = sum(abs, sigma_half * x) / sqrt(num_stocks)

    model = JuMP.Model()
    JuMP.@variable(model, invest[1:num_stocks] >= 0)
    JuMP.@objective(model, Max, dot(returns, invest))
    JuMP.@constraint(model, sum(invest) == 1)

    aff_expr = sigma_half * invest
    if epipernormeucl_constr
        JuMP.@constraint(model, vcat(gamma, aff_expr) in JuMP.SecondOrderCone())
    end
    if epinorminf_constrs
        JuMP.@constraint(model, vcat(gamma * sqrt(num_stocks), aff_expr) in MOI.NormOneCone(num_stocks + 1))
        JuMP.@constraint(model, vcat(gamma, aff_expr) in MOI.NormInfinityCone(num_stocks + 1))
    end

    return (model = model,)
end

function test_portfolio_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = portfolio_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

portfolio_JuMP_fast = [
    (3, true, false),
    (3, false, true),
    (3, true, true),
    (10, true, false),
    (10, false, true),
    (10, true, true),
    (50, true, false),
    (50, false, true),
    (50, true, true),
    (400, true, false),
    (400, false, true),
    (400, true, true),
    (400, true, false),
    (400, false, true),
    (400, true, true),
    ]
portfolio_JuMP_slow = [
    # TODO
    ]
