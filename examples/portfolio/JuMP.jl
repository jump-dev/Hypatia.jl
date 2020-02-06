#=
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

function portfolioJuMP(
    num_stocks::Int;
    epipernormeucl_constr::Bool = false,
    epinorminf_constr::Bool = false,
    epinorminfdual_constr::Bool = false,
    )
    returns = rand(num_stocks)
    sigma_half = randn(num_stocks, num_stocks)
    x = randn(num_stocks)
    x ./= norm(x)
    gamma = sum(abs, sigma_half * x) / sqrt(num_stocks)

    model = JuMP.Model()
    JuMP.@variable(model, invest[1:num_stocks] >= 0)
    JuMP.@objective(model, Min, dot(returns, invest))
    JuMP.@constraint(model, sum(invest) == 1)
    if epipernormeucl_constr
        JuMP.@constraint(model, vcat(gamma, sigma_half * invest) in JuMP.SecondOrderCone())
    end
    if epinorminfdual_constr
        JuMP.@constraint(model, vcat(gamma * sqrt(num_stocks), sigma_half * invest) in MOI.NormOneCone(num_stocks + 1))
    end
    if epinorminf_constr
        JuMP.@constraint(model, vcat(gamma, sigma_half * invest) in MOI.NormInfinityCone(num_stocks + 1))
    end
    return (model = model,)
end

portfolioJuMP1() = portfolioJuMP(6, epipernormeucl_constr = true)
portfolioJuMP2() = portfolioJuMP(6, epinorminf_constr = true)
portfolioJuMP3() = portfolioJuMP(6, epinorminfdual_constr = true)
portfolioJuMP4() = portfolioJuMP(6, epipernormeucl_constr = true, epinorminf_constr = true)
portfolioJuMP5() = portfolioJuMP(6, epipernormeucl_constr = true, epinorminfdual_constr = true)
portfolioJuMP6() = portfolioJuMP(6, epinorminf_constr = true, epinorminfdual_constr = true)
portfolioJuMP7() = portfolioJuMP(6, epipernormeucl_constr = true, epinorminf_constr = true, epinorminfdual_constr = true)
portfolioJuMP8() = portfolioJuMP(20, epipernormeucl_constr = true, epinorminf_constr = true)
portfolioJuMP9() = portfolioJuMP(20, epipernormeucl_constr = true, epinorminfdual_constr = true)
portfolioJuMP10() = portfolioJuMP(20, epinorminf_constr = true, epinorminfdual_constr = true)
portfolioJuMP11() = portfolioJuMP(40, epipernormeucl_constr = true, epinorminf_constr = true)
portfolioJuMP12() = portfolioJuMP(40, epipernormeucl_constr = true, epinorminfdual_constr = true)
portfolioJuMP13() = portfolioJuMP(40, epinorminf_constr = true, epinorminfdual_constr = true)

function test_portfolioJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_portfolioJuMP_all(; options...) = test_portfolioJuMP.([
    portfolioJuMP1,
    portfolioJuMP2,
    portfolioJuMP3,
    portfolioJuMP4,
    portfolioJuMP5,
    portfolioJuMP6,
    portfolioJuMP7,
    portfolioJuMP8,
    portfolioJuMP9,
    portfolioJuMP10,
    ], options = options)

test_portfolioJuMP(; options...) = test_portfolioJuMP.([
    portfolioJuMP1,
    portfolioJuMP2,
    portfolioJuMP3,
    portfolioJuMP4,
    portfolioJuMP5,
    portfolioJuMP6,
    portfolioJuMP7,
    portfolioJuMP8,
    portfolioJuMP9,
    portfolioJuMP10,
    portfolioJuMP11,
    portfolioJuMP12,
    portfolioJuMP13,
    ], options = options)
