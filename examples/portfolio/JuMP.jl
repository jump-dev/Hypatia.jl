#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))

struct PortfolioJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_stocks::Int
    epipernormeucl_constr::Bool # add an L2 ball constraint, else don't add
    epinorminf_constrs::Bool # add L1 and Linfty ball constraints, elsle don't add
end

example_tests(::Type{PortfolioJuMP{Float64}}, ::MinimalInstances) = [
    ((3, true, false), false),
    ((3, false, true), false),
    ((3, true, true), false),
    ]
example_tests(::Type{PortfolioJuMP{Float64}}, ::FastInstances) = [
    ((10, true, false), false),
    ((10, false, true), false),
    ((10, true, true), false),
    ((50, true, false), false),
    ((50, false, true), false),
    ((50, true, true), false),
    ((400, true, false), false),
    ((400, false, true), false),
    ((400, true, true), false),
    ((400, true, false), false),
    ((400, false, true), false),
    ((400, true, true), false),
    ]
example_tests(::Type{PortfolioJuMP{Float64}}, ::SlowInstances) = [
    ((1000, true, false), false),
    ((1000, false, true), false),
    ((1000, true, true), false),
    ((3000, true, false), false),
    ((3000, false, true), false),
    ((3000, true, true), false),
    ]

function build(inst::PortfolioJuMP{T}) where {T <: Float64} # TODO generic reals
    num_stocks = inst.num_stocks
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
    if inst.epipernormeucl_constr
        JuMP.@constraint(model, vcat(gamma, aff_expr) in JuMP.SecondOrderCone())
    end
    if inst.epinorminf_constrs
        JuMP.@constraint(model, vcat(gamma * sqrt(num_stocks), aff_expr) in MOI.NormOneCone(num_stocks + 1))
        JuMP.@constraint(model, vcat(gamma, aff_expr) in MOI.NormInfinityCone(num_stocks + 1))
    end

    return model
end

function test_extra(inst::PortfolioJuMP, model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

# @testset "PortfolioJuMP" for inst in example_tests(PortfolioJuMP{Float64}, MinimalInstances()) test(inst...) end

return PortfolioJuMP
