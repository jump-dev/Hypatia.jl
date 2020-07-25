#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

struct PortfolioJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_stocks::Int
    epipernormeucl_constr::Bool # add an L2 ball constraint, else don't add
    epinorminf_constrs::Bool # add L1 and Linfty ball constraints, else don't add
end

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

instances[PortfolioJuMP]["minimal"] = [
    ((3, true, false),),
    ((3, false, true),),
    ((3, false, true), ClassicConeOptimizer),
    ((3, true, true),),
    ]
instances[PortfolioJuMP]["fast"] = [
    ((10, true, false),),
    ((10, false, true),),
    ((10, false, true), ClassicConeOptimizer),
    ((10, true, true),),
    ((50, true, false),),
    ((50, false, true),),
    ((50, true, true),),
    ((400, true, false),),
    ((400, false, true),),
    ((400, true, true),),
    ((400, true, false),),
    ((400, false, true),),
    ((400, false, true), ClassicConeOptimizer),
    ((400, true, true),),
    ]
instances[PortfolioJuMP]["slow"] = [
    ((1000, true, false),),
    ((1000, false, true),),
    ((1000, false, true), ClassicConeOptimizer),
    ((1000, true, true),),
    ((3000, true, false),),
    ((3000, false, true),),
    ((3000, false, true), ClassicConeOptimizer),
    ((3000, true, true),),
    ]

# benchmark 1 instances
instances[PortfolioJuMP]["bench1"] = (
    ((num_stocks, false, true), ext)
    for num_stocks in vcat(10, 500:500:8000) # includes compile run
    for ext in (nothing, ClassicConeOptimizer)
    )
