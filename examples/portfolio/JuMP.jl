#=
portfolio rebalancing problem
=#

struct PortfolioJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_stocks::Int
    epipernormeucl_constr::Bool # add an L2 ball constraint, else don't add
    epinorminf_constrs::Bool # add L1 and Linfty ball constraints, else don't add
end

function build(inst::PortfolioJuMP{T}) where {T <: Float64}
    num_stocks = inst.num_stocks
    returns = rand(num_stocks)
    sigma_half = randn(num_stocks, num_stocks)
    x = randn(num_stocks)
    x ./= norm(x)
    gamma = sum(abs, sigma_half * x) / sqrt(num_stocks)

    model = JuMP.Model()
    JuMP.@variable(model, invest[1:num_stocks])
    JuMP.@objective(model, Max, dot(returns, invest))
    JuMP.@constraint(model, sum(invest) == 0)

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
