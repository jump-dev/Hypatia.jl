#=
portfolio rebalancing problem
=#

struct PortfolioJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_stocks::Int
    epinormeucl_constr::Bool # add an L2 ball constraint
    epinorminf_constrs::Bool # add L1 and Linfty ball constraints
end

function build(inst::PortfolioJuMP{T}) where {T <: Float64}
    @assert xor(inst.epinormeucl_constr, inst.epinorminf_constrs)
    num_stocks = inst.num_stocks

    returns = rand(num_stocks)
    sigma_half = randn(num_stocks, num_stocks)
    x = randn(num_stocks)
    gamma = sum(abs, sigma_half * x) / norm(x)
    A = randn(div(num_stocks, 2), num_stocks)

    model = JuMP.Model()
    JuMP.@variable(model, invest[1:num_stocks])
    JuMP.@objective(model, Max, dot(returns, invest))
    JuMP.@constraint(model, sum(invest) == 0)
    JuMP.@constraint(model, A * invest .== 0)

    if inst.epinormeucl_constr
        JuMP.@constraint(model, vcat(gamma / sqrt(num_stocks),
            sigma_half * invest) in JuMP.SecondOrderCone())
    end
    if inst.epinorminf_constrs
        JuMP.@constraint(model, vcat(1, invest) in
            MOI.NormInfinityCone(1 + num_stocks))
        JuMP.@constraint(model, vcat(gamma, sigma_half * invest) in
            MOI.NormOneCone(1 + num_stocks))
    end

    return model
end
