#=
calculate a maximum margin classifier for a support vector machine, when data can be perfectly separated
see lecture 4 https://gitlab.com/vanparys/15.084
=#
import Random

struct SVMJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    points::Matrix{T}
    labels::Vector{Int}
    separated::Bool # data can be separated and therefore problem is feasible, else infeasible
    use_primal::Bool # if true use a primal formulation, else use dual
end

function SVMJuMP{Float64}(num_points::Int, num_features::Int, separated::Bool, use_primal::Bool)
    @assert num_points >= 2
    labels = [Random.rand([-1, 1]) for _ in 1:num_points]
    points = rand(num_points, num_features)
    # dummy feature for offset
    points = hcat(points, ones(num_points))
    if separated
        points .+= labels
    else
        while all(labels .== 1) || all(labels .== -1)
            labels = [Random.rand([-1, 1]) for _ in 1:num_points]
        end
        # manually mix points together
        p1 = findfirst(isequal(1), labels)
        m1 = findfirst(isequal(-1), labels)
        while !isnothing(p1) && !isnothing(m1)
            n = findnext(isequal(-1), labels, m1 + 1)
            m2 = (isnothing(n) ? m1 : n)
            points[p1, :] = (points[m1, :] + points[m2, :]) / 2
            p1 = findnext(isequal(1), labels, p1 + 1)
            m1 = findnext(isequal(-1), labels, m2 + 1)
        end

        # p = findfirst(isequal(1), labels)
        # m1 = findfirst(isequal(-1), labels)
        # m2 = findlast(isequal(-1), labels)
        # points[p, :] = (points[m1, :] + points[m2, :]) / 2
    end
    return SVMJuMP{Float64}(points, labels, separated, use_primal)
end

function build(inst::SVMJuMP{T}) where {T <: Float64}
    (points, labels) = (inst.points, inst.labels)
    (num_points, num_features) = size(points)
    model = JuMP.Model()
    if inst.use_primal
        JuMP.@variable(model, classifier[1:(num_features)])
        JuMP.@variable(model, t)
        JuMP.@objective(model, Min, t)
        JuMP.@constraint(model, vcat(t, classifier) in JuMP.SecondOrderCone())
        JuMP.@constraint(model, labels .* (points * classifier) .>= 1)
    else
    end
    return model
end

function test_extra(inst::SVMJuMP{T}, model::JuMP.Model) where T
    if inst.separated
        @test JuMP.termination_status(model) == MOI.OPTIMAL
    else
        @test JuMP.termination_status(model) == (inst.use_primal ? MOI.INFEASIBLE : MOI.DUAL_INFEASIBLE)
    end
    return
end
