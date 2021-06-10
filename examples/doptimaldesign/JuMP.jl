#=
D-optimal experiment design maximizes the determinant of the information matrix
adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.5
maximize    F(V × Diagonal(x) × V')
subject to  sum(np) == n
            0 .<= np .<= n_max
where np is a vector of variables representing the number of experiment to run,
and the columns of V are the vectors representing each experiment

if logdet_obj or rootdet_obj is true, F is the logdet or rootdet function
if geomean_obj is true, we use a formulation from
https://picos-api.gitlab.io/picos/optdes.html that finds an equivalent minimizer
=#

struct DOptimalDesignJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    q::Int
    p::Int
    n::Int
    n_max::Int
    logdet_obj::Bool # use formulation with logdet objective
    rootdet_obj::Bool # use formulation with rootdet objective
    geomean_obj::Bool # use formulation with geomean objective
end

function build(inst::DOptimalDesignJuMP{T}) where {T <: Float64}
    (q, p, n, n_max) = (inst.q, inst.p, inst.n, inst.n_max)
    @assert (p > q) && (n > q) && (n_max <= n)
    @assert +(inst.logdet_obj, inst.geomean_obj, inst.rootdet_obj) == 1
    V = randn(q, p)

    model = JuMP.Model()
    JuMP.@variable(model, np[1:p])
    JuMP.@constraint(model, sum(np) == n)
    mid = n_max / 2
    JuMP.@constraint(model, vcat(mid, np .- mid) in MOI.NormInfinityCone(p + 1))
    JuMP.@variable(model, hypo)
    JuMP.@objective(model, Max, hypo)

    if inst.logdet_obj || inst.rootdet_obj
        # information matrix lower triangle
        v1 = [JuMP.@expression(model,
            sum(V[i, k] * np[k] * V[j, k] for k in 1:p)) for i in 1:q for j in 1:i]
        if inst.logdet_obj
            JuMP.@constraint(model, vcat(hypo, 1, v1) in MOI.LogDetConeTriangle(q))
        else
            JuMP.@constraint(model, vcat(hypo, v1) in MOI.RootDetConeTriangle(q))
        end
    else
        # hypogeomean + epinormeucl formulation
        JuMP.@variable(model, L[i in 1:q, j in 1:i])
        JuMP.@variable(model, W[1:p, 1:q])
        VW = V * W
        JuMP.@constraints(model, begin
            [i in 1:q, j in 1:i], VW[i, j] == L[i, j]
            [i in 1:q, j in (i + 1):q], VW[i, j] == 0
            vcat(hypo, [L[i, i] for i in 1:q]) in MOI.GeometricMeanCone(1 + q)
            [i in 1:p], vcat(sqrt(q) * np[i], W[i, :]) in JuMP.SecondOrderCone()
        end)
    end

    return model
end
