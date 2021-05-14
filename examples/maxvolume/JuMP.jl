#=
find maximum volume hypercube with edges parallel to the axes inside a polyhedron
or an ellipsoid defined with l_1, l_infty, or l_2 ball constraints
=#

struct MaxVolumeJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    epinormeucl_constr::Bool # add an L2 ball constraint
    epinorminf_constrs::Bool # add L1 and Linfty ball constraints
end

function build(inst::MaxVolumeJuMP{T}) where {T <: Float64}
    @assert xor(inst.epinormeucl_constr, inst.epinorminf_constrs)
    n = inst.n

    A = randn(n, n)
    # ensure there will be a feasible solution
    x = randn(n)
    A = A * A' + 10I
    gamma = norm(A * x) / sqrt(n)

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@variable(model, x[1:n])
    JuMP.@objective(model, Max, t)
    JuMP.@constraint(model, vcat(t, x) in MOI.GeometricMeanCone(n + 1))

    if inst.epinormeucl_constr
        JuMP.@constraint(model, vcat(gamma, A * x) in JuMP.SecondOrderCone())
    end
    if inst.epinorminf_constrs
        JuMP.@constraint(model, vcat(gamma, A * x) in MOI.NormInfinityCone(n + 1))
        JuMP.@constraint(model, vcat(sqrt(n) * gamma, A * x) in
            MOI.NormOneCone(n + 1))
    end

    return model
end
