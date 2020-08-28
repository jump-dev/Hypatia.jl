#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhedron
defined with l_1, l_infty, or l_2 ball constraints (different to native.jl)
=#

struct MaxVolumeJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    epipernormeucl_constr::Bool # add an L2 ball constraint, else don't add
    epinorminf_constrs::Bool # add L1 and Linfty ball constraints, else don't add
end

function build(inst::MaxVolumeJuMP{T}) where {T <: Float64} # TODO generic reals
    n = inst.n
    A = randn(n, n)
    # ensure there will be a feasible solution
    x = randn(n)
    gamma = norm(A * x) / sqrt(n)

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@variable(model, end_pts[1:n])
    JuMP.@objective(model, Max, t)
    JuMP.@constraint(model, vcat(t, end_pts) in MOI.GeometricMeanCone(n + 1))

    if inst.epipernormeucl_constr
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in JuMP.SecondOrderCone())
    end
    if inst.epinorminf_constrs
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in MOI.NormInfinityCone(n + 1))
        JuMP.@constraint(model, vcat(sqrt(n) * gamma, A * end_pts) in MOI.NormOneCone(n + 1))
    end

    return model
end
