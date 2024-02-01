#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
nearest correlation matrix, in the quantum relative entropy sense, adapted from
https://github.com/hfawzi/cvxquad/blob/master/examples/nearest_correlation_matrix.m
=#

struct NearestCorrelationJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    side::Int
end

function build(inst::NearestCorrelationJuMP{T}) where {T <: Real}
    side = inst.side
    M = randn(T, side, side)
    M = M * M'
    vec_dim = Cones.svec_length(side)
    m_vec = Vector{T}(undef, vec_dim)
    Cones.smat_to_svec!(m_vec, M, sqrt(T(2)))

    model = JuMP.GenericModel{T}()
    JuMP.@variable(model, x_vec[1:vec_dim])
    X = Matrix{JuMP.GenericAffExpr{T, JuMP.GenericVariableRef{T}}}(undef, side, side)
    Cones.svec_to_smat!(X, one(T) * x_vec, sqrt(T(2)))
    JuMP.@constraint(model, diag(X) .== 1)

    JuMP.@variable(model, y)
    JuMP.@objective(model, Min, y)
    JuMP.@constraint(
        model,
        vcat(y, x_vec, m_vec) in Hypatia.EpiTrRelEntropyTriCone{T, T}(1 + 2 * vec_dim)
    )

    return model
end
