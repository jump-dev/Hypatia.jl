#=
nearest correlation matrix, in the quantum relative entropy sense, adapted from
https://github.com/hfawzi/cvxquad/blob/master/examples/nearest_correlation_matrix.m
=#

struct NearestCorrelationJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    side::Int
end

function build(inst::NearestCorrelationJuMP{T}) where {T <: Float64}
    side = inst.side
    M = randn(T, side, side)
    M = M * M'
    vec_dim = Cones.svec_length(side)
    m_vec = zeros(T, vec_dim)
    Cones.smat_to_svec!(m_vec, M, sqrt(T(2)))

    model = JuMP.Model()
    JuMP.@variable(model, x_vec[1:vec_dim])
    X = zeros(JuMP.AffExpr, side, side)
    Cones.svec_to_smat!(X, one(T) * x_vec, sqrt(T(2)))
    JuMP.@constraint(model, diag(X) .== 1)

    JuMP.@variable(model, y)
    JuMP.@objective(model, Min, y)
    JuMP.@constraint(model, vcat(y, x_vec, m_vec) in
        Hypatia.EpiTrRelEntropyTriCone{T}(1 + 2 * vec_dim))

    return model
end
