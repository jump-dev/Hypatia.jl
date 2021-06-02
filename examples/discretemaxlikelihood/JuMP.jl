#=
maximize likelihood of d observations at discrete points appearing with random
frequencies, subject to probability vector not being too far from a uniform
prior
=#

struct DiscreteMaxLikelihood{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    use_standard_cones::Bool
end

function build(inst::DiscreteMaxLikelihood{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 2
    freq = rand(1:(2 * d), d)

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d])
    JuMP.@variable(model, hypo)
    JuMP.@objective(model, Max, hypo)

    JuMP.@constraints(model, begin
        vcat(hypo, p) in Hypatia.HypoPowerMeanCone{T}(freq / sum(freq))
        sum(p) == 1
    end)

    add_sepspectral(Cones.NegEntropySSF(), Cones.VectorCSqr{T}, d,
        vcat(inv(d), inv(d), p), model, inst.use_standard_cones)

    return model
end
