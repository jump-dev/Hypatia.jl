#=
maximize likelihood of d observations at discrete points appearing with random
frequencies, subject to probability vector not being too far from a uniform
prior
=#

struct DiscreteMaxLikelihood{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    use_EF::Bool
end

function build(inst::DiscreteMaxLikelihood{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 2
    freq = rand(1:(2 * d), d)

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d])
    JuMP.@variable(model, hypo)
    JuMP.@objective(model, Max, hypo)
    JuMP.@constraint(model, sum(p) == 1)

    # TODO extend
    JuMP.@constraint(model, vcat(hypo, p) in
        Hypatia.HypoPowerMeanCone{T}(freq / sum(freq)))

    ext = (inst.use_EF ? VecNegEntropyEF : VecNegEntropy)
    add_spectral(ext(), d, vcat(inv(d), inv(d), p), model)

    return model
end
