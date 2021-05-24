#=
maximize likelihood of d observations at discrete points appearing with random
frequencies, subject to probability vector not being too far from a uniform
prior
=#

struct DiscreteMaxLikelihood{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
end

function build(inst::DiscreteMaxLikelihood{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 2
    freq = rand(1:(2 * d), d)

    model = JuMP.Model()
    JuMP.@variables(model, begin
        p[1:d] >= 0
        prodp
    end)
    JuMP.@constraints(model, begin
        vcat(prodp, p) in Hypatia.HypoPowerMeanCone{T}(freq / sum(freq))
        vcat(inv(d), inv(d), p) in Hypatia.EpiPerSepSpectralCone{T}(
            Cones.NegEntropySSF(), Cones.VectorCSqr{T}, d)
        sum(p) == 1
    end)
    JuMP.@objective(model, Max, prodp)

    return model
end
