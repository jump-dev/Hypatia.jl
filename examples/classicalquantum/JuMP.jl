#=
compute the capacity of a classical-quantum channel
adapted from https://github.com/hfawzi/cvxquad/blob/master/examples/cq_channel_capacity.m
and listing 1 in "Efficient optimization of the quantum relative entropy" by H. Fawzi and O. Fawzi
=#

import QuantumInformation: ptrace

struct ClassicalQuantum{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
end

function build(inst::ClassicalQuantum{T}) where {T <: Float64}
    n = inst.n
    sn = div(n * (n + 1), 2)
    ρs = Matrix[]
    Hs = Float64[]
    for _ in 1:n
        # TODO ρ = randn(Complex{T}, n, n)
        ρ = randn(T, n, n)
        ρ = ρ * ρ'
        ρ /= tr(ρ)
        push!(ρs, ρ)
        push!(Hs, dot(ρ, log(ρ)))
    end

    model = JuMP.Model()
    JuMP.@variable(model, prob[1:n] >= 0)
    JuMP.@variable(model, qe_epi)
    entr_sum = sum(ρ * p for (ρ, p) in zip(ρs, prob))
    JuMP.@constraint(model, vcat(qe_epi, 1, Cones.smat_to_svec!(zeros(JuMP.AffExpr, sn), entr_sum, sqrt(2))) in Hypatia.EpiPerTraceEntropyTriCone{Float64}(2 + sn))
    JuMP.@constraint(model, sum(prob) == 1)
    JuMP.@objective(model, Max, -qe_epi + dot(prob, Hs))

    return model
end
