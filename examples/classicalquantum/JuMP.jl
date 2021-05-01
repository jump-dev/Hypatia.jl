#=
compute the capacity of a classical-quantum channel
adapted from https://github.com/hfawzi/cvxquad/blob/master/examples/cq_channel_capacity.m
and listing 1 in "Efficient optimization of the quantum relative entropy" by H. Fawzi and O. Fawzi
=#

import QuantumInformation: ptrace

struct ClassicalQuantum{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    complex::Bool
end

function build(inst::ClassicalQuantum{T}) where {T <: Float64}
    n = inst.n
    R = (inst.complex ? Complex{T} : T)
    function hermtr1()
        ρ = randn(R, n, n)
        ρ = ρ * ρ'
        ρ ./= tr(ρ)
        return ρ
    end
    ρs = [hermtr1() for _ in 1:n]
    Hs = [dot(ρ, log(ρ)) for ρ in ρs]

    model = JuMP.Model()
    JuMP.@variable(model, prob[1:n] >= 0)
    JuMP.@variable(model, qe_epi)

    entr_sum = sum(ρ * p for (ρ, p) in zip(ρs, prob))
    sdn = div(n * (n + 1), 2)
    entr_sum_vec = Cones.smat_to_svec!(zeros(JuMP.AffExpr, sdn), entr_sum, sqrt(T(2)))
    cone = Hypatia.EpiPerTraceEntropyTriCone{T, R}(2 + sdn)
    JuMP.@constraint(model, vcat(qe_epi, 1, entr_sum_vec) in cone)
    JuMP.@constraint(model, sum(prob) == 1)
    JuMP.@objective(model, Max, -qe_epi + dot(prob, Hs))

    return model
end
