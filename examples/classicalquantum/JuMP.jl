#=
compute the capacity of a classical-quantum channel
adapted from https://github.com/hfawzi/cvxquad/blob/master/examples/cq_channel_capacity.m
and listing 1 in "Efficient optimization of the quantum relative entropy" by H. Fawzi and O. Fawzi
=#

struct ClassicalQuantum{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    complex::Bool
    use_EF::Bool
end

function build(inst::ClassicalQuantum{T}) where {T <: Float64}
    n = inst.n
    @assert !(inst.complex && inst.use_EF)
    rt2 = sqrt(T(2))
    R = (inst.complex ? Complex{T} : T)
    function hermtr1()
        ρ = randn(R, n, n)
        ρ = ρ * ρ'
        ρ ./= tr(ρ)
        return Hermitian(ρ)
    end
    ρs = [hermtr1() for _ in 1:n]
    Hs = [dot(ρ, log(ρ)) for ρ in ρs]

    model = JuMP.Model()
    JuMP.@variable(model, prob[1:n] >= 0)
    JuMP.@constraint(model, sum(prob) == 1)
    JuMP.@variable(model, qe_epi)
    JuMP.@objective(model, Max, -qe_epi + dot(prob, Hs))

    entr_sum_vec = zeros(JuMP.AffExpr, Cones.svec_length(R, n))
    ρ_vec = zeros(T, length(entr_sum_vec))
    for (ρ, p) in zip(ρs, prob)
        Cones.smat_to_svec!(ρ_vec, ρ, rt2)
        entr_sum_vec += p * ρ_vec
    end

    aff = vcat(qe_epi, 1, entr_sum_vec)
    if inst.use_EF
        add_spectral(MatNegEntropyEigOrd(), n, aff, model)
    else
        JuMP.@constraint(model, aff in Hypatia.EpiPerSepSpectralCone{Float64}(
            Cones.NegEntropySSF(), Cones.MatrixCSqr{T, R}, n))
    end

    return model
end
