#=
compute the capacity of a classical-quantum channel
adapted from https://github.com/hfawzi/cvxquad/blob/master/examples/cq_channel_capacity.m
and listing 1 in "Efficient optimization of the quantum relative entropy" by H. Fawzi and O. Fawzi
=#

struct ClassicalQuantum{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    complex::Bool
    use_EF::Bool
end

function build(inst::ClassicalQuantum{T}) where {T <: Float64}
    d = inst.d
    ext = (inst.use_EF ? MatNegEntropyEigOrd() : MatNegEntropy())
    rt2 = sqrt(T(2))
    R = (inst.complex ? Complex{T} : T)

    function hermtr1()
        P = randn(R, d, d)
        P = Hermitian(P * P', :U)
        P.data ./= tr(P)
        return Hermitian(P)
    end
    Ps = [hermtr1() for _ in 1:d]
    Hs = [real(dot(P, log(P))) for P in Ps]

    model = JuMP.Model()
    JuMP.@variable(model, ρ[1:d] >= 0)
    JuMP.@constraint(model, sum(ρ) == 1)
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Max, -epi + dot(ρ, Hs))

    vec_dim = Cones.svec_length(R, d)
    entr = JuMP.AffExpr.(zeros(vec_dim))
    P_vec = zeros(T, vec_dim)
    for (Pi, ρi) in zip(Ps, ρ)
        Cones.smat_to_svec!(P_vec, Pi, rt2)
        JuMP.add_to_expression!.(entr, ρi, P_vec)
    end
    add_homog_spectral(ext, d, vcat(epi, entr), model)

    # save for use in tests
    model.ext[:epi] = epi
    model.ext[:entr] = entr

    return model
end

function test_extra(inst::ClassicalQuantum{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check constraint
    tol = eps(T)^0.2
    epi_opt = JuMP.value(model.ext[:epi])
    Entr_opt = zeros(inst.complex ? Complex{T} : T, inst.d, inst.d)
    Cones.svec_to_smat!(Entr_opt, JuMP.value.(model.ext[:entr]), sqrt(T(2)))
    λ = eigvals(Hermitian(Entr_opt, :U))
    @test minimum(λ) >= -tol
    qe_result = get_val(pos_only(λ), MatNegEntropy())
    @test epi_opt ≈ qe_result atol=tol rtol=tol
    return
end
