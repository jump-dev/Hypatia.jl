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
    freq /= sum(freq)

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d])
    JuMP.@variable(model, hypo)
    JuMP.@objective(model, Max, hypo)
    JuMP.@constraint(model, sum(p) == 1)

    JuMP.@constraint(model, vcat(hypo, p) in Hypatia.HypoPowerMeanCone{T}(freq))

    form = (inst.use_EF ? VecNegEntropyEF : VecNegEntropy)
    add_spectral(form(), d, vcat(inv(d), inv(d), p), model)

    # save for use in tests
    model.ext[:freq] = freq
    model.ext[:p_var] = p

    return model
end

function test_extra(inst::DiscreteMaxLikelihood{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective and constraints
    tol = eps(T)^0.2
    freq = model.ext[:freq]
    p_opt = JuMP.value.(model.ext[:p_var])
    @test sum(p_opt) ≈ 1 atol=tol rtol=tol
    @test minimum(p_opt) >= -tol
    p_opt = pos_only(p_opt)
    obj_result = exp(sum(f_i * log(p_i) for (f_i, p_i) in zip(freq, p_opt)))
    @test JuMP.objective_value(model) ≈ obj_result atol=tol rtol=tol
    return
end
