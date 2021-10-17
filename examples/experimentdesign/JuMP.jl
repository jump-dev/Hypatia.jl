#=
choose the frequency of experiments to minimize a given convex spectral function
of the information matrix and satisfy an experiment budget constraint

adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.5

minimize    f(V × Diagonal(x) × V')
subject to  x ≥ 0
            e'x = k
where k = 2d, variable x ∈ ℝᵏ is the frequency of each experiment, k is the
number of experiments to run, the columns of V ∈ ℝ^(d × k) correspond to each
experiment, and f is a convex spectral function
=#

struct ExperimentDesignJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    ext::MatSpecExt # formulation specifier
end

function build(inst::ExperimentDesignJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 1
    @assert is_domain_pos(inst.ext)
    k = 2 * d

    V = randn(T, d, k)
    V .*= sqrt(d / sum(eigvals(Symmetric(V * V'))))

    model = JuMP.Model()
    JuMP.@variable(model, x[1:k] >= 0)
    JuMP.@constraint(model, sum(x) == k)

    vec_dim = Cones.svec_length(d)
    Q = V * diagm(x) * V' # information matrix
    Q_vec = zeros(JuMP.AffExpr, vec_dim)
    Cones.smat_to_svec!(Q_vec, Q, sqrt(T(2)))

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(inst.ext, d, vcat(1.0 * epi, Q_vec), model)

    # save for use in tests
    model.ext[:Q_var] = Q

    return model
end

function test_extra(inst::ExperimentDesignJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective
    tol = eps(T)^0.2
    Q_opt = JuMP.value.(model.ext[:Q_var])
    λ = eigvals(Symmetric(Q_opt, :U))
    @test minimum(λ) >= -tol
    obj_result = get_val(pos_only(λ), inst.ext)
    @test JuMP.objective_value(model) ≈ obj_result atol=tol rtol=tol
    return
end
