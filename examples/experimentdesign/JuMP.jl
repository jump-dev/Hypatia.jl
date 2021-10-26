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

    # vectorized information matrix
    rt2 = sqrt(T(2))
    Q_vec = [JuMP.@expression(model, (i == j ? one(T) : rt2) *
        sum(V[i, k] * x[k] * V[j, k] for k in 1:k)) for i in 1:d for j in 1:i]

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(inst.ext, d, vcat(1.0 * epi, Q_vec), model)

    # save for use in tests
    model.ext[:V] = V
    model.ext[:x] = x

    return model
end

function test_extra(inst::ExperimentDesignJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective
    tol = eps(T)^0.2
    V = model.ext[:V]
    x_opt = JuMP.value.(model.ext[:x])
    λ = eigvals(Symmetric(V * Diagonal(x_opt) * V', :U))
    @test minimum(λ) >= -tol
    obj_result = get_val(pos_only(λ), inst.ext)
    @test JuMP.objective_value(model) ≈ obj_result atol=tol rtol=tol

    # inv hess prod oracle vs explicit hess factorization and solve
    # timing and numerics checks at final point already loaded in cone
    @assert stat == MOI.OPTIMAL # has to be feasible point
    println("\noracle timings results")
    d = inst.d
    println("d = $d")

    # get the EpiPerSepSpectral cone
    cone = JuMP.backend(model).optimizer.model.cones[end]
    @assert cone isa Hypatia.Cones.EpiPerSepSpectral
    g = copy(Hypatia.Cones.grad(cone))
    nu = Hypatia.Cones.get_nu(cone)

    println("\ninv hess prod oracle")
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
        cone.inv_hess_aux_updated = cone.hess_fact_updated = false
    stats1 = @timed begin
        Hig = Hypatia.Cones.inv_hess_prod!(cone.vec1, g, cone)
    end
    LHviol1 = abs(1 - dot(Hig, g) / nu)
    println("LH viol:\n$LHviol1")
    println("time:\n$(stats1.time)")
    println("bytes:\n$(stats1.bytes)")

    println("\nexplicit hess factorization and solve")
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
        cone.inv_hess_aux_updated = cone.hess_fact_updated = false
    println("allocate")
    @time begin
        Hypatia.Cones.alloc_hess!(cone)
        cone.hess_fact_mat = zero(cone.hess)
    end
    println("compute Hessian, factorize, and solve")
    stats2 = @timed begin
        fact_ok = Hypatia.Cones.update_hess_fact(cone)
        if fact_ok
            Hig = ldiv!(cone.vec1, cone.hess_fact, g)
        end
    end
    if fact_ok
        LHviol2 = abs(1 - dot(Hig, g) / nu)
        println("LH viol:\n$LHviol2")
        println("time:\n$(stats2.time)")
        println("bytes:\n$(stats2.bytes)")
    else
        println("hess fact failed")
        LHviol2 = Inf
    end
    fact_name = nameof(typeof(cone.hess_fact))
    println(fact_name)
    @assert fact_name in (:Cholesky, :BunchKaufman)
    fact_str = (fact_name == :Cholesky ? "Ch" : "BK")

    # print output line for table
    println()
    @printf("& %d & %.2f & %.2f & %.2f & %.2f & %s \\\\", d,
        log10(stats1.time), log10(LHviol1), log10(stats2.time), log10(LHviol2),
        fact_str)
    println("\n")
        return
end
