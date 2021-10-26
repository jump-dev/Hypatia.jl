#=
given a random X variable taking values in the finite set {α₁,...,αₙ}, compute
the distribution minimizing a given convex spectral function over all distributions
satisfying some prior information (expressed using equality constraints)

adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.2

p ∈ ℝᵈ is the probability variable
minimize    f(p)            (note: enforces p ≥ 0)
subject to  Σᵢ pᵢ = d       (probability distribution, scaled by d)
            A p = b         (prior info)
where f is a convex spectral function
=#

struct NonparametricDistrJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    ext::VecSpecExt # formulation specifier
end

function build(inst::NonparametricDistrJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 2
    @assert is_domain_pos(inst.ext) # domain must be positive
    p0 = rand(T, d)
    p0 .*= d / sum(p0)

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d])
    JuMP.@constraint(model, sum(p) == d)

    # linear prior constraints
    A = randn(T, round(Int, d / 2), d)
    b = A * p0
    JuMP.@constraint(model, A * p .== b)

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(inst.ext, d, vcat(1.0 * epi, p), model)

    # save for use in tests
    model.ext[:p_var] = p

    return model
end

function test_extra(inst::NonparametricDistrJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective and constraints
    tol = eps(T)^0.2
    p_opt = JuMP.value.(model.ext[:p_var])
    @test sum(p_opt) ≈ inst.d atol=tol rtol=tol
    @test minimum(p_opt) >= -tol
    p_opt = pos_only(p_opt)
    obj_result = get_val(p_opt, inst.ext)
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
