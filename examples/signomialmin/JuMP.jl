#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

signomial minimization problem
see "Signomial and Polynomial Optimization via Relative Entropy and Partial Dualization" (2019) by Murray, Chandrasekaran, & Wierman

a signomial f = Sig(A, c) of variables
    x in R^n
with parameters
    c in R^m
    A in R^{m, n}
takes values
    f(x) = sum_{i in [m]} c_i exp(A_{i, :} * x)

for signomials f, (g_p)_{p in 1:q}, the signomial minimization problem is
    inf_{x in R^n} f(x) : g_p(x) >= 0, p in 1:q

the convex SAGE relaxation problem is
    max_{γ in R, μ in R_+^q} γ : c - γe₁ - sum_{p in 1:q} μ_p * g_p in C_SAGE(A)

a vector d in R^m belongs to SAGE cone C_SAGE(A) if
    exists C in R^{m, m} :
    d = sum_{k in [m]} C_{k, :}
    C_{k, :} in C_AGE(A, k)
which is equivalent to the feasibility problem over C in R^{m, m} and V in R^{m, m-1}
    d = sum_{k in [m]} C_{k, :}
    for k in [m]:
        (A_{\k, :} - [1,...,1] * A_{k, :}')' * V_{k, :} == [0,...,0]
        [C_{k, k} + sum(V_{k, :}), C_{k, \k}, V_{k, :}] in RelEntr(1 + 2(m - 1))
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
include(joinpath(@__DIR__, "data.jl"))

struct SignomialMinJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    fc::Vector
    fA::AbstractMatrix
    gc::Vector
    gA::Vector
    obj_ub::Real
end
SignomialMinJuMP{Float64}(sig_name::Symbol) = SignomialMinJuMP{Float64}(signomialmin_data[sig_name]...)
SignomialMinJuMP{Float64}(m::Int, n::Int) = SignomialMinJuMP{Float64}(signomialmin_random(m, n)...)

example_tests(::Type{SignomialMinJuMP{Float64}}, ::MinimalInstances) = [
    ((:CS16ex12,),),
    ((2, 2),),
    ((2, 2), ClassicConeOptimizer),
    ]
example_tests(::Type{SignomialMinJuMP{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-5, tol_rel_opt = 1e-5, tol_abs_opt = 1e-5)
    # relaxed_options = (tol_feas = 1e-5, tol_rel_opt = 1e-5, tol_abs_opt = 1e-5)
    # ext = nothing
    ext = ClassicConeOptimizer
    return [
    ((:motzkin2,), ext, options),
    # # ((:motzkin2,), ClassicConeOptimizer, options),
    ((:motzkin3,), ext, options),
    ((:CS16ex8_13,), ext, options),
    ((:CS16ex8_14,), ext, options),
    ((:CS16ex18,), ext, options),
    ((:CS16ex12,), ext, options),
    ((:CS16ex13,), ext, options),
    ((:MCW19ex1_mod,), ext, options),
    ((:MCW19ex8,), ext, options),
    # # ((:MCW19ex8,), ClassicConeOptimizer, options),
    ((3, 2), ext, options),
    # # ((3, 2), ClassicConeOptimizer, options),
    ((6, 6), ext, options),
    ((20, 3), ext, options),
    ((10, 6), ext, options),
    ((9, 9), ext, options),
    ((15, 4), ext, options),
    # ((15, 5), ext, options),
    # ((25, 3), ext, options),
    # ((20, 3), ClassicConeOptimizer, options),
    ]
end
example_tests(::Type{SignomialMinJuMP{Float64}}, ::SlowInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    relaxed_options = (tol_feas = 1e-5, tol_rel_opt = 1e-4, tol_abs_opt = 1e-4)
    return [
    ((10, 10), nothing, options),
    ((10, 10), ClassicConeOptimizer, options),
    ((20, 6), nothing, options),
    ((40, 3), nothing, options),
    ]
end

function build(inst::SignomialMinJuMP{T}) where {T <: Float64} # TODO generic reals
    (fc, fA, gc, gA) = (inst.fc, inst.fA, inst.gc, inst.gA)
    (fm, n) = size(fA)
    @assert length(fc) == fm
    q = length(gc)
    @assert length(gA) == q
    @assert all(size(gA_p, 2) == n for gA_p in gA)
    @assert all(size(gA_p, 1) == length(gc_p) for (gc_p, gA_p) in zip(gc, gA))

    # find unique terms
    unique_terms = Dict{Vector, Tuple{Vector{Int}, Vector{Tuple{Int, Int}}}}()
    for k in 1:size(fA, 1)
        row_k = fA[k, :]
        if row_k in keys(unique_terms)
            push!(unique_terms[row_k][1], k)
        else
            unique_terms[row_k] = (Int[k], Tuple{Int, Int}[])
        end
    end
    for (p, gA_p) in enumerate(gA)
        for k in 1:size(gA_p, 1)
            p_row_k = gA_p[k, :]
            if p_row_k in keys(unique_terms)
                push!(unique_terms[p_row_k][2], (p, k))
            else
                unique_terms[p_row_k] = (Int[], Tuple{Int, Int}[(p, k)])
            end
        end
    end

    A = vcat((row_k' for row_k in keys(unique_terms) if !iszero(norm(row_k, Inf)))...)
    A = vcat(zeros(1, n), A)
    m = size(A, 1)

    # TODO sparse?
    fc_new = zeros(m)
    gc_new = [zeros(m) for p in 1:q]
    for k in 1:m
        (ks, pks) = unique_terms[A[k, :]]
        for k2 in ks
            fc_new[k] += fc[k2]
        end
        for (p, k2) in pks
            gc_new[p][k] += gc[p][k2]
        end
    end
    fc = fc_new
    gc = gc_new

    model = JuMP.Model()

    use_primal = true # TODO option
    # use_primal = false # TODO option

    if use_primal
        JuMP.@variable(model, γ)
        JuMP.@objective(model, Max, γ)
        JuMP.@variable(model, μ[1:q] >= 0)

        d = zeros(JuMP.AffExpr, m)
        d[1] -= γ
        d += fc
        for p in 1:q
            d -= μ[p] * gc[p]
        end

        # setup SAGE constraints
        notk = [[l for l in 1:m if l != k] for k in 1:m]
        JuMP.@variable(model, C[1:m, 1:m])
        JuMP.@variable(model, V[1:m, 1:(m - 1)])
        JuMP.@constraint(model, [k in 1:m], d[k] == sum(C[:, k]))
        JuMP.@constraint(model, [k in 1:m, i in 1:n], dot(A[notk[k], i] .- A[k, i], V[k, :]) == 0)
        JuMP.@constraint(model, [k in 1:m], vcat(C[k, k] + sum(V[k, :]), C[k, notk[k]], V[k, :]) in MOI.RelativeEntropyCone(2m - 1))
    else
        # TODO dual formulation
        # primal is: min_{γ in R, μ in R^q} -γ : c - γe₁ - sum_{p in 1:q} μ_p * g_p in C_SAGE(A), μ in R_+^q
        # dual is: max_{π in R^m, ψ in R^q} -c'π : -1 + π_1 = 0, g_1'π - ψ_1 = ... = g_q'π - ψ_q = 0, π in C_SAGE^*(A), ψ in R_+^q

        # JuMP.@variable(model, π[1:m])
        # JuMP.@constraint(model, π[1] == 1)
        π = vcat(1, JuMP.@variable(model, [1:(m - 1)]))

        JuMP.@objective(model, Min, dot(fc, π))

        # JuMP.@variable(model, ψ[1:q] >= 0)
        # JuMP.@constraint(model, [p in 1:q], dot(gc[p], π) - ψ[p] == 0)
        JuMP.@constraint(model, [p in 1:q], dot(gc[p], π) >= 0) # TODO can include this in the SAGE cone definition (changes it a bit though - what's the analogy to dual WSOS?)

        # setup dual SAGE constraints: π in C_SAGE^*(A)
        JuMP.@variable(model, τ[1:m, 1:n])
        for k in 1:m, l in 1:m
            k == l && continue
            # π_k*log(π_k/π_l) <= (A_k - A_l)'*τ_(k,:)
            epi = dot(τ[k, :], A[k, :] - A[l, :])
            JuMP.@constraint(model, [epi, π[l], π[k]] in MOI.RelativeEntropyCone(3))
        end
        # for k in 1:m
        # notk = [l for l in 1:m if l != k]
        #     vec_trans = zeros(JuMP.AffExpr, 2m - 1)
        #     vec_trans[1] = π[k]
        #     vec_trans[2:2:end] = π[notk]
        #     τk = JuMP.@variable(model, [1:n])
        #     vec_trans[3:2:end] = [dot(τk, A[k, :] - A[l, :]) - π[k] for l in notk]
        #     JuMP.@constraint(model, vec_trans in Hypatia.EpiSumPerEntropyCone{Float64}(2m - 1, true)) # use higher dim RE dual cone
        # end
    end

    return model
end

function test_extra(inst::SignomialMinJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    if JuMP.termination_status(model) == MOI.OPTIMAL && !isnan(inst.obj_ub)
        # check objective value is correct
        tol = eps(T)^0.2
        @test JuMP.objective_value(model) <= inst.obj_ub + tol
    end
end

return SignomialMinJuMP
