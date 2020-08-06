#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

robust geometric programming problem
given a convex set C in R_+^k (described by conic constraints) and a matrix B in R^{k, n}, calculate
    f(C, B) = sup_{c in C} (inf_{x in R^n, z in R_+^k} c'*z : B_i*x <= log(z_i), i = 1..k)
note the inner problem is an unconstrained GP, and C specifies coefficient uncertainty
for more details, see section 4.4 of:
"Relative entropy optimization and its applications" (2017) by Chandrasekaran & Shah
the authors show that:
    f(C, B) = -inf_{c in C, v in R_+^k} (d(v, e*c) : B_j'*v = 0, j = 1..n)
where e = exp(1) and d(a, b) = sum_i a_i*log(a_i/b_i) is the relative entropy of a and b
=#

struct BallsJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
end

function build(inst::BallsJuMP{T}) where {T <: Float64} # TODO generic reals
    n = inst.n
    model = JuMP.Model()
    # JuMP.@variable(model, t)
    JuMP.@variable(model, ray[1:n])
    d = div(n - 1, 2)
    (u, v, w) = (ray[1], ray[2:(d + 1)], ray[(d + 2):end])
    JuMP.@objective(model, Min, u)

    JuMP.@constraint(model, entr, vcat(u, ℯ * v, w) in MOI.RelativeEntropyCone(n))
    # JuMP.@constraint(model, vcat(t, ray) in JuMP.SecondOrderCone())
    # @show sqrt(d) / 2
    JuMP.@constraint(model, vcat(20, 1 .- ray) in JuMP.SecondOrderCone()) # tends to bind

    return model
end

function test_extra(inst::BallsJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    # tval = JuMP.value(model[:t])
    rayval = JuMP.value.(model[:ray])
    # @show tval
    # @show rayval
    # @show norm(rayval)
    d = div(length(rayval) - 1, 2)
    @show sqrt(d) / 2 - sqrt(sum(abs2, rayval .- 1))
    v = rayval[2:(d + 1)]
    w = rayval[(d + 2):end]
    @show rayval[1] - sum(w .* log.(w ./ (ℯ * v)))
    # @show JuMP.value(model[:entr])
    return
end

tols6 = (tol_feas = 1e-6, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, reduce = true, rescale = true)
instances[BallsJuMP]["fastnat"] = [
    # ((5,), nothing, tols6),
    ((81,), nothing, tols6),
    # ((101,), nothing, tols6),
    # ((501,), nothing, tols6),
    ]
instances[BallsJuMP]["fastext"] = [
    # ((5,), ClassicConeOptimizer, tols6),
    ((81,), ClassicConeOptimizer, tols6),
    # ((101,), ClassicConeOptimizer, tols6),
    # ((501,), ClassicConeOptimizer, tols6),
    ]
