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

struct RobustGeomProgJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    k::Int
end

function build(inst::RobustGeomProgJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, k) = (inst.n, inst.k)
    @assert n < k # want some degrees of freedom for v
    B = randn(T, k, n) # GP powers matrix
    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)

    JuMP.@variable(model, c[1:k])
    JuMP.@variable(model, v[1:k])
    JuMP.@constraint(model, vcat(t, ℯ * c, v) in MOI.RelativeEntropyCone(1 + 2k))
    JuMP.@constraint(model, B' * v .== 0)

    # use bounded convex set C of R_+^k excluding origin (note that the entropy constraint already forces c >= 0)
    JuMP.@constraint(model, vcat(sqrt(k) / 2, 1 .- c) in MOI.NormOneCone(1 + k))

    return model
end

instances[RobustGeomProgJuMP]["minimal"] = [
    ((2, 3),),
    ((2, 3), ClassicConeOptimizer),
    ]
instances[RobustGeomProgJuMP]["fast"] = [
    ((5, 10),),
    ((5, 10), ClassicConeOptimizer),
    ((10, 20),),
    ((10, 20), ClassicConeOptimizer),
    ((20, 40),),
    ((20, 40), ClassicConeOptimizer),
    ((40, 80),),
    ((40, 80), ClassicConeOptimizer),
    ((100, 150),),
    ((100, 150), ClassicConeOptimizer),
    ]
instances[RobustGeomProgJuMP]["slow"] = [
    ((40, 80), ClassicConeOptimizer),
    ((100, 200),),
    ]
