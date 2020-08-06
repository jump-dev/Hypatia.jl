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
    B
end

function RobustGeomProgJuMP{T}(n, k) where {T <: Real}
    B = randn(T, k, n)
    # B = ones(T, k, n)
    return RobustGeomProgJuMP{T}(n, k, B)
end

function build(inst::RobustGeomProgJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, k) = (inst.n, inst.k)
    @assert n < k # want some degrees of freedom for v
    # B = randn(T, k, n) # GP powers matrix
    B = inst.B
    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@variable(model, c[1:k])
    JuMP.@variable(model, v[1:k])
    JuMP.@objective(model, Min, t)
    JuMP.@constraint(model, vcat(t, ℯ * c, v) in MOI.RelativeEntropyCone(1 + 2k))
    # JuMP.@constraint(model, B' * v .== 0)

    # use bounded convex set C of R_+^k excluding origin (note that the entropy constraint already forces c >= 0)
    # JuMP.@constraint(model, vcat(sqrt(k) / 2, 1 .- c) in MOI.NormOneCone(1 + k))
    # @show sqrt(k) / 2
    # JuMP.@constraint(model, vcat(sqrt(k) / 2, 1 .- vcat(t, c, v)) in JuMP.SecondOrderCone())
    JuMP.@constraint(model, vcat(sqrt(k) / 2, 1 .- c) in JuMP.SecondOrderCone())

    return model
end

# function build(inst::RobustGeomProgJuMP{T}) where {T <: Float64} # TODO generic reals
#     (n, k) = (inst.n, inst.k)
#     @assert n < k # want some degrees of freedom for v
#     model = JuMP.Model()
#     JuMP.@variables(model, begin
#         a
#         x2
#         y2[1:k]
#         z2[1:k]
#     end)
#     JuMP.@objective(model, Min, sqrt(k) / 2 * a + x2 + sum(y2 + z2))
#     JuMP.@constraint(model, vcat(-1 + x2, y2 / ℯ, z2) in Hypatia.EpiSumPerEntropyCone{Float64}(1 + 2k, false))
#     JuMP.@constraint(model, vcat(a, x2, y2, z2) in JuMP.SecondOrderCone())
#     return model
# end

function test_extra(inst::RobustGeomProgJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    # tval = JuMP.value(model[:t])
    # cval = JuMP.value.(model[:c])
    # vval = JuMP.value.(model[:v])
    # @show norm(1 .- cval) - sqrt(inst.k) / 2
    # @show tval
    # @show cval
    # @show vval
    #
    # @show tval - sum(vval .* log.(abs.(vval ./ (ℯ * cval))))
    # @show norm(inst.B' * vval)
    return
end

tols6 = (tol_feas = 1e-6, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, reduce = true, rescale = true)
instances[RobustGeomProgJuMP]["minimal"] = [
    ((2, 3),),
    ((2, 3), ClassicConeOptimizer),
    ]
instances[RobustGeomProgJuMP]["fastnat"] = [
    # ((5, 10), nothing, tols6),
    # # ((5, 10), ClassicConeOptimizer, tols6),
    # ((10, 20), nothing, tols6),
    # # ((10, 20), ClassicConeOptimizer, tols6),
    # ((20, 40), nothing, tols6),
    # ((20, 40), ClassicConeOptimizer, tols6),
    # ((40, 80), nothing, tols6),
    # # ((40, 80), ClassicConeOptimizer, tols6),
    # ((100, 150), nothing, tols6),
    # # ((100, 150), ClassicConeOptimizer, tols6),
    ]
instances[RobustGeomProgJuMP]["fastext"] = [
    # ((5, 10), nothing, tols6),
    # ((5, 10), ClassicConeOptimizer, tols6),
    # # ((10, 20), nothing, tols6),
    # ((10, 20), ClassicConeOptimizer, tols6),
    # ((20, 40), nothing, tols6),
    # ((20, 40), ClassicConeOptimizer, tols6),
    # # ((40, 80), nothing, tols6),
    ((40, 80), ClassicConeOptimizer, tols6),
    # # ((100, 150), nothing, tols6),
    # ((100, 150), ClassicConeOptimizer, tols6),
    ]
instances[RobustGeomProgJuMP]["slow"] = [
    ((40, 80), ClassicConeOptimizer, tols6),
    ((100, 200), nothing, tols6),
    ]
