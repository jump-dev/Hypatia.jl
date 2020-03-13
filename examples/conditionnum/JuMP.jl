#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

minimize the condition number of positive definite matrix M(x) = M_0 + sum_i x_i*M_i
subject to F(x) = F_0 + sum_i x_i*F_i in S_+

from section 3.2 "Linear Matrix Inequalities in System and Control Theory" by
S. Boyd, L. El Ghaoui, E. Feron, and V. Balakrishnan

original formulation:
min gamma
mu >= 0
F(x) in S_+
M(x) - mu*I in S_+
mu*gamma*I - M(x) in S_+

introduce nu = inv(mu), y = x/mu:
min gamma
nu >= 0
nu*F_0 + sum_i y_i*F_i in S_+
nu*M_0 + sum_i y_i*M_i - I in S_+
gamma*I - nu*M_0 - sum_i y_i*M_i in S_+

we make F_0 and M_0 positive definite to ensure existence of a feasible solution
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))

struct ConditionNumJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    side::Int
    len_y::Int
    use_linmatrixineq::Bool # use linmatrixineq cone, else PSD formulation
end

example_tests(::Type{ConditionNumJuMP{Float64}}, ::MinimalInstances) = [
    ((2, 2, true), false),
    ((2, 2, false), false),
    ]
example_tests(::Type{ConditionNumJuMP{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-5,)
    relaxed_options = (tol_feas = 1e-4, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((2, 3, true), false, options),
    ((2, 3, false), false, options),
    ((3, 2, true), false, options),
    ((3, 2, false), false, options),
    ((50, 15, true), false, options),
    ((50, 15, false), false, options),
    ((100, 10, false), false, relaxed_options),
    ((100, 40, false), false, relaxed_options),
    ]
end
example_tests(::Type{ConditionNumJuMP{Float64}}, ::SlowInstances) = begin
    options = (tol_feas = 1e-5,)
    return [
    ((100, 10, true), false, options),
    ((100, 40, true), false, options),
    ]
end

function build(inst::ConditionNumJuMP{T}) where {T <: Float64} # TODO generic reals
    (side, len_y) = (inst.side, inst.len_y)

    Mi = [zeros(side, side) for i in 1:len_y]
    for i in eachindex(Mi)
        Mi_half = randn(side)
        Mi[i] = Symmetric(Mi_half * Mi_half')
    end
    M0 = randn(side, side)
    M0 = Symmetric(M0 * M0')
    Fi = [Symmetric(randn(side, side)) for i in 1:len_y]
    F0 = randn(side, side)
    F0 = Symmetric(F0 * F0')
    # choose to make some F_i matrices pd so several feasible solutions exists
    pd_idxs = rand(1:len_y, max(1, div(len_y, 5)))
    for i in pd_idxs
        Fi[i] = Symmetric(Fi[i] * Fi[i]')
    end

    model = JuMP.Model()
    JuMP.@variables(model, begin
        gamma
        nu >= 0
        y[1:len_y]
    end)
    JuMP.@objective(model, Min, gamma)

    if inst.use_linmatrixineq
        JuMP.@constraints(model, begin
            vcat(nu, y) in Hypatia.LinMatrixIneqCone{Float64}([F0, Fi...])
            vcat(-1, nu, y) in Hypatia.LinMatrixIneqCone{Float64}([I, M0, Mi...])
            vcat(gamma, -nu, -y) in Hypatia.LinMatrixIneqCone{Float64}([I, M0, Mi...])
        end)
    else
        JuMP.@constraints(model, begin
            Symmetric(nu .* F0 + sum(y[i] .* Fi[i] for i in eachindex(y))) in JuMP.PSDCone()
            Symmetric(nu .* M0 + sum(y[i] .* Mi[i] for i in eachindex(y)) - I) in JuMP.PSDCone()
            Symmetric(gamma .* Matrix(I, side, side) - nu .* M0 - sum(y[i] .* Mi[i] for i in eachindex(y))) in JuMP.PSDCone()
        end)
    end

    return model
end

return ConditionNumJuMP
