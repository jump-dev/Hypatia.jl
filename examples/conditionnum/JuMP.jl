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

struct ConditionNumJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    side::Int
    len_y::Int
    use_linmatrixineq::Bool # use linmatrixineq cone, else PSD formulation
end

function build(inst::ConditionNumJuMP{T}) where {T <: Float64} # TODO generic reals
    (side, len_y) = (inst.side, inst.len_y)

    rand_pd() = (Mh = randn(side, side); Symmetric(Mh * Mh'))
    Mi = [rand_pd() for i in 1:len_y]
    M0 = rand_pd()
    # make some F_i matrices pos def
    Fi = [(rand() > 0.5 || i <= 2) ? rand_pd() : Symmetric(randn(side, side)) for i in 1:len_y]
    F0 = rand_pd() + I

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
            Symmetric(nu * F0 + sum(y[i] * Fi[i] for i in eachindex(y))) in JuMP.PSDCone()
            Symmetric(nu * M0 + sum(y[i] * Mi[i] for i in eachindex(y)) - I) in JuMP.PSDCone()
            Symmetric(gamma * Matrix(I, side, side) - nu * M0 - sum(y[i] * Mi[i] for i in eachindex(y))) in JuMP.PSDCone()
        end)
    end

    return model
end

tols7 = (tol_feas = 1e-7, tol_rel_opt = 1e-7, tol_abs_opt = 1e-7)
tols6 = (tol_feas = 1e-6, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
insts[ConditionNumJuMP]["minimal"] = [
    ((3, 2, true),),
    ((3, 2, false),),
    ]
insts[ConditionNumJuMP]["fast"] = [
    ((3, 4, true), nothing, tols7),
    ((3, 4, false), nothing, tols7),
    ((10, 15, true), nothing, tols7),
    ((10, 15, false), nothing, tols7),
    ((20, 10, true), nothing, tols6),
    ((100, 40, false), nothing, tols6),
    ]
insts[ConditionNumJuMP]["slow"] = [
    ((100, 10, true), nothing, tols7),
    ((100, 40, true), nothing, tols7),
    ]
