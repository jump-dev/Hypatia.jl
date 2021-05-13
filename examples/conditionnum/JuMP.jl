#=
minimize the condition number of positive definite matrix
M(x) = M_0 + sum_i x_i*M_i
subject to F(x) = F_0 + sum_i x_i*F_i in S_+

original formulation:
minimize    gamma
subject to  mu >= 0
            F(x) in S_+
            M(x) - mu*I in S_+
            mu*gamma*I - M(x) in S_+
introduce nu = inv(mu), y = x/mu:
minimize    gamma
subject to  nu >= 0
            nu*F_0 + sum_i y_i*F_i in S_+
            nu*M_0 + sum_i y_i*M_i - I in S_+
            gamma*I - nu*M_0 - sum_i y_i*M_i in S_+
we make F_0 and M_0 positive definite to ensure existence of a feasible solution

see section 3.2 "Linear Matrix Inequalities in System and Control Theory" by
S. Boyd, L. El Ghaoui, E. Feron, and V. Balakrishnan
=#

struct ConditionNumJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    side::Int
    len_y::Int
    use_linmatrixineq::Bool # use linmatrixineq cone, else PSD formulation
end

function build(inst::ConditionNumJuMP{T}) where {T <: Float64}
    (side, len_y) = (inst.side, inst.len_y)

    rand_pd() = (Mh = randn(side, side); Symmetric(Mh * Mh'))
    Mi = [rand_pd() for i in 1:len_y]
    M0 = rand_pd()
    # make some F_i matrices pos def
    Fi = [(rand() > 0.5 || i <= 2) ? rand_pd() :
        Symmetric(randn(side, side)) for i in 1:len_y]
    F0 = rand_pd() + I

    model = JuMP.Model()
    JuMP.@variables(model, begin
        gamma
        nu >= 0
        y[1:len_y]
    end)
    JuMP.@objective(model, Min, gamma)

    if inst.use_linmatrixineq
        lmiT = Hypatia.LinMatrixIneqCone{T}
        JuMP.@constraints(model, begin
            vcat(nu, y) in lmiT([F0, Fi...])
            vcat(-1, nu, y) in lmiT([I, M0, Mi...])
            vcat(gamma, -nu, -y) in lmiT([I, M0, Mi...])
        end)
    else
        S1 = Symmetric(nu * F0 + sum(y[i] * Fi[i] for i in eachindex(y)))
        S2 = Symmetric(nu * M0 + sum(y[i] * Mi[i] for i in eachindex(y)) - I)
        S3 = Symmetric(gamma * Matrix(I, side, side) - nu * M0 -
            sum(y[i] * Mi[i] for i in eachindex(y)))

        JuMP.@constraints(model, begin
            S1 in JuMP.PSDCone()
            S2 in JuMP.PSDCone()
            S3 in JuMP.PSDCone()
        end)
    end

    return model
end
