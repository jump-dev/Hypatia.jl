#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

Lyapunov stability example from https://stanford.edu/class/ee363/sessions/s4notes.pdf

min 0
P - I in S_+
[A'*P + P*A + alpha*P + t*gamma^2*I, P;
P, -tI] in S_-

TODO use triangle cone?
figure out how to ensure problem is feasible so we can test
=#

using LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia

function lyapunov(
    side::Int,
    use_matrixepipersquare::Bool = true,
    )
    A = randn(side, side)
    alpha = rand()
    gamma = rand()
    cone_dim = div(side * (side + 1), 2)

    model = JuMP.Model()
    JuMP.@variable(model, P[1:side, 1:side], Symmetric)
    JuMP.@variable(model, t)
    U = A' * P .+ P * A .+ alpha * P .+ t * gamma ^ 2
    JuMP.@constraints(model, begin
        P - I in JuMP.PSDCone()
        vcat(vec(-U), t, vec(-P)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(side, side)
    end)
    JuMP.@objective(model, Min, 0)

    return (model = model,)
end

lyapunovJuMP1() = lyapunovJuMP(10, 20)

function test_lyapunovJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, Hypatia.Optimizer)
    JuMP.optimize!(d.model,)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_expdesignJuMP_all(; options...) = test_lyapunovJuMP.([
    lyapunovJuMP1,
    ], options = options)

test_lyapunovJuMP(; options...) = test_lyapunovJuMP.([
    lyapunovJuMP1,
    ], options = options)
