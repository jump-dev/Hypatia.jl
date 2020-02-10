#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

using LinearAlgebra
using SparseArrays
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia

function matrixregressionJuMP(
    Y::Matrix{Float64},
    X::Matrix{Float64};
    lam_fro::Float64 = 0.0,
    lam_nuc::Float64 = 0.0,
    lam_las::Float64 = 0.0,
    lam_glr::Float64 = 0.0,
    lam_glc::Float64 = 0.0,
    )
    @assert lam_fro >= 0
    @assert lam_nuc >= 0
    @assert lam_las >= 0
    (data_n, data_m) = size(Y)
    data_p = size(X, 2)
    @assert size(X, 1) == data_n
    @assert data_p >= data_m

    if data_n > data_p
        Xhalf = qr(X).R
    else
        Xhalf = X
    end

    model = JuMP.Model()
    JuMP.@variable(model, A[1:data_p, 1:data_m])
    JuMP.@variable(model, u_fro)
    JuMP.@constraint(model, vcat(u_fro, 0.5, vec(Xhalf * A)) in JuMP.RotatedSecondOrderCone())
    obj = (u_fro - 2 * dot(X' * Y, A)) / (2 * data_n)

    if !iszero(lam_fro)
        JuMP.@variable(model, t_fro)
        JuMP.@constraint(model, vec(t_fro, 0.5, vec(A)) in JuMP.RotatedSecondOrderCone())
        obj += lam_nuc * t_nuc
    end
    if !iszero(lam_nuc)
        JuMP.@variable(model, t_nuc)
        JuMP.@constraint(model, vec(t_nuc, vec(A)) in MOI.NormNuclearCone(data_n, data_p))
        obj += lam_nuc * t_nuc
    end
    if !iszero(lam_las)
        JuMP.@variable(model, t_las)
        JuMP.@constraint(model, vec(t_las, vec(A)) in MOI.NormOneCone(data_p * data_m + 1))
        obj += lam_las * t_nuc
    end
    if !iszero(lam_glr)
        JuMP.@variable(model, t_glr[1:data_p])
        JuMP.@constraint(model, [i = 1:data_p], vec(t_glr[i], 0.5, A[i, :]) in JuMP.RotatedSecondOrderCone())
        obj += lam_glr * sum(t_glr)
    end
    if !iszero(lam_glr)
        JuMP.@variable(model, t_glc[1:data_m])
        JuMP.@constraint(model, [i = 1:data_m], vec(t_glc[i], 0.5, A[:, i]) in JuMP.RotatedSecondOrderCone())
        obj += lam_glc * sum(t_glc)
    end

    return (model = model,)
end

function matrixregressionJuMP(
    n::Int,
    m::Int,
    p::Int;
    A_max_rank::Int = div(m, 2) + 1,
    A_sparsity::Real = max(0.2, inv(sqrt(m * p))),
    Y_noise::Real = 0.01,
    model_kwargs...
    ) where {T <: Real}
    @assert p >= m
    @assert 1 <= A_max_rank <= m
    @assert 0 < A_sparsity <= 1

    A_left = sprandn(p, A_max_rank, A_sparsity)
    A_right = sprandn(A_max_rank, m, A_sparsity)
    A = 10 * A_left * A_right
    X = randn(n, p)
    Y = X * A + Y_noise * randn(n, m)

    Y = Matrix(Y)
    X = Matrix(X)

    return matrixregressionJuMP(Y, X; model_kwargs...)
end

matrixregressionJuMP1() = matrixregression(5, 3, 4)
matrixregressionJuMP2() = matrixregression(5, 3, 4, lam_fro = 0.1, lam_nuc = 0.1, lam_las = 0.1, lam_glc = 0.2, lam_glr = 0.2)
matrixregressionJuMP3() = matrixregression(3, 4, 5)
matrixregressionJuMP4() = matrixregression(3, 4, 5, lam_fro = 0.1, lam_nuc = 0.1, lam_las = 0.1, lam_glc = 0.2, lam_glr = 0.2)

function test_matrixregressionJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_matrixregressionJuMP_all(; options...) = test_matrixregressionJuMP.([
    matrixregressionJuMP1,
    matrixregressionJuMP2,
    matrixregressionJuMP3,
    matrixregressionJuMP4,
    ], options = options)

test_matrixregressionJuMP(; options...) = test_matrixregressionJuMP.([
    matrixregressionJuMP1,
    ], options = options)
