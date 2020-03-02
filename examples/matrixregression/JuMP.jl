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

function matrixregression_JuMP(
    T::Type{Float64}, # TODO support generic reals
    Y::Matrix{T},
    X::Matrix{T},
    lam_fro::Real, # penalty on Frobenius norm
    lam_nuc::Real, # penalty on nuclear norm
    lam_las::Real, # penalty on l1 norm
    lam_glr::Real, # penalty on penalty on row group l1 norm
    lam_glc::Real, # penalty on penalty on column group l1 norm
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
        # NOTE this penalty is usually squared
        JuMP.@constraint(model, vcat(t_fro, vec(A)) in JuMP.SecondOrderCone())
        obj += lam_fro * t_fro
    end
    if !iszero(lam_nuc)
        JuMP.@variable(model, t_nuc)
        JuMP.@constraint(model, vcat(t_nuc, vec(A')) in MOI.NormNuclearCone(data_m, data_p))
        obj += lam_nuc * t_nuc
    end
    if !iszero(lam_las)
        JuMP.@variable(model, t_las)
        JuMP.@constraint(model, vcat(t_las, vec(A)) in MOI.NormOneCone(data_p * data_m + 1))
        obj += lam_las * t_las
    end
    if !iszero(lam_glr)
        JuMP.@variable(model, t_glr[1:data_p])
        JuMP.@constraint(model, [i = 1:data_p], vcat(t_glr[i], A[i, :]) in JuMP.SecondOrderCone())
        obj += lam_glr * sum(t_glr)
    end
    if !iszero(lam_glr)
        JuMP.@variable(model, t_glc[1:data_m])
        JuMP.@constraint(model, [i = 1:data_m], vcat(t_glc[i], A[:, i]) in JuMP.SecondOrderCone())
        obj += lam_glc * sum(t_glc)
    end
    JuMP.@objective(model, Min, obj)

    return (model = model,)
end

function matrixregression_JuMP(
    T::Type{Float64}, # TODO support generic reals
    n::Int,
    m::Int,
    p::Int,
    args...;
    A_max_rank::Int = div(m, 2) + 1,
    A_sparsity::Real = max(0.2, inv(sqrt(m * p))),
    Y_noise::Real = 0.01,
    )
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

    return matrixregression_JuMP(T, Y, X, args...)
end

function test_matrixregression_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = matrixregression_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

matrixregression_JuMP_fast = [
    (5, 3, 4, 0, 0, 0, 0, 0),
    (5, 3, 4, 0.1, 0.1, 0.1, 0.2, 0.2),
    (5, 3, 4, 0, 0.1, 0.1, 0, 0),
    (3, 4, 5, 0, 0, 0, 0, 0),
    (3, 4, 5, 0.1, 0.1, 0.1, 0.2, 0.2),
    (3, 4, 5, 0, 0.1, 0.1, 0, 0),
    (15, 10, 20, 0, 0, 0, 0, 0),
    (15, 10, 20, 0.1, 0.1, 0.1, 0.2, 0.2),
    (15, 10, 20, 0, 0.1, 0.1, 0, 0),
    (100, 8, 12, 0, 0, 0, 0, 0),
    (100, 8, 12, 0.1, 0.1, 0.1, 0.2, 0.2),
    (100, 8, 12, 0, 0.1, 0.1, 0, 0),
    ]
matrixregression_JuMP_slow = [
    # TODO
    ]
