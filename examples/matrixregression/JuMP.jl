#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

using LinearAlgebra
using SparseArrays
import Random
using Test
import JuMP
const MOI = JuMP.MOI
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
    @assert min(lam_fro, lam_nuc, lam_las, lam_glr, lam_glc) >= 0
    (data_n, data_m) = size(Y)
    data_p = size(X, 2)
    @assert size(X, 1) == data_n
    @assert data_p >= data_m

    Qhalf = (data_n > data_p) ? qr(X).R : X # dimension reduction via QR if helpful

    model = JuMP.Model()
    JuMP.@variable(model, A[1:data_p, 1:data_m])
    JuMP.@variable(model, u_fro)
    JuMP.@constraint(model, vcat(u_fro, 0.5, vec(Qhalf * A)) in JuMP.RotatedSecondOrderCone())
    obj = (u_fro - 2 * dot(X' * Y, A)) / (2 * data_n)

    if !iszero(lam_fro)
        JuMP.@variable(model, t_fro)
        JuMP.@constraint(model, vcat(t_fro, vec(A)) in JuMP.SecondOrderCone())
        obj += lam_fro * t_fro
    end
    if !iszero(lam_nuc)
        JuMP.@variable(model, t_nuc)
        JuMP.@constraint(model, vcat(t_nuc, vec(A)) in MOI.NormNuclearCone(data_p, data_m))
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

    return (
        model = model,
        n = data_n, m = data_m, p = data_p, Y = Y, X = X,
        lam_fro = lam_fro, lam_nuc = lam_nuc, lam_las = lam_las, lam_glr = lam_glr, lam_glc = lam_glc,
        )
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

    return matrixregression_JuMP(T, Matrix(Y), Matrix(X), args...)
end

function test_matrixregression_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = matrixregression_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    r = d.model.moi_backend.optimizer.model.optimizer.result
    if r.status == :Optimal
        # check objective value is correct
        R = eltype(d.Y)
        if R <: Complex
            A_opt_real = reshape(r.x[2:(1 + 2 * d.p * d.m)], 2 * d.p, d.m)
            A_opt = zeros(R, d.p, d.m)
            for k in 1:d.p
                @. @views A_opt[k, :] = A_opt_real[2k - 1, :] + A_opt_real[2k, :] * im
            end
        else
            A_opt = reshape(r.x[2:(1 + d.p * d.m)], d.p, d.m)
        end
        loss = (1/2 * sum(abs2, d.X * A_opt) - real(dot(d.X' * d.Y, A_opt))) / d.n
        obj_try = loss + d.lam_fro * norm(vec(A_opt), 2) +
            d.lam_nuc * sum(svd(A_opt).S) + d.lam_las * norm(vec(A_opt), 1) +
            d.lam_glr * sum(norm, eachrow(A_opt)) + d.lam_glc * sum(norm, eachcol(A_opt))
        @test r.primal_obj ≈ obj_try atol = 1e-4 rtol = 1e-4
    end
    return r
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
