#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
==#

using LinearAlgebra
import Random
using Test
import DelimitedFiles
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
import PolyJuMP
import Hypatia
const MU = Hypatia.ModelUtilities

iris_data = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "iris.txt"))
cancer_data = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "iris.txt"))

function densityest_JuMP(
    T::Type{Float64}, # TODO support generic reals
    X::Matrix{T},
    deg::Int,
    use_monomial_space::Bool, # use variables in monomial space, else interpolation space
    use_wsos::Bool; # use WSOS cone formulation, else PSD formulation
    sample::Bool = true,
    sample_factor::Int = 100,
    )
    (num_obs, dim) = size(X)

    domain = MU.Box{Float64}(-ones(dim), ones(dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, Ps, w) = MU.interpolate(domain, halfdeg, calc_w = true, sample = sample, sample_factor = sample_factor)

    model = JuMP.Model()

    if use_monomial_space
        DynamicPolynomials.@polyvar x[1:dim]
        PX = DynamicPolynomials.monomials(x, 0:(2 * halfdeg))
        JuMP.@variable(model, f, PolyJuMP.Poly(PX))
        f_pts = [f(pts_i) for pts_i in eachrow(pts)]
        f_X = [f(X_i) for X_i in eachrow(X)]
    else
        lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)
        basis_evals = Matrix{Float64}(undef, num_obs, U)
        for i in 1:num_obs, j in 1:U
            basis_evals[i, j] = lagrange_polys[j](X[i, :])
        end
        JuMP.@variable(model, f_pts[1:U])
        f_X = [dot(f_pts, b_i) for b_i in eachrow(basis_evals)]
    end

    JuMP.@constraint(model, dot(w, f_pts) == 1.0) # integrate to 1

    JuMP.@variable(model, z)
    JuMP.@objective(model, Max, z)
    JuMP.@constraint(model, vcat(z, f_X) in MOI.GeometricMeanCone(1 + length(f_X)))

    # density nonnegative
    if use_wsos
        JuMP.@constraint(model, f_pts in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
    else
        psd_vars = []
        for (r, Pr) in enumerate(Ps)
            Lr = size(Pr, 2)
            psd_r = JuMP.@variable(model, [1:Lr, 1:Lr], Symmetric)
            push!(psd_vars, psd_r)
            JuMP.@SDconstraint(model, psd_r >= 0)
        end
        JuMP.@constraint(model, sum(diag(Pr * psd_r * Pr') for (Pr, psd_r) in zip(Ps, psd_vars)) .== f_pts)
    end

    return (model = model,)
end

densityest_JuMP(T::Type{Float64}, data_name::Symbol, args...; kwargs...) = densityest_JuMP(T, eval(data_name), args...; kwargs...)

densityest_JuMP(T::Type{Float64}, num_obs::Int, n::Int, args...; kwargs...) = densityest_JuMP(T, randn(T, num_obs, n), args...; kwargs...)

function test_densityest_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = densityest_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

densityest_JuMP_fast = [
    (:iris_data, 4, true, true),
    (:iris_data, 4, true, false),
    (:iris_data, 4, false, true),
    (:iris_data, 4, false, false),
    (:iris_data, 6, true, true),
    (:iris_data, 6, false, true),
    (:cancer_data, 4, true, true),
    (:cancer_data, 4, false, true),
    (:cancer_data, 6, false, true),
    (200, 2, 2, true, true),
    (200, 2, 2, true, false),
    (200, 2, 2, false, true),
    (200, 2, 2, false, false),
    (100, 8, 2, true, true),
    (100, 8, 2, true, false),
    (100, 8, 2, false, true),
    (100, 8, 2, false, false),
    (250, 4, 4, true, true),
    (250, 4, 4, true, false),
    (250, 4, 4, false, true),
    (250, 4, 4, false, false),
    ]
densityest_JuMP_slow = [
    # TODO
    ]
