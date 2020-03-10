#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
==#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
import DelimitedFiles
import DynamicPolynomials
import PolyJuMP

function densityest_JuMP(
    ::Type{T},
    X::Matrix{T},
    deg::Int,
    use_monomial_space::Bool, # use variables in monomial space, else interpolation space
    use_wsos::Bool, # use WSOS cone formulation, else PSD formulation
    ) where {T <: Float64} # TODO support generic reals
    (num_obs, dim) = size(X)

    domain = MU.Box{Float64}(-ones(dim), ones(dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, Ps, w) = MU.interpolate(domain, halfdeg, calc_w = true)

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

    return (model, ())
end

densityest_JuMP(
    ::Type{T},
    data_name::Symbol,
    args...; kwargs...
    ) where {T <: Float64} = densityest_JuMP(T, eval(data_name), args...; kwargs...)

densityest_JuMP(
    ::Type{T},
    num_obs::Int,
    n::Int,
    args...; kwargs...
    ) where {T <: Float64} = densityest_JuMP(T, randn(T, num_obs, n), args...; kwargs...)

function test_densityest_JuMP(model, test_helpers, test_options)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

iris_data = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "iris.txt"))
cancer_data = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "cancer.txt"))

options = ()
densityest_JuMP_fast = [
    ((Float64, :iris_data, 4, true, true), false, (), options),
    ((Float64, :iris_data, 5, true, true), false, (), options),
    ((Float64, :iris_data, 6, true, true), false, (), options),
    ((Float64, :iris_data, 4, true, false), false, (), options),
    ((Float64, :iris_data, 4, false, true), false, (), options),
    ((Float64, :iris_data, 6, false, true), false, (), options),
    ((Float64, :iris_data, 4, false, false), false, (), options),
    ((Float64, :cancer_data, 4, true, true), false, (), options),
    ((Float64, :cancer_data, 4, false, true), false, (), options),
    ((Float64, 50, 2, 2, true, true), false, (), options),
    ((Float64, 50, 2, 2, true, false), false, (), options),
    ((Float64, 50, 2, 2, false, true), false, (), options),
    ((Float64, 50, 2, 2, false, false), false, (), options),
    ((Float64, 100, 8, 2, true, true), false, (), options),
    ((Float64, 100, 8, 2, true, false), false, (), options),
    ((Float64, 100, 8, 2, false, true), false, (), options),
    ((Float64, 100, 8, 2, false, false), false, (), options),
    ((Float64, 250, 4, 4, true, true), false, (), options),
    ((Float64, 250, 4, 4, true, false), false, (), options),
    ((Float64, 250, 4, 4, false, true), false, (), options),
    ]
densityest_JuMP_slow = [
    ((Float64, 200, 4, 4, false, false), false, (), options),
    ((Float64, 200, 4, 6, false, true), false, (), options),
    ((Float64, 200, 4, 6, false, false), false, (), options),
    ]

@testset "densityest_JuMP" begin test_JuMP_instance.(densityest_JuMP, test_densityest_JuMP, densityest_JuMP_fast) end
;
