#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

import DelimitedFiles

struct DensityEstJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    dataset_name::Symbol
    X::Matrix{T}
    deg::Int
    use_wsos::Bool # use WSOS cone formulation, else PSD formulation
end
function DensityEstJuMP{Float64}(dataset_name::Symbol, deg::Int, use_wsos::Bool)
    X = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "$dataset_name.txt"))
    return DensityEstJuMP{Float64}(dataset_name, X, deg, use_wsos)
end
function DensityEstJuMP{Float64}(num_obs::Int, n::Int, args...)
    X = randn(num_obs, n)
    return DensityEstJuMP{Float64}(:Random, X, args...)
end

function build(inst::DensityEstJuMP{T}) where {T <: Float64} # TODO generic reals
    X = inst.X
    (num_obs, n) = size(X)
    domain = ModelUtilities.Box{Float64}(-ones(n), ones(n)) # domain is unit box [-1,1]^n

    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    # setup interpolation
    halfdeg = div(inst.deg + 1, 2)
    (U, _, Ps, V, w) = ModelUtilities.interpolate(domain, halfdeg, calc_V = true, calc_w = true)
    # TODO maybe incorporate this interp-basis transform into MU, and do something smarter for uni/bi-variate
    F = qr!(Array(V'), Val(true))
    V_X = ModelUtilities.make_chebyshev_vandermonde(X, 2halfdeg)
    X_pts_polys = F \ V_X'

    model = JuMP.Model()
    JuMP.@variable(model, z)
    JuMP.@objective(model, Max, z)
    JuMP.@variable(model, f_pts[1:U])

    # objective epigraph
    JuMP.@constraint(model, vcat(z, X_pts_polys' * f_pts) in MOI.GeometricMeanCone(1 + num_obs))

    # density integrates to 1
    JuMP.@constraint(model, dot(w, f_pts) == 1)

    # density nonnegative
    if inst.use_wsos
        # WSOS formulation
        JuMP.@constraint(model, f_pts in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
    else
        # PSD formulation
        psd_vars = []
        for Pr in Ps
            Lr = size(Pr, 2)
            psd_r = JuMP.@variable(model, [1:Lr, 1:Lr], Symmetric)
            if Lr == 1
                # Mosek cannot handle 1x1 PSD constraints
                JuMP.@constraint(model, psd_r[1, 1] >= 0)
            else
                JuMP.@SDconstraint(model, psd_r >= 0)
            end
            push!(psd_vars, psd_r)
        end
        coeffs_lhs = JuMP.@expression(model, [u in 1:U], sum(sum(Pr[u, k] * Pr[u, l] * psd_r[k, l] * (k == l ? 1 : 2) for k in 1:size(Pr, 2) for l in 1:k) for (Pr, psd_r) in zip(Ps, psd_vars)))
        JuMP.@constraint(model, coeffs_lhs .== f_pts)
    end

    return model
end
