#=
see description in native.jl
=#

import DelimitedFiles

struct DensityEstJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    dataset_name::Symbol
    X::Matrix{T}
    deg::Int
    geomean_obj::Bool # use geomean in objective, else sum of logs
    use_wsos::Bool # use WSOS cone formulation, else PSD formulation
    use_nlog::Bool # if using log obj, use n-dim HypoPerLog cone, else use many 3-dim HypoPerLog cones
end
function DensityEstJuMP{Float64}(dataset_name::Symbol, deg::Int, use_wsos::Bool)
    X = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "$dataset_name.txt"))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2
    return DensityEstJuMP{Float64}(dataset_name, X, deg, use_wsos)
end
function DensityEstJuMP{Float64}(num_obs::Int, n::Int, args...)
    X = 2 * rand(num_obs, n) .- 1
    return DensityEstJuMP{Float64}(:Random, X, args...)
end

function build(inst::DensityEstJuMP{T}) where {T <: Float64}
    X = inst.X
    (num_obs, n) = size(X)
    domain = ModelUtilities.Box{Float64}(-ones(n), ones(n)) # domain is unit box [-1,1]^n

    # setup interpolation
    halfdeg = div(inst.deg + 1, 2)
    (U, _, Ps, V, w) = ModelUtilities.interpolate(domain, halfdeg, calc_V = true, calc_w = true)
    # TODO maybe incorporate this interp-basis transform into MU, and do something smarter for uni/bi-variate
    F = qr!(Array(V'), Val(true))
    V_X = ModelUtilities.make_chebyshev_vandermonde(X, 2 * halfdeg)
    X_pts_polys = F \ V_X'

    model = JuMP.Model()
    JuMP.@variable(model, z)
    JuMP.@objective(model, Max, z)
    JuMP.@variable(model, f_pts[1:U])

    # objective epigraph
    obj_vec = X_pts_polys' * f_pts
    if inst.geomean_obj
        JuMP.@constraint(model, vcat(z, obj_vec) in MOI.GeometricMeanCone(1 + num_obs))
    elseif inst.use_nlog
        JuMP.@constraint(model, vcat(z, 1, obj_vec) in Hypatia.HypoPerLogCone{Float64}(2 + num_obs))
    else
        # EF for big log cone using 3-dim log cones (equivalent to MOI exponential cones)
        JuMP.@variable(model, y[1:num_obs])
        JuMP.@constraint(model, z <= sum(y))
        JuMP.@constraint(model, [i in 1:num_obs], [y[i], 1, obj_vec[i]] in MOI.ExponentialCone())
    end

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
