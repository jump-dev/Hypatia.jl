#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
import DelimitedFiles
import DynamicPolynomials
import PolyJuMP

struct DensityEstJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    dataset_name::Symbol
    num_obs::Int
    n::Int
    X::Matrix{T}
    deg::Int
    use_monomial_space::Bool # use variables in monomial space, else interpolation space
    use_wsos::Bool # use WSOS cone formulation, else PSD formulation
end
function DensityEstJuMP{Float64}(
    dataset_name::Symbol,
    deg::Int,
    use_monomial_space::Bool,
    use_wsos::Bool)
    X = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "$dataset_name.txt"))
    (num_obs, n) = size(X)
    return DensityEstJuMP{Float64}(dataset_name, num_obs, n, X, deg, use_monomial_space, use_wsos)
end
function DensityEstJuMP{Float64}(
    num_obs::Int,
    n::Int,
    args...)
    X = randn(num_obs, n)
    return DensityEstJuMP{Float64}(:Random, num_obs, n, X, args...)
end

example_tests(::Type{DensityEstJuMP{Float64}}, ::MinimalInstances) = [
    ((5, 1, 2, true, true), false),
    ((:iris, 2, true, true), false),
    # ((5, 1, 2, true, true), true),
    # ((:iris, 2, true, true), true),
    ]
example_tests(::Type{DensityEstJuMP{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((50, 2, 2, true, true), false, options),
    ((50, 2, 2, true, false), false, options),
    ((50, 2, 2, false, true), false, options),
    ((50, 2, 2, false, false), false, options),
    ((100, 8, 2, true, true), false, options),
    ((100, 8, 2, true, false), false, options),
    ((100, 8, 2, false, true), false, options),
    ((100, 8, 2, false, false), false, options),
    ((250, 4, 4, true, true), false, options),
    ((250, 4, 4, true, false), false, options),
    ((250, 4, 4, false, true), false, options),
    ((:iris, 4, true, true), false, options),
    ((:iris, 5, true, true), false, options),
    ((:iris, 6, true, true), false, options),
    ((:iris, 4, true, false), false, options),
    ((:iris, 4, false, true), false, options),
    ((:iris, 6, false, true), false, options),
    ((:iris, 4, false, false), false, options),
    ((:cancer, 4, true, true), false, options),
    ((:cancer, 4, false, true), false, options),
    ]
end
example_tests(::Type{DensityEstJuMP{Float64}}, ::SlowInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((200, 4, 4, false, false), false, options),
    ((200, 4, 6, false, true), false, options),
    ((200, 4, 6, false, false), false, options),
    ]
end

function build(inst::DensityEstJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, X) = (inst.n, inst.X)

    domain = ModelUtilities.Box{Float64}(-ones(n), ones(n))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(inst.deg + 1, 2)
    (U, pts, Ps, w) = ModelUtilities.interpolate(domain, halfdeg, calc_w = true)

    model = JuMP.Model()

    if inst.use_monomial_space
        DynamicPolynomials.@polyvar x[1:n]
        PX = DynamicPolynomials.monomials(x, 0:(2 * halfdeg))
        JuMP.@variable(model, f, PolyJuMP.Poly(PX))
        f_pts = [f(pts_i) for pts_i in eachrow(pts)]
        f_X = [f(X_i) for X_i in eachrow(X)]
    else
        lagrange_polys = ModelUtilities.recover_lagrange_polys(pts, 2 * halfdeg)
        basis_evals = Matrix{Float64}(undef, inst.num_obs, U)
        for i in 1:inst.num_obs, j in 1:U
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
    if inst.use_wsos
        JuMP.@constraint(model, f_pts in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
    else
        psd_vars = []
        for (r, Pr) in enumerate(Ps)
            Lr = size(Pr, 2)
            psd_r = JuMP.@variable(model, [1:Lr, 1:Lr], Symmetric)
            push!(psd_vars, psd_r)
            JuMP.@SDconstraint(model, psd_r >= 0)
        end
        coeffs_lhs = JuMP.@expression(model, [u in 1:U], sum(sum(Pr[u, k] * Pr[u, l] * psd_r[k, l] * (k == l ? 1 : 2) for k in 1:size(Pr, 2) for l in 1:k) for (Pr, psd_r) in zip(Ps, psd_vars)))
        JuMP.@constraint(model, coeffs_lhs .== f_pts)
    end

    return model
end

function test_extra(inst::DensityEstJuMP, model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

# @testset "DensityEstJuMP" for inst in example_tests(DensityEstJuMP{Float64}, MinimalInstances()) test(inst...) end

return DensityEstJuMP
