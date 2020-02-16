#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
==#

using LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
import PolyJuMP
import Hypatia
const MU = Hypatia.ModelUtilities

include(joinpath(@__DIR__, "data.jl"))

function densityestJuMP(
    X::Matrix{Float64},
    deg::Int;
    use_monomials::Bool = false, # use variables in monomial space, else interpolation space
    use_wsos::Bool = true, # use WSOS cone formulation, else PSD formulation
    geomean_obj::Bool = true, # use geomean formulation, else exponential cone sum-log formulation
    sample_factor::Int = 100,
    )
    (nobs, dim) = size(X)

    domain = MU.Box{Float64}(-ones(dim), ones(dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, Ps, w) = MU.interpolate(domain, halfdeg, sample = true, calc_w = true, sample_factor = sample_factor)

    model = JuMP.Model()

    if use_monomials
        DynamicPolynomials.@polyvar x[1:dim]
        PX = DynamicPolynomials.monomials(x, 0:(2 * halfdeg))
        JuMP.@variable(model, f, PolyJuMP.Poly(PX))
        f_pts = [f(pts_i) for pts_i in eachrow(pts)]
        f_X = [f(X_i) for X_i in eachrow(X)]
    else
        lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)
        basis_evals = Matrix{Float64}(undef, nobs, U)
        for i in 1:nobs, j in 1:U
            basis_evals[i, j] = lagrange_polys[j](X[i, :])
        end
        JuMP.@variable(model, f_pts[1:U])
        f_X = [dot(f_pts, b_i) for b_i in eachrow(basis_evals)]
    end

    JuMP.@constraint(model, dot(w, f_pts) == 1.0) # integrate to 1

    if geomean_obj
        JuMP.@variable(model, z)
        JuMP.@objective(model, Max, z)
        JuMP.@constraint(model, vcat(z, f_X) in MOI.GeometricMeanCone(1 + length(f_X)))
    else
        JuMP.@variable(model, z[1:nobs])
        JuMP.@objective(model, Max, sum(z))
        JuMP.@constraint(model, [i in 1:nobs], vcat(z[i], 1.0, f_X[i]) in MOI.ExponentialCone()) # hypograph of log
    end
    @show "got exp constraints"
    flush(stdout)

    # density nonnegative
    if use_wsos
        JuMP.@constraint(model, f_pts in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
    else
        @show "starting sdp buildup"
        flush(stdout)
        coeffs_vec = zeros(JuMP.GenericAffExpr{Float64,JuMP.VariableRef}, U)
        for Pr in Ps
            Lr = size(Pr, 2)
            JuMP.@variable(model, psd_r[1:Lr, 1:Lr], Symmetric)
            JuMP.@SDconstraint(model, psd_r >= 0)
            for i in 1:U
                JuMP.add_to_expression!(coeffs_vec[i], sum(Pr[i, j] * Pr[i, k] * psd_r[j, k] for j in 1:Lr for k in 1:Lr))
            end
        end
        # @show "starting sdp constraint"
        # flush(stdout)
        # for (r, Pr) in enumerate(Ps)
        #     L = size(Pr, 2)
        #     for i in 1:U
        #         JuMP.add_to_expression!(coeffs_vec[i], sum(Pr[i, j] * Pr[i, k] * psd_vars[r][j, k] for j in 1:L for k in 1:L))
        #         # coeffs_vec[i] += sum(Pr[i, j] * Pr[i, k] * psd_vars[r][j, k] for j in 1:L for k in 1:L)
        #         # for j in 1:L
        #         #     for k in 1:(j - 1)
        #         #         JuMP.add_to_expression!(coeffs_vec[i], Pr[i, j] * Pr[i, k] * psd_vars[r][j, k] * 2)
        #         #     end
        #         #     JuMP.add_to_expression!(coeffs_vec[i], Pr[i, j] * Pr[i, j] * psd_vars[r][j, j])
        #         # end
        #     end
        # end
        # expr = JuMP.@expression(model, diag(Ps[1] * psd_vars[1] * Ps[1]'))
        # for i in eachindex(Ps[2:end])
        #     expr .+= diag(Ps[i] * psd_vars[i] * Ps[i]')
        # end
        JuMP.@constraint(model, coeffs_vec .== f_pts)
        # JuMP.@constraint(model, sum(diag(Pr * psd_r * Pr') for (Pr, psd_r) in zip(Ps, psd_vars)) .== f_pts)
        @show "finished sdp constraint"
        flush(stdout)
    end

    return (model = model,)
end

densityestJuMP(nobs::Int, n::Int, deg::Int; options...) = densityestJuMP(randn(nobs, n), deg; options...)

densityestJuMP1() = densityestJuMP(iris_data(), 4)
densityestJuMP2() = densityestJuMP(iris_data(), 6)
densityestJuMP3() = densityestJuMP(cancer_data(), 4)
densityestJuMP4() = densityestJuMP(cancer_data(), 6)
densityestJuMP5() = densityestJuMP(200, 2, 3, use_monomials = false, use_wsos = true, geomean_obj = false)
densityestJuMP6() = densityestJuMP(200, 2, 3, use_monomials = true, use_wsos = true, geomean_obj = false)
densityestJuMP7() = densityestJuMP(200, 2, 3, use_monomials = false, use_wsos = false, geomean_obj = false)
densityestJuMP8() = densityestJuMP(200, 2, 3, use_monomials = true, use_wsos = false, geomean_obj = false)
densityestJuMP9() = densityestJuMP(200, 2, 3, use_monomials = false, use_wsos = true, geomean_obj = true)
densityestJuMP10() = densityestJuMP(200, 2, 3, use_monomials = true, use_wsos = true, geomean_obj = true)
densityestJuMP11() = densityestJuMP(200, 2, 3, use_monomials = false, use_wsos = false, geomean_obj = true)
densityestJuMP12() = densityestJuMP(200, 2, 3, use_monomials = true, use_wsos = false, geomean_obj = true)

function test_densityestJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_densityestJuMP_all(; options...) = test_densityestJuMP.([
    densityestJuMP1,
    densityestJuMP2,
    densityestJuMP3,
    densityestJuMP4,
    densityestJuMP5,
    densityestJuMP6,
    densityestJuMP7,
    densityestJuMP8,
    densityestJuMP9,
    densityestJuMP10,
    densityestJuMP11,
    densityestJuMP12,
    ], options = options)

test_densityestJuMP(; options...) = test_densityestJuMP.([
    densityestJuMP1,
    densityestJuMP3,
    densityestJuMP5,
    densityestJuMP6,
    densityestJuMP7,
    densityestJuMP8,
    densityestJuMP9,
    densityestJuMP10,
    densityestJuMP11,
    densityestJuMP12,
    ], options = options)
