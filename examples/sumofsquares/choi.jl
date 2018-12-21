#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/choi.jl
Verifies that a given polynomial matrix is not a Sum-of-Squares matrix
See Choi, M. D., "Positive semidefinite biquadratic forms", Linear Algebra and its Applications, 1975, 12(2), 95-100
=#

using JuMP
using MathOptInterface
MOI = MathOptInterface
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using PolyJuMP
using Test

function run_JuMP_choi_wsos(use_matrixwsos::Bool)
    @polyvar x y z
    C = [x^2+2y^2 -x*y -x*z; -x*y y^2+2z^2 -y*z; -x*z -y*z z^2+2x^2] .* (x*y*z)^0
    d = div(maximum(DynamicPolynomials.maxdegree.(C)), 2)
    dom = Hypatia.FreeDomain(3)

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true)) #, tolabsopt=1e-9, tolrelopt=1e-7, tolfeas=1e-9))
    if use_matrixwsos
        (U, pts, P0, _, _) = Hypatia.interpolate(dom, d, sample_factor=10, sample=true)
        mat_wsos_cone = WSOSPolyInterpMatCone(3, U, [P0])
        @constraint(model, [AffExpr(C[i,j](pts[u, :])) for i in 1:3 for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        dom2 = Hypatia.add_free_vars(dom)
        (U, pts, P0, _, _) = Hypatia.interpolate(dom2, d+2, sample_factor=20, sample=true)
        scalar_wsos_cone = WSOSPolyInterpCone(U, [P0])
        @polyvar w[1:3]
        wCw = w'*C*w
        @constraint(model, [AffExpr(wCw(pts[u, :])) for u in 1:U] in scalar_wsos_cone)
    end

    JuMP.optimize!(model)
    @test JuMP.dual_status(model) == MOI.INFEASIBILITY_CERTIFICATE
end

run_JuMP_choi_scalarwsos() = run_JuMP_choi_wsos(false)
run_JuMP_choi_matrixwsos() = run_JuMP_choi_wsos(true)

coeffs = repeat(Matrix{Float64}(I, 3, 3), 4) .* [ones(3); x .* ones(3); y .* ones(3); z .* ones(3)]
