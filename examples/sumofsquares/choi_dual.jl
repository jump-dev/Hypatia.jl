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

const rt2 = sqrt(2)

function run_JuMP_choi_wsos_dual()
    @polyvar x y z
    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))

    C = -[x^2+2y^2 -x*y -x*z; -x*y y^2+2z^2 -y*z; -x*z -y*z z^2+2x^2] .* (x*y*z)^0
    d = div(maximum(DynamicPolynomials.maxdegree.(C)), 2)
    dom = Hypatia.FreeDomain(3)

    (U, pts, P0, _, _) = Hypatia.interpolate(dom, d, sample_factor=10, sample=true)

    @variable(model, z[i=1:3, j=1:i, 1:U])

    mat_wsos_cone = WSOSPolyInterpMatCone(3, U, [P0], true)
    @constraint(model, [z[i,j,u] * (i == j ? 1.0 : rt2) for i in 1:3 for j in 1:i for u in 1:U] in mat_wsos_cone)

    @objective(model, Max,
        2*sum(z[i,j,u] * C[i,j](pts[u, :]...) for i in 1:3 for j in 1:i-1 for u in 1:U) +
        sum(z[i,i,u] * C[i,i](pts[u, :]...) for i in 1:3 for u in 1:U)
        )

    JuMP.optimize!(model)

    @test JuMP.primal_status(model) == MOI.INFEASIBILITY_CERTIFICATE
    return
end
