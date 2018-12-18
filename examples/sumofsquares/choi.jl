#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Example taken from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/choi.jl
=#

# @testset "Choi with $(factory.constructor)" for factory in sdp_factories
#     @polyvar x y z
#
#     m = SOSModel(factory)
#
#     C = [x^2+2y^2 -x*y -x*z;
#          -x*y y^2+2z^2 -y*z;
#          -x*z -y*z z^2+2x^2]
#
#     @SDconstraint m C >= 0
#
#     JuMP.optimize!(m)
#     @test JuMP.dual_status(m) == MOI.InfeasibilityCertificate
# end


using JuMP
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using PolyJuMP
using Test

model = Model(with_optimizer(Hypatia.Optimizer))

@polyvar x y z

C = [x^2+2y^2 -x*y -x*z;
     -x*y y^2+2z^2 -y*z;
     -x*z -y*z z^2+2x^2]

n = size(C, 1)
@polyvar w[1:n]
d = maximum(DynamicPolynomials.maxdegree.(C))

dom = Hypatia.FreeDomain(n)
full_dom = Hypatia.add_free_vars(dom)
(U, pts, P0, _, _) = Hypatia.interpolate(full_dom, d+2, sample_factor=50, sample=true)

scalar_wsos_cone = WSOSPolyInterpCone(U, [P0])
conv_condition = w'*C*w
@constraint(model, [conv_condition(pts[i, :]) for i in 1:U] in scalar_wsos_cone)

JuMP.optimize!(model)
