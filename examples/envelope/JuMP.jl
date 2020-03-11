#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))

function envelope_JuMP(
    ::Type{T},
    n::Int,
    rand_halfdeg::Int,
    num_polys::Int,
    env_halfdeg::Int;
    domain::ModelUtilities.Domain = ModelUtilities.Box{T}(-ones(T, n), ones(T, n)),
    ) where {T <: Float64} # TODO support generic reals
    @assert n == ModelUtilities.get_dimension(domain)
    @assert rand_halfdeg <= env_halfdeg

    # generate interpolation
    (U, pts, Ps, w) = ModelUtilities.interpolate(domain, env_halfdeg, calc_w = true)

    # generate random polynomials
    L = binomial(n + rand_halfdeg, n)
    polys = Ps[1][:, 1:L] * rand(-9:9, L, num_polys)

    model = JuMP.Model()
    JuMP.@variable(model, fpv[j in 1:U]) # values at Fekete points
    JuMP.@objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    JuMP.@constraint(model, [i in 1:num_polys], polys[:, i] .- fpv in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))

    return (model, ())
end

function test_envelope_JuMP(model, test_helpers, test_options)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

options = ()
envelope_JuMP_fast = [
    ((Float64, 2, 2, 3, 4), false, (), options),
    ((Float64, 2, 3, 2, 4), false, (), options),
    ((Float64, 3, 3, 3, 3), false, (), options),
    ((Float64, 3, 3, 5, 4), false, (), options),
    ((Float64, 5, 2, 5, 2), false, (), options),
    ((Float64, 1, 30, 2, 30), false, (), options),
    ((Float64, 10, 1, 3, 1), false, (), options),
    ]
envelope_JuMP_slow = [
    ((Float64, 4, 6, 4, 5), false, (), options),
    ((Float64, 2, 30, 4, 30), false, (), options),
    ]

@testset "envelope_JuMP" begin test_JuMP_instance.(envelope_JuMP, test_envelope_JuMP, envelope_JuMP_fast) end
;
