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
    domain::MU.Domain = MU.Box{T}(-ones(T, n), ones(T, n)),
    sample::Bool = true,
    sample_factor::Int = 100,
    ) where {T <: Float64} # TODO support generic reals
    @assert n == MU.get_dimension(domain)
    @assert rand_halfdeg <= env_halfdeg

    # generate interpolation
    (U, pts, Ps, w) = MU.interpolate(domain, env_halfdeg, calc_w = true, sample = sample, sample_factor = sample_factor)

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

envelope_JuMP_fast = [
    ((Float64, 2, 2, 3, 4), false, (), ()),
    ((Float64, 2, 2, 3, 4), true, (), ()),
    # (2, 3, 2, 4),
    # (3, 3, 3, 3),
    # (3, 3, 5, 4),
    # (5, 2, 5, 2),
    # (1, 30, 2, 30),
    # (10, 1, 3, 1),
    ]
envelope_JuMP_slow = [
    # (4, 6, 4, 5),
    # (2, 30, 4, 30),
    ]

test_JuMP_instance.(envelope_JuMP, test_envelope_JuMP, envelope_JuMP_fast)
;
