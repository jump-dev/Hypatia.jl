#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))

struct EnvelopeJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    rand_halfdeg::Int
    num_polys::Int
    env_halfdeg::Int
end

function build(inst::EnvelopeJuMP{T}) where {T <: Float64} # TODO generic reals
    n = inst.n
    @assert inst.rand_halfdeg <= inst.env_halfdeg
    domain = ModelUtilities.Box{T}(-ones(T, n), ones(T, n))

    # generate interpolation
    (U, pts, Ps, _, w) = ModelUtilities.interpolate(domain, inst.env_halfdeg, calc_w = true)

    # generate random polynomials
    L = binomial(n + inst.rand_halfdeg, n)
    polys = Ps[1][:, 1:L] * rand(-9:9, L, inst.num_polys)

    model = JuMP.Model()
    JuMP.@variable(model, fpv[j in 1:U]) # values at Fekete points
    JuMP.@objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    JuMP.@constraint(model, [i in 1:inst.num_polys], polys[:, i] .- fpv in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))

    return model
end

instances[EnvelopeJuMP]["minimal"] = [
    ((1, 2, 2, 2),),
    ]
instances[EnvelopeJuMP]["fast"] = [
    ((2, 2, 3, 2),),
    ((3, 3, 3, 3),),
    ((3, 3, 5, 4),),
    ((5, 2, 5, 3),),
    ((1, 30, 2, 30),),
    ((10, 1, 3, 1),),
    ]
instances[EnvelopeJuMP]["slow"] = [
    ((4, 5, 4, 6),),
    ((2, 30, 4, 30),),
    ]
