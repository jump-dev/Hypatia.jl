#=
see description in examples/envelope/native.jl
=#

struct EnvelopeJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    rand_halfdeg::Int
    num_polys::Int
    env_halfdeg::Int
end

function build(inst::EnvelopeJuMP{T}) where {T <: Float64}
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
