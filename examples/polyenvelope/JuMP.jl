#=
see description in examples/polyenvelope/native.jl
=#

struct PolyEnvelopeJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    rand_halfdeg::Int
    num_polys::Int
    env_halfdeg::Int
end

function build(inst::PolyEnvelopeJuMP{T}) where {T <: Float64}
    n = inst.n
    @assert inst.rand_halfdeg <= inst.env_halfdeg
    domain = PolyUtils.BoxDomain{T}(-ones(T, n), ones(T, n))

    # generate interpolation
    (U, pts, Ps, _, w) = PolyUtils.interpolate(domain, inst.env_halfdeg,
        get_quadr = true)

    # generate random polynomials
    L = binomial(n + inst.rand_halfdeg, n)
    polys = Ps[1][:, 1:L] * rand(-9:9, L, inst.num_polys)

    model = JuMP.Model()
    JuMP.@variable(model, fpv[j in 1:U]) # values at Fekete points
    JuMP.@objective(model, Max, dot(fpv, w)) # integral over domain (quadrature)
    wsosT = Hypatia.WSOSInterpNonnegativeCone{T, T}
    JuMP.@constraint(model, [i in 1:inst.num_polys],
        polys[:, i] .- fpv in wsosT(U, Ps))

    return model
end
