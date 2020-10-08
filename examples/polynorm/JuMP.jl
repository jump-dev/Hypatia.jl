#=
find a polynomial f such that f² >= Σᵢ gᵢ² where gᵢ are arbitrary polynomials, and the volume under f is minimized

TODO add scalar SOS formulation
=#

struct PolyNormJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    deg::Int
    num_polys::Int
end

function build(inst::PolyNormJuMP{T}) where {T <: Float64}
    (n, num_polys) = (inst.n, inst.num_polys)

    dom = ModelUtilities.FreeDomain{Float64}(n)
    halfdeg = div(inst.deg + 1, 2)
    (U, pts, Ps, _, w) = ModelUtilities.interpolate(dom, halfdeg, calc_w = true)
    polys = Ps[1] * rand(-9:9, size(Ps[1], 2), num_polys)

    model = JuMP.Model()
    JuMP.@variable(model, f[1:U])
    JuMP.@objective(model, Min, dot(w, f))
    JuMP.@constraint(model, vcat(f, vec(polys)) in Hypatia.WSOSInterpEpiNormEuclCone{Float64}(num_polys + 1, U, Ps))

    return model
end
