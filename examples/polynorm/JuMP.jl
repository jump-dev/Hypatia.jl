#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find a polynomial f such that f² >= Σᵢ gᵢ² where gᵢ are arbitrary polynomials, and the volume under f is minimized

TODO add scalar SOS formulation
=#

struct PolyNormJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    deg::Int
    num_polys::Int
end

function build(inst::PolyNormJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, num_polys) = (inst.n, inst.num_polys)

    dom = ModelUtilities.FreeDomain{Float64}(n)
    halfdeg = div(inst.deg + 1, 2)
    (U, pts, Ps, _, w) = ModelUtilities.interpolate(dom, halfdeg, calc_w = true)
    lagrange_polys = ModelUtilities.recover_lagrange_polys(pts, 2 * halfdeg)

    model = JuMP.Model()
    JuMP.@variable(model, f[1:U])
    JuMP.@objective(model, Min, dot(w, f))

    L = size(Ps[1], 2)
    polys = Ps[1] * rand(-9:9, L, num_polys)

    fpoly = dot(f, lagrange_polys)
    rand_polys = [dot(polys[:, i], lagrange_polys) for i in 1:num_polys]
    cone = Hypatia.WSOSInterpEpiNormEuclCone{Float64}(num_polys + 1, U, Ps)
    JuMP.@constraint(model, vcat(f, [polys[:, i] for i in 1:num_polys]...) in cone)

    return model
end

instances[PolyNormJuMP]["minimal"] = [
    ((1, 2, 2),),
    ]
instances[PolyNormJuMP]["fast"] = [
    ((2, 2, 2),),
    ((2, 1, 3),),
    ]
instances[PolyNormJuMP]["slow"] = Tuple[]
