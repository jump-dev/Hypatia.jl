#=
find a polynomial f such that f² >= Σᵢ gᵢ² or f >= Σᵢ |gᵢ| where gᵢ are arbitrary polynomials, and the volume under f is minimized
=#

struct PolyNormJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    deg::Int
    num_polys::Int
    use_l1::Bool # use epigraph of one norm, otherwise Euclidean norm
    use_norm_cone::Bool # use Euclidean / one norm cone, otherwise use WSOS matrix / WSOS cones
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

    if inst.use_norm_cone
        cone = (inst.use_l1 ? Hypatia.WSOSInterpEpiNormOneCone : Hypatia.WSOSInterpEpiNormEuclCone)
        JuMP.@constraint(model, vcat(f, vec(polys)) in cone{Float64}(num_polys + 1, U, Ps))
    elseif inst.use_l1
        lhs = one(T) * f
        for i in 1:inst.num_polys
            p_plus_i = JuMP.@variable(model, [1:U])
            p_minus_i = JuMP.@variable(model, [1:U])
            lhs .-= p_plus_i
            lhs .-= p_minus_i
            JuMP.@constraints(model, begin
                polys[:, i] .== p_plus_i - p_minus_i
                p_plus_i in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps)
                p_minus_i in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps)
            end)
        end
        JuMP.@constraint(model, lhs in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
    else
        R = num_polys + 1
        polyvec = zeros(JuMP.AffExpr, div(R * (R + 1), 2) * U)
        # construct vectorized arrow matrix polynomial
        polyvec[1:U] .= f
        idx = 2
        rt2 = sqrt(T(2))
        for i in 1:inst.num_polys
            polyvec[Cones.block_idxs(U, idx)] .= rt2 * polys[:, i]
            idx += i
            polyvec[Cones.block_idxs(U, idx)] .= f
            idx += 1
        end
        JuMP.@constraint(model, polyvec in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(R, U, Ps))
    end

    return model
end
