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
    if inst.use_l1
        if inst.use_norm_cone
            JuMP.@constraint(model, vcat(f, vec(polys)) in Hypatia.WSOSInterpEpiNormOneCone{Float64}(num_polys + 1, U, Ps))
        else
            p_plus = []
            p_minus = []
            for i in 1:inst.num_polys
                p_plus_i = JuMP.@variable(model, [1:U])
                p_minus_i = JuMP.@variable(model, [1:U])
                push!(p_plus, p_plus_i)
                push!(p_minus, p_minus_i)
                JuMP.@constraints(model, begin
                    polys[:, i] .== p_plus_i - p_minus_i
                    p_plus_i in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps)
                    p_minus_i in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps)
                end)
            end
            JuMP.@constraint(model, f - sum(p_plus) - sum(p_minus) in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
        end
    else
        if inst.use_norm_cone
            JuMP.@constraint(model, vcat(f, vec(polys)) in Hypatia.WSOSInterpEpiNormEuclCone{Float64}(num_polys + 1, U, Ps))
        else
            R = num_polys + 1
            svec_dim = div(R * (R + 1), 2)
            polyvec = Vector{JuMP.AffExpr}(undef, svec_dim * U)
            # construct vectorized arrow matrix polynomial
            polyvec[1:U] .= f
            idx = 2
            for j in 2:R
                polyvec[Cones.block_idxs(U, idx)] .= polys[:, (j - 1)] * sqrt(2)
                idx += 1
                for i in 2:(j - 1)
                    polyvec[Cones.block_idxs(U, idx)] .= 0
                    idx += 1
                end
                polyvec[Cones.block_idxs(U, idx)] .= f
                idx += 1
            end
            JuMP.@constraint(model, polyvec in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(R, U, Ps))
        end
    end

    return model
end
