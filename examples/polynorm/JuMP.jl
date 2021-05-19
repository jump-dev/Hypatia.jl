#=
find a polynomial f such that f² >= Σᵢ gᵢ² or f >= Σᵢ |gᵢ|
where gᵢ are arbitrary polynomials, and the volume under f is minimized
=#

struct PolyNormJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    rand_halfdeg::Int # maximum degree of randomly generated polynomials
    epi_halfdeg::Int # half of max degree of cone polynomials
    num_polys::Int
    use_l1::Bool # use epigraph of one norm, otherwise Euclidean norm
    use_norm_cone::Bool # use Euclidean / L1 norm WSOS cones
    use_wsos_scalar::Bool # use WSOS scalar cone
end

function build(inst::PolyNormJuMP{T}) where {T <: Float64}
    (n, num_polys, epi_halfdeg, rand_halfdeg) =
        (inst.n, inst.num_polys, inst.epi_halfdeg, inst.rand_halfdeg)
    @assert epi_halfdeg >= rand_halfdeg

    dom = PolyUtils.BoxDomain{T}(-ones(T, n), ones(T, n))
    rand_U = binomial(n + 2 * rand_halfdeg, n)
    rand_coeffs = rand(-9:9, rand_U, num_polys)
    (U, pts, Ps, V, w) = PolyUtils.interpolate(dom, epi_halfdeg,
        calc_V = true, get_quadr = true)
    polys = V[:, 1:rand_U] * rand_coeffs

    model = JuMP.Model()
    JuMP.@variable(model, f[1:U])
    JuMP.@objective(model, Min, dot(w, f))

    if inst.use_norm_cone
        cone = (inst.use_l1 ? Hypatia.WSOSInterpEpiNormOneCone :
            Hypatia.WSOSInterpEpiNormEuclCone)
        JuMP.@constraint(model, vcat(f, vec(polys)) in
            cone{T}(num_polys + 1, U, Ps))
    elseif inst.use_l1
        lhs = one(T) * f
        for i in 1:inst.num_polys
            p_plus_i = JuMP.@variable(model, [1:U])
            p_minus_i = JuMP.@variable(model, [1:U])
            lhs .-= p_plus_i
            lhs .-= p_minus_i
            JuMP.@constraints(model, begin
                polys[:, i] .== p_plus_i - p_minus_i
                p_plus_i in Hypatia.WSOSInterpNonnegativeCone{T, T}(U, Ps)
                p_minus_i in Hypatia.WSOSInterpNonnegativeCone{T, T}(U, Ps)
            end)
        end
        JuMP.@constraint(model, lhs in
            Hypatia.WSOSInterpNonnegativeCone{T, T}(U, Ps))
    else
        R = num_polys + 1
        svec_dim = Cones.svec_length(R)
        polyvec = zeros(JuMP.AffExpr, svec_dim * U)
        polyvec[1:U] .= f

        if !inst.use_wsos_scalar
            idx = 2
            rt2 = sqrt(T(2))
            for i in 1:inst.num_polys
                polyvec[Cones.block_idxs(U, idx)] = rt2 * polys[:, i]
                idx += i
                polyvec[Cones.block_idxs(U, idx)] = f
                idx += 1
            end
            cone = Hypatia.WSOSInterpPosSemidefTriCone{T}(R, U, Ps)
        else
            ypts = zeros(svec_dim, R)
            ypts[1, 1] = 1
            idx = 2
            for j in 2:R
                polyvec[Cones.block_idxs(U, idx)] = 2 * (polys[:, j - 1] + f)
                ypts[idx, 1] = ypts[idx, j] = 1
                idx += 1
                for i in 2:(j - 1)
                    polyvec[Cones.block_idxs(U, idx)] = 2 * f
                    ypts[idx, i] = ypts[idx, j] = 1
                    idx += 1
                end
                polyvec[Cones.block_idxs(U, idx)] = f
                ypts[idx, j] = 1
                idx += 1
            end
            new_Ps = Matrix{T}[]
            for P in Ps
                push!(new_Ps, kron(ypts, P))
            end
            cone = Hypatia.WSOSInterpNonnegativeCone{T, T}(U * svec_dim, new_Ps)
        end
        JuMP.@constraint(model, polyvec in cone)
    end

    return model
end
