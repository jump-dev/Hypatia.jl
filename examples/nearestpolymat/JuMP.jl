#=
given a random polynomial matrix H, find a polynomial matrix Q such that Q - H is SOS-PSD over a unit box
and the total volume of all polynomials in Q over the unit box is minimized
=#

struct NearestPolyMatJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    halfdeg::Int
    R::Int # side dimension of polynomial matrices
    use_wsos::Bool # use WSOS cone with auxiliary polynomial
    use_matrixwsos::Bool # use WSOS PSD cone
    use_sdp::Bool # use SDP formulation with auxiliary polynomial
end

function build(inst::NearestPolyMatJuMP{T}) where {T <: Float64}
    (n, halfdeg, R) = (inst.n, inst.halfdeg, inst.R)
    @assert inst.use_wsos + inst.use_matrixwsos + inst.use_sdp == 1
    U = binomial(n + 2 * halfdeg, n)
    L = binomial(n + halfdeg, n)
    svec_dim = div(R * (R + 1), 2)

    domain = ModelUtilities.Box{T}(-ones(T, n), ones(T, n))
    (U, points, Ps, V, w) = ModelUtilities.interpolate(domain, halfdeg, calc_V = true, calc_w = true)

    model = JuMP.Model()
    JuMP.@variable(model, q_poly[1:(U * svec_dim)])

    # generate random polynomial matrix H and symmetrize
    full_coeffs = [Vector{JuMP.AffExpr}(undef, U) for _ in 1:R, _ in 1:R]
    idx = 1
    for j in 1:R
        for i in 1:(j - 1)
            full_coeffs[i, j] = full_coeffs[j, i] = V * rand(-9:9, U) - q_poly[Cones.block_idxs(U, idx)]
            idx += 1
        end
        full_coeffs[j, j] = V * rand(-9:9, U) - q_poly[Cones.block_idxs(U, idx)]
        idx += 1
    end

    if inst.use_matrixwsos
        JuMP.@constraint(model, vcat([full_coeffs[i, j] * (i == j ? 1 : sqrt(2)) for j in 1:R for i in 1:j]...) in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(R, U, Ps))
    elseif inst.use_wsos
        ypts = zeros(svec_dim, R)
        idx = 1
        for j in 1:R
            for i in 1:(j - 1)
                full_coeffs[i, j] .*= 2
                full_coeffs[i, j] .+= full_coeffs[i, i] + full_coeffs[j, j]
                ypts[idx, i] = ypts[idx, j] = 1
                idx += 1
            end
            ypts[idx, j] = 1
            idx += 1
        end
        new_Ps = Matrix{Float64}[]
        for P in Ps
            push!(new_Ps, kron(ypts, P))
        end
        JuMP.@constraint(model, vcat([full_coeffs[i, j] for j in 1:R for i in 1:j]...) in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U * svec_dim, new_Ps))
    else
        JuMP.@variable(model, psd_var[1:(L * R), 1:(L * R)], PSD)
        for x2 in 1:R, x1 in 1:x2
            coeffs_lhs = JuMP.@expression(model, [u in 1:U], sum(Ps[1][u, k] * Ps[1][u, l] * psd_var[(x1 - 1) * L + k, (x2 - 1) * L + l] for k in 1:L for l in 1:L))
            JuMP.@constraint(model, coeffs_lhs .== full_coeffs[x1, x2])
        end
    end
    JuMP.@objective(model, Max, sum(dot(w, q_poly[Cones.block_idxs(U, Cones.svec_idx(i, j))]) * (i == j ? 1 : 2) for i in 1:R for j in 1:i))

    return model
end
