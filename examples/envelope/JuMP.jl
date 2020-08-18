#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl
=#

struct EnvelopeJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    rand_halfdeg::Int
    num_polys::Int
    env_halfdeg::Int
    use_norm_constraint::Bool
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
    if inst.use_norm_constraint
        JuMP.@objective(model, Min, dot(fpv, w)) # integral over domain (via quadrature)
        if inst.use_wsosinterpepinormeucl
            R = inst.num_polys + 1
            JuMP.@constraint(model, vcat(fpv, [polys[:, i] for i in 1:inst.num_polys]...) in Hypatia.WSOSInterpEpiNormEuclCone{Float64}(R, U, Ps))
        elseif inst.use_wsosinterppossemideftri
            svec_dim = div(R * (R + 1), 2)
            polyvec = Vector{JuMP.AffExpr}(undef, svec_dim * U)
            polyvec[1:U] .= fpv
            idx = 2
            for j in 2:R
                polyvec[Cones.block_idxs(U, idx)] .= polys[:, (j - 1)] * sqrt(2)
                idx += 1
                for i in 2:(j - 1)
                    polyvec[Cones.block_idxs(U, idx)] .= 0
                    idx += 1
                end
                polyvec[Cones.block_idxs(U, idx)] .= fpv
                idx += 1
            end
            JuMP.@constraint(model, polyvec in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(R, U, Ps))
        end
    else
        JuMP.@objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
        JuMP.@constraint(model, [i in 1:inst.num_polys], polys[:, i] .- fpv in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
    end

    return model
end

instances[EnvelopeJuMP]["minimal"] = [
    ((1, 2, 2, 2, false),),
    ((1, 2, 2, 2, true),),
    ]
instances[EnvelopeJuMP]["fast"] = [
    ((2, 2, 3, 2, false),),
    ((3, 3, 3, 3, false),),
    ((3, 3, 5, 4, false),),
    ((5, 2, 5, 3, false),),
    ((1, 30, 2, 30, false),),
    ((10, 1, 3, 1, false),),
    ((2, 2, 3, 2, true),),
    ((3, 3, 3, 3, true),),
    ((3, 3, 5, 4, true),),
    ((5, 2, 5, 3, true),),
    ((1, 30, 2, 30, true),),
    ((10, 1, 3, 1, true),),
    ]
instances[EnvelopeJuMP]["slow"] = [
    ((4, 5, 4, 6, false),),
    ((2, 30, 4, 30, false),),
    ((4, 5, 4, 6, true),),
    ((2, 30, 4, 30, true),),
    ]
