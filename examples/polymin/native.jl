#=
polynomial optimization problem
see:
D. Papp and S. Yildiz
Sum-of-squares optimization without semidefinite programming

real polynomials and real-valued complex polynomials

TODO
- implement PSD formulation for complex case
=#

include(joinpath(@__DIR__, "data_real.jl"))
include(joinpath(@__DIR__, "data_complex.jl"))

struct PolyMinNative{T <: Real} <: ExampleInstanceNative{T}
    is_complex::Bool
    interp_vals::Vector
    Ps::Vector{<:Matrix}
    true_min::Real
    use_primal::Bool # solve primal, else solve dual
    use_wsos::Bool # use wsosinterpnonnegative cone, else PSD formulation
end

function PolyMinNative{T}(
    is_complex::Bool,
    poly_name::Symbol,
    halfdeg::Int,
    args...) where {T <: Real}
    R = (is_complex ? Complex{T} : T)
    interp = get_interp_data(R, poly_name, halfdeg)
    return PolyMinNative{T}(is_complex, interp..., args...)
end

function PolyMinNative{T}(
    is_complex::Bool,
    n::Int,
    halfdeg::Int,
    args...) where {T <: Real}
    interp = random_interp_data(T, n, halfdeg)
    return PolyMinNative{T}(is_complex, interp..., args...)
end

function build(inst::PolyMinNative{T}) where {T <: Real}
    if inst.is_complex
        if !inst.use_wsos
            error("PSD formulation is not implemented yet")
        end
        return build_complex(inst)
    else
        if inst.use_primal && !inst.use_wsos
            error("primal psd formulation is not implemented yet")
        end
        return build_real(inst)
    end
end

function build_real(inst::PolyMinNative{T}) where {T <: Real}
    U = length(inst.interp_vals)

    cones = Cones.Cone{T}[]
    if inst.use_wsos
        push!(cones, Cones.WSOSInterpNonnegative{T, T}(U, inst.Ps, use_dual = !inst.use_primal))
    end

    if inst.use_primal
        c = T[-1]
        A = zeros(T, 0, 1)
        G = ones(T, U, 1)
        b = T[]
        h = inst.interp_vals
    else
        c = inst.interp_vals
        A = ones(T, 1, U) # can eliminate constraint and a variable
        b = T[1]
        if inst.use_wsos
            G = -one(T) * I
            h = zeros(T, U)
        else
            svec_lengths = [Cones.svec_length(size(Pk, 2)) for Pk in inst.Ps]
            G = zeros(T, sum(svec_lengths), U)
            offset = 0
            nonneg_cone_size = 0
            for (Pk, dk) in zip(inst.Ps, svec_lengths)
                Lk = size(Pk, 2)
                if dk == 1
                    nonneg_cone_size += 1
                else
                    if nonneg_cone_size > 0
                        push!(cones, Cones.Nonnegative{T}(nonneg_cone_size))
                    end
                    push!(cones, Cones.PosSemidefTri{T, T}(dk))
                end
                l = 1
                for i in 1:Lk, j in 1:i
                    @. @views G[offset + l, :] = -Pk[:, i] * Pk[:, j]
                    l += 1
                end
                @views Cones.scale_svec!(G[offset .+ (1:dk), :], sqrt(T(2)))
                offset += dk
            end
            if nonneg_cone_size > 0
                push!(cones, Cones.Nonnegative{T}(nonneg_cone_size))
            end
            h = zeros(T, size(G, 1))
        end
    end

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end

function build_complex(inst::PolyMinNative{T}) where {T <: Real}
    U = length(inst.interp_vals)

    if inst.use_primal
        c = T[-1]
        A = zeros(T, 0, 1)
        b = T[]
        G = ones(T, U, 1)
        h = inst.interp_vals
    else
        c = inst.interp_vals
        A = ones(T, 1, U)
        b = T[1]
        G = -one(T) * I
        h = zeros(T, U)
    end

    cones = Cones.Cone{T}[Cones.WSOSInterpNonnegative{T, Complex{T}}(U, inst.Ps, use_dual = !inst.use_primal)]

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end

function test_extra(inst::PolyMinNative{T}, solve_stats::NamedTuple, ::NamedTuple) where T
    @test solve_stats.status == Solvers.Optimal
    if solve_stats.status == Solvers.Optimal && !isnan(inst.true_min)
        # check objective value is correct
        tol = eps(T)^0.1
        true_min = (inst.use_primal ? -1 : 1) * inst.true_min
        @test solve_stats.primal_obj ≈ true_min atol=tol rtol=tol
    end
end
