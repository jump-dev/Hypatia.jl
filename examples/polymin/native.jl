#=
polymin real: formulates and solves the real polynomial optimization problem for a given polynomial; see:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming.

polymin complex: minimizes a real-valued complex polynomial over a domain defined by real-valued complex polynomials

TODO
- generalize ModelUtilities interpolation code for complex polynomials space
- merge real and complex polyvars data
- implement PSD formulation for complex case
=#

import Hypatia.BlockMatrix
import Combinatorics
include(joinpath(@__DIR__, "data.jl"))

struct PolyMinNative{T <: Real} <: ExampleInstanceNative{T}
    is_complex::Bool
    interp_vals::Vector
    Ps::Vector{<:Matrix}
    true_min::Real
    use_primal::Bool # solve primal, else solve dual
    use_wsos::Bool # use wsosinterpnonnegative cone, else PSD formulation
    use_linops::Bool
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
        if inst.use_linops
            A = BlockMatrix{T}(0, 1, Any[[]], [0:-1], [1:1])
            G = BlockMatrix{T}(U, 1, [ones(T, U, 1)], [1:U], [1:1])
        else
            A = zeros(T, 0, 1)
            G = ones(T, U, 1)
        end
        b = T[]
        h = inst.interp_vals
    else
        c = inst.interp_vals
        if inst.use_linops
            A = BlockMatrix{T}(1, U, [ones(T, 1, U)], [1:1], [1:U])
        else
            A = ones(T, 1, U) # NOTE can eliminate constraint and a variable
        end
        b = T[1]
        if inst.use_wsos
            if inst.use_linops
                G = BlockMatrix{T}(U, U, [-I], [1:U], [1:U])
            else
                G = -one(T) * I
            end
            h = zeros(T, U)
        else
            svec_lengths = [Cones.svec_length(size(Pk, 2)) for Pk in inst.Ps]
            G_full = zeros(T, sum(svec_lengths), U)
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
                    @. @views G_full[offset + l, :] = -Pk[:, i] * Pk[:, j]
                    l += 1
                end
                @views ModelUtilities.vec_to_svec!(G_full[(offset + 1):(offset + dk), :], rt2 = sqrt(T(2)))
                offset += dk
            end
            if nonneg_cone_size > 0
                push!(cones, Cones.Nonnegative{T}(nonneg_cone_size))
            end
            if inst.use_linops
                (nrows, ncols) = size(G_full)
                G = BlockMatrix{T}(nrows, ncols, [G_full], [1:nrows], [1:ncols])
            else
                G = G_full
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

function test_extra(inst::PolyMinNative{T}, result::NamedTuple) where T
    @test result.status == :Optimal
    if result.status == :Optimal && !isnan(inst.true_min)
        # check objective value is correct
        tol = eps(T)^0.1
        true_min = (inst.use_primal ? -1 : 1) * inst.true_min
        @test result.primal_obj ≈ true_min atol = tol rtol = tol
    end
end
