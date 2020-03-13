#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

polymin real: formulates and solves the real polynomial optimization problem for a given polynomial; see:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming.

polymin complex: minimizes a real-valued complex polynomial over a domain defined by real-valued complex polynomials

TODO
- generalize ModelUtilities interpolation code for complex polynomials space
- merge real and complex polyvars data
- implement PSD formulation for complex case
=#

include(joinpath(@__DIR__, "../common_native.jl"))
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

options = ()
example_tests(::Type{PolyMinNative{Float64}}, ::MinimalInstances) = [
    ((false, 1, 2, true, true, false), options),
    ((false, 1, 2, false, true, false), options),
    ((false, 1, 2, false, false, false), options),
    ((true, :abs1d, 1, true, true, false), options),
    ((true, :abs1d, 1, false, true, false), options),
    ((false, :rosenbrock, 4, true, true, false), options),
    ]
example_tests(::Type{PolyMinNative{Float64}}, ::FastInstances) = [
    ((false, 1, 3, true, true, false), options),
    ((false, 1, 30, true, true, false), options),
    ((false, 1, 30, false, true, false), options),
    ((false, 1, 30, false, false, false), options),
    ((false, 2, 8, true, true, false), options),
    ((false, 3, 6, true, true, false), options),
    ((false, 5, 3, true, true, false), options),
    ((false, 10, 1, true, true, false), options),
    ((false, 10, 1, false, true, false), options),
    ((false, 10, 1, false, false, false), options),
    ((false, 4, 4, true, true, false), options),
    ((false, 4, 4, false, true, false), options),
    ((false, 4, 4, false, false, false), options),
    ((true, :abs1d, 3, true, true, false), options),
    ((true, :absunit1d, 1, true, true, false), options),
    ((true, :absunit1d, 3, true, true, false), options),
    ((true, :negabsunit1d, 2, true, true, false), options),
    ((true, :absball2d, 1, true, true, false), options),
    ((true, :absbox2d, 2, true, true, false), options),
    ((true, :negabsbox2d, 1, true, true, false), options),
    ((true, :denseunit1d, 2, true, true, false), options),
    ((true, :negabsunit1d, 2, false, true, false), options),
    ((true, :absball2d, 1, false, true, false), options),
    ((true, :negabsbox2d, 1, false, true, false), options),
    ((true, :denseunit1d, 2, false, true, false), options),
    ((false, :butcher, 2, true, true, false), options),
    ((false, :caprasse, 4, true, true, false), options),
    ((false, :goldsteinprice, 7, true, true, false), options),
    ((false, :goldsteinprice_ball, 6, true, true, false), options),
    ((false, :goldsteinprice_ellipsoid, 7, true, true, false), options),
    ((false, :heart, 2, true, true, false), options),
    ((false, :lotkavolterra, 3, true, true, false), options),
    ((false, :magnetism7, 2, true, true, false), options),
    ((false, :magnetism7_ball, 2, true, true, false), options),
    ((false, :motzkin, 3, true, true, false), options),
    ((false, :motzkin_ball, 3, true, true, false), options),
    ((false, :motzkin_ellipsoid, 3, true, true, false), options),
    ((false, :reactiondiffusion, 4, true, true, false), options),
    ((false, :robinson, 8, true, true, false), options),
    ((false, :robinson_ball, 8, true, true, false), options),
    ((false, :rosenbrock, 5, true, true, false), options),
    ((false, :rosenbrock_ball, 5, true, true, false), options),
    ((false, :schwefel, 2, true, true, false), options),
    ((false, :schwefel_ball, 2, true, true, false), options),
    ((false, :lotkavolterra, 3, false, true, false), options),
    ((false, :motzkin, 3, false, true, false), options),
    ((false, :motzkin_ball, 3, false, true, false), options),
    ((false, :schwefel, 2, false, true, false), options),
    ((false, :lotkavolterra, 3, false, false, false), options),
    ((false, :motzkin, 3, false, false, false), options),
    ((false, :motzkin_ball, 3, false, false, false), options),
    ]
example_tests(::Type{PolyMinNative{Float64}}, ::SlowInstances) = [
    ((false, 4, 5, true, true, false), options),
    ((false, 4, 5, false, true, false), options),
    ((false, 4, 5, false, false, false), options),
    ((false, 2, 30, true, true, false), options),
    ((false, 2, 30, false, true, false), options),
    ((false, 2, 30, false, false, false), options),
    ]
example_tests(::Type{PolyMinNative{Float64}}, ::LinearOperatorsInstances) = [
    ((false, 1, 8, true, true, true), options),
    ((false, 2, 5, true, true, true), options),
    ((false, 3, 3, true, true, true), options),
    ((false, 5, 2, true, true, true), options),
    ((false, 3, 3, false, true, true), options),
    ((false, 3, 3, false, false, true), options),
    ((false, :butcher, 2, true, true, true), options),
    ((false, :caprasse, 4, true, true, true), options),
    ((false, :goldsteinprice, 7, true, true, true), options),
    ]

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
                G = Diagonal(-one(T) * I, U)
            end
            h = zeros(T, U)
        else
            svec_lengths = [Cones.svec_length(size(Pk, 2)) for Pk in inst.Ps]
            G_full = zeros(T, sum(svec_lengths), U)
            offset = 0
            for (Pk, dk) in zip(inst.Ps, svec_lengths)
                Lk = size(Pk, 2)
                push!(cones, Cones.PosSemidefTri{T, T}(dk))
                l = 1
                for i in 1:Lk, j in 1:i
                    @. @views G_full[offset + l, :] = -Pk[:, i] * Pk[:, j]
                    l += 1
                end
                @views ModelUtilities.vec_to_svec!(G_full[(offset + 1):(offset + dk), :], rt2 = sqrt(T(2)))
                offset += dk
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
        G = Diagonal(-one(T) * I, U)
        h = zeros(T, U)
    end

    cones = Cones.Cone{T}[Cones.WSOSInterpNonnegative{T, Complex{T}}(U, inst.Ps, use_dual = !inst.use_primal)]

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end

function test_extra(inst::PolyMinNative{T}, result) where T
    @test result.status == :Optimal
    if result.status == :Optimal && !isnan(inst.true_min)
        # check objective value is correct
        tol = eps(T)^0.2
        true_min = (inst.use_primal ? -1 : 1) * inst.true_min
        @test result.primal_obj ≈ true_min atol = tol rtol = tol
    end
end

# @testset "PolyMinNative" for inst in example_tests(PolyMinNative{T}, MinimalInstances()) test(inst...) end

return PolyMinNative
