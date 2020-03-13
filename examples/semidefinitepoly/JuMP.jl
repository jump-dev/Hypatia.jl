#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

test whether a given matrix has a SOS decomposition,
and use this procedure to check whether a polynomial is globally convex
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
include(joinpath(@__DIR__, "data.jl"))
import SumOfSquares
import PolyJuMP

struct SemidefinitePolyJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    x::Vector{<:DP.PolyVar}
    H::Matrix{<:DP.Polynomial}
    is_feas::Bool # whether model should be primal-dual feasible; only for testing
    use_wsosmatrix::Bool # use wsosinterppossemideftri cone, else PSD formulation
    use_dual::Bool # use dual formulation, else primal formulation
end
function SemidefinitePolyJuMP{Float64}(
    x::Vector{DP.PolyVar{true}},
    poly::DP.Polynomial,
    args...)
    return SemidefinitePolyJuMP{Float64}(x, DP.differentiate(poly, x, 2), args...)
end
function SemidefinitePolyJuMP{Float64}(
    matpoly::Symbol,
    args...)
    return SemidefinitePolyJuMP{Float64}(get_semidefinitepoly_data(matpoly)..., args...)
end

example_tests(::Type{SemidefinitePolyJuMP{Float64}}, ::MinimalInstances) = [
    ((:matpoly2, true, true), false),
    ((:matpoly5, true, true), false),
    ((:matpoly5, true, false), false),
    ((:matpoly5, false, false), false),
    ]
example_tests(::Type{SemidefinitePolyJuMP{Float64}}, ::FastInstances) = [
    ((:matpoly1, true, true), false),
    ((:matpoly1, true, false), false),
    ((:matpoly1, false, false), false),
    ((:matpoly2, true, true), false),
    ((:matpoly2, true, false), false),
    ((:matpoly2, false, false), false),
    ((:matpoly3, true, true), false),
    ((:matpoly3, true, false), false),
    ((:matpoly3, false, false), false),
    ((:matpoly4, true, true), false),
    ((:matpoly4, true, false), false),
    ((:matpoly4, false, false), false),
    ((:matpoly6, true, true), false),
    ((:matpoly6, true, false), false),
    ((:matpoly6, false, false), false),
    ((:matpoly7, true, true), false),
    ((:matpoly7, true, false), false),
    ((:matpoly7, false, false), false),
    ]
example_tests(::Type{SemidefinitePolyJuMP{Float64}}, ::SlowInstances) = [
    ]

function build(inst::SemidefinitePolyJuMP{T}) where {T <: Float64} # TODO generic reals
    (x, H) = (inst.x, inst.H)

    model = JuMP.Model()

    if inst.use_wsosmatrix
        side = size(H, 1)
        halfdeg = div(maximum(DP.maxdegree.(H)) + 1, 2)
        n = DP.nvariables(x)
        dom = ModelUtilities.FreeDomain{Float64}(n)
        (U, pts, Ps, _) = ModelUtilities.interpolate(dom, halfdeg)
        mat_wsos_cone = Hypatia.WSOSInterpPosSemidefTriCone{Float64}(side, U, Ps, inst.use_dual)

        rt2 = sqrt(2)
        H_svec = [H[i, j](pts[u, :]) for i in 1:side for j in 1:i for u in 1:U]
        ModelUtilities.vec_to_svec!(H_svec, rt2 = rt2, incr = U)
        if inst.use_dual
            JuMP.@variable(model, z[i in 1:side, 1:i, 1:U])
            z_svec = [1.0 * z[i, j, u] for i in 1:side for j in 1:i for u in 1:U]
            ModelUtilities.vec_to_svec!(z_svec, rt2 = rt2, incr = U)
            JuMP.@constraint(model, z_svec in mat_wsos_cone)
            JuMP.@objective(model, Min, dot(z_svec, H_svec))
        else
            JuMP.@constraint(model, H_svec in mat_wsos_cone)
        end
    else
        if inst.use_dual
            error("dual formulation not implemented for scalar SOS formulation")
        else
            PolyJuMP.setpolymodule!(model, SumOfSquares)
            JuMP.@constraint(model, H in JuMP.PSDCone())
        end
    end

    return model
end

function test_extra(inst::SemidefinitePolyJuMP, model)
    @test JuMP.termination_status(model) in (inst.is_feas ? (MOI.OPTIMAL,) : (MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE))
end

# @testset "SemidefinitePolyJuMP" for inst in example_tests(SemidefinitePolyJuMP{Float64}, MinimalInstances()) test(inst...) end

return SemidefinitePolyJuMP
