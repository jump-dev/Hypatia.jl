#=
test whether a given matrix has a SOS decomposition,
and use this procedure to check whether a polynomial is globally convex
=#

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
    return SemidefinitePolyJuMP{Float64}(get_psdpoly_data(matpoly)..., args...)
end

function build(inst::SemidefinitePolyJuMP{T}) where {T <: Float64}
    (x, H) = (inst.x, inst.H)

    model = JuMP.Model()

    if inst.use_wsosmatrix
        side = size(H, 1)
        halfdeg = div(maximum(DP.maxdegree.(H)) + 1, 2)
        n = DP.nvariables(x)
        dom = PolyUtils.FreeDomain{T}(n)
        (U, pts, Ps) = PolyUtils.interpolate(dom, halfdeg)
        mat_wsos_cone = Hypatia.WSOSInterpPosSemidefTriCone{T}(
            side, U, Ps, inst.use_dual)

        rt2 = sqrt(T(2))
        H_svec = [H[i, j](pts[u, :]) for i in 1:side for j in 1:i for u in 1:U]
        Cones.scale_svec!(H_svec, rt2, incr = U)
        if inst.use_dual
            JuMP.@variable(model, z[i in 1:side, 1:i, 1:U])
            z_svec = [1.0 * z[i, j, u] for i in 1:side for j in 1:i for u in 1:U]
            Cones.scale_svec!(z_svec, rt2, incr = U)
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

function test_extra(inst::SemidefinitePolyJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) in
        (inst.is_feas ? (MOI.OPTIMAL,) : (MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE))
end
