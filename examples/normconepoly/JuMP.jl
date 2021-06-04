#=
check a sufficient condition for pointwise membership of vector valued
polynomials in the epinormone/epinormeucl cone
=#

struct NormConePoly{T <: Real} <: ExampleInstanceJuMP{T}
    polys_name::Symbol
    deg::Int
    is_feas::Bool # whether model should be primal-dual feasible; only for testing
    use_L2::Bool # check for membership in epigraph of L2, otherwise use L1
end

function build(inst::NormConePoly{T}) where {T <: Float64}
    halfdeg = div(inst.deg + 1, 2)
    (U, pts, Ps) = PolyUtils.interpolate(
        PolyUtils.FreeDomain{T}(1), halfdeg)
    vals = normconepoly_data[inst.polys_name].(pts)
    l = length(vals[1])
    if inst.use_L2
        cone = Hypatia.WSOSInterpEpiNormEuclCone{T}(l, U, Ps)
    else
        cone = Hypatia.WSOSInterpEpiNormOneCone{T}(l, U, Ps)
    end

    model = JuMP.Model()
    JuMP.@constraint(model, [v[i] for i in 1:l for v in vals] in cone)

    return model
end

function test_extra(inst::NormConePoly{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == (inst.is_feas ? MOI.OPTIMAL : MOI.INFEASIBLE)
end

normconepoly_data = Dict(
    :polys1 => (x -> [x^2 + 2, x]),
    :polys2 => (x -> [2x^2 + 2, x, x]),
    :polys3 => (x -> [x^2 + 2, x, x]),
    :polys4 => (x -> [2 * x^4 + 8 * x^2 + 4, x + 2 + (x + 1)^2, x]),
    :polys5 => (x -> [x, x^2 + x]),
    :polys6 => (x -> [x, x + 1]),
    :polys7 => (x -> [x^2, x]),
    :polys8 => (x -> [x + 2, x]),
    :polys9 => (x -> [x - 1, x, x]),
    )
