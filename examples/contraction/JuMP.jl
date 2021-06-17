#=
contraction analysis example adapted from
"Stability and robustness analysis of nonlinear systems via contraction metrics
and SOS programming" by Aylward, E.M., Parrilo, P.A. and Slotine, J.J.E
=#

import DynamicPolynomials
const DP = DynamicPolynomials
import Combinatorics
import PolyJuMP
import SumOfSquares

struct ContractionJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    beta::Real
    M_deg::Int
    delta::Real
    use_matrixwsos::Bool # use wsos matrix cone, else PSD formulation
    is_feas::Bool
end

function build(inst::ContractionJuMP{T}) where {T <: Float64}
    delta = inst.delta
    n = 2
    dom = PolyUtils.FreeDomain{T}(n)

    M_halfdeg = div(inst.M_deg + 1, 2)
    (U_M, pts_M, Ps_M) = PolyUtils.interpolate(dom, M_halfdeg)
    lagrange_polys = get_lagrange_polys(pts_M, 2 * M_halfdeg)
    x = DP.variables(lagrange_polys)

    # dynamics according to the Moore-Greitzer model
    dx1dt = -x[2] - 1.5 * x[1]^2 - 0.5 * x[1]^3
    dx2dt = 3 * x[1] - x[2]
    dynamics = [dx1dt; dx2dt]

    model = JuMP.Model()
    JuMP.@variable(model, polys[1:3], PolyJuMP.Poly(
        PolyJuMP.MultivariateBases.FixedPolynomialBasis(lagrange_polys)))

    M = [polys[1] polys[2]; polys[2] polys[3]]
    dMdt = [dot(DP.differentiate(M[i, j], x), dynamics) for i in 1:n, j in 1:n]
    dfdx = DP.differentiate(dynamics, x)'
    Mdfdx = [sum(M[i, k] * dfdx[k, j] for k in 1:n) for i in 1:n, j in 1:n]
    R = Mdfdx + Mdfdx' + dMdt + inst.beta * M

    if inst.use_matrixwsos
        deg_R = maximum(DP.maxdegree.(R))
        d_R = div(deg_R + 1, 2)
        (U_R, pts_R, Ps_R) = PolyUtils.interpolate(dom, d_R)
        M_gap = [M[i, j](pts_M[u, :]) - (i == j ? delta : 0.0)
            for i in 1:n for j in 1:i for u in 1:U_M]
        R_gap = [-R[i, j](pts_R[u, :]) - (i == j ? delta : 0.0)
            for i in 1:n for j in 1:i for u in 1:U_R]

        wsosmatT = Hypatia.WSOSInterpPosSemidefTriCone{T}
        rt2 = sqrt(2)
        M_scal = Cones.scale_svec!(M_gap, rt2, incr = U_M)
        R_scal = Cones.scale_svec!(R_gap, rt2, incr = U_R)
        JuMP.@constraint(model, M_scal in wsosmatT(n, U_M, Ps_M))
        JuMP.@constraint(model, R_scal in wsosmatT(n, U_R, Ps_R))
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, M - Matrix(delta * I, n, n) in JuMP.PSDCone())
        JuMP.@constraint(model, -R - Matrix(delta * I, n, n) in JuMP.PSDCone())
    end

    return model
end

function test_extra(inst::ContractionJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == (inst.is_feas ? MOI.OPTIMAL : MOI.INFEASIBLE)
    return
end

# TODO below functions used for recovering lagrange polys are not numerically
# stable for high degree (since they essentially calculate 2^degree);
# may be unnecessary if already doing a QR of the U*U Vandermonde,
# so better to replace them with linear algebra without using DP
function get_lagrange_polys(pts::Matrix{T}, deg::Int) where {T <: Real}
    if deg > 8
        @warn("get_lagrange_polys is not numerically stable for large degree")
    end
    (U, n) = size(pts)
    DP.@polyvar x[1:n]
    basis = get_chebyshev_polys(x, deg)
    @assert length(basis) == U
    vand_inv = inv([basis[j](x => view(pts, i, :)) for i in 1:U, j in 1:U])
    lagrange_polys = [dot(view(vand_inv, :, i), basis) for i in 1:U]
    return lagrange_polys
end

# returns the multivariate Chebyshev polynomials in x up to degree deg
function get_chebyshev_polys(x::Vector{DP.PolyVar{true}}, deg::Int)
    if deg > 8
        @warn("get_chebyshev_polys is not numerically stable for large degree")
    end
    n = length(x)
    u = get_chebyshev_univ(x, deg)
    V = Vector{DP.Polynomial{true, Int}}(undef, PolyUtils.get_L(n, deg))
    V[1] = DP.Monomial(1)
    col = 1
    for t in 1:deg, xp in Combinatorics.multiexponents(n, t)
        col += 1
        V[col] = u[1][xp[1] + 1]
        for j in 2:n
            V[col] *= u[j][xp[j] + 1]
        end
    end
    return V
end

function get_chebyshev_univ(monovec::Vector{DP.PolyVar{true}}, deg::Int)
    if deg > 8
        @warn("get_chebyshev_univ is not numerically stable for large degree")
    end
    n = length(monovec)
    u = Vector{Vector}(undef, n)
    for j in 1:n
        uj = u[j] = Vector{DP.Polynomial{true, Int}}(undef, deg + 1)
        uj[1] = DP.Monomial(1)
        uj[2] = monovec[j]
        for t in 3:(deg + 1)
            uj[t] = 2 * uj[2] * uj[t - 1] - uj[t - 2]
        end
    end
    return u
end
