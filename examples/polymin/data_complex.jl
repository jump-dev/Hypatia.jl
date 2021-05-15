#=
real-valued complex polynomials
=#

import Hypatia.PolyUtils

# get interpolation for a predefined complex poly
function get_interp_data(
    R::Type{Complex{T}},
    poly_name::Symbol,
    halfdeg::Int,
    ) where {T <: Real}
    (n, f, gs, g_halfdegs, true_min) = complex_poly_data[poly_name]
    (points, Ps) = PolyUtils.interpolate(R, halfdeg, n, gs, g_halfdegs)
    interp_vals = f.(points)
    return (interp_vals, Ps, true_min)
end

# merge with real polys when complex polyvars are allowed in DynamicPolynomials:
# real-valued complex polynomials
complex_poly_data = Dict{Symbol, NamedTuple}(
    :abs1d => (n = 1,
        f = (z -> 1 + sum(abs2, z)),
        gs = Function[],
        g_halfdegs = Int[],
        true_min = 1,
        ),
    :absunit1d => (n = 1,
        f = (z -> 1 + sum(abs2, z)),
        gs = [z -> 1 - sum(abs2, z)],
        g_halfdegs = [1],
        true_min = 1,
        ),
    :negabsunit1d => (n = 1,
        f = (z -> -sum(abs2, z)),
        gs = [z -> 1 - sum(abs2, z)],
        g_halfdegs = [1],
        true_min = -1,
        ),
    :absball2d => (n = 2,
        f = (z -> 1 + sum(abs2, z)),
        gs = [z -> 1 - sum(abs2, z)],
        g_halfdegs = [1],
        true_min = 1,
        ),
    :absbox2d => (n = 2,
        f = (z -> 1 + sum(abs2, z)),
        gs = [z -> 1 - abs2(z[1]), z -> 1 - abs2(z[2])],
        g_halfdegs = [1, 1],
        true_min = 1,
        ),
    :negabsbox2d => (n = 2,
        f = (z -> -sum(abs2, z)),
        gs = [z -> 1 - abs2(z[1]), z -> 1 - abs2(z[2])],
        g_halfdegs = [1, 1],
        true_min = -2,
        ),
    :denseunit1d => (n = 1,
        f = (z -> 1 + 2real(z[1]) + abs(z[1])^2 + 2real(z[1]^2) +
            2real(z[1]^2 * conj(z[1])) + abs(z[1])^4),
        gs = [z -> 1 - abs2(z[1])],
        g_halfdegs = [1],
        true_min = 0,
        ),
    )
