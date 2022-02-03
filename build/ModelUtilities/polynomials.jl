#=
utilities for constructing some useful types of polynomials

TODO consider removing these functions
they are not numerically stable for high degree (since they essentially calculate 2^degree)
they may be redundant if already doing a QR of the U*U Vandermonde
=#

function recover_lagrange_polys(pts::Matrix{T}, deg::Int) where {T <: Real}
    deg > 8 && @warn("recover_lagrange_polys is not numerically stable for large degree")
    (U, n) = size(pts)
    DP.@polyvar x[1:n]
    basis = get_chebyshev_polys(x, deg)
    @assert length(basis) == U
    vandermonde_inv = inv([basis[j](x => view(pts, i, :)) for i in 1:U, j in 1:U])
    lagrange_polys = [DP.polynomial(view(vandermonde_inv, :, i), basis) for i in 1:U]
    return lagrange_polys
end

# returns the multivariate Chebyshev polynomials in x up to degree deg
function get_chebyshev_polys(x::Vector{DP.PolyVar{true}}, deg::Int)
    deg > 8 && @warn("get_chebyshev_polys is not numerically stable for large degree")
    n = length(x)
    u = calc_chebyshev_univariate(x, deg)
    V = Vector{DP.Polynomial{true, Int}}(undef, get_L(n, deg))
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

function calc_chebyshev_univariate(monovec::Vector{DP.PolyVar{true}}, deg::Int)
    deg > 8 && @warn("calc_chebyshev_univariate is not numerically stable for large degree")
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
