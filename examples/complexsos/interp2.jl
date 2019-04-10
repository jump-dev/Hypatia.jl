
using LinearAlgebra
using Test


# univariate
d = 5

# full vandermonde
V_basis = [x -> x^i * conj(x)^j for j in 0:d for i in 0:d] # TODO columns are dependent if not doing j in 0:i
U = length(V_basis)
# @show U
# @show div(d * (d + 1), 2)
points = randn(ComplexF64, (d + 1)^2) # TODO try roots of unity
V = [b(p) for p in points, b in V_basis]
# @show rank(V)

# # points are the roots of unity
# points = [cospi(2k / U) + sinpi(2k / U) * im for k = 0:(U - 1)]
# @show points
# V = [b(p) for p in points, b in basis]
# @test rank(V) == U


# rand real poly coefs
fh = randn(ComplexF64, d + 1, d + 1)
f = Hermitian(fh * fh')

# values at points given coefs
vals = [sum(f[i+1, j+1] * p^i * conj(p)^j for i in 0:d, j in 0:d) for p in points]
@test real(vals) ≈ vals

fvec = vec(f)
@test vals ≈ V * fvec

# from values at points, recover coefs
test_coefs = V \ vals
@test test_coefs ≈ fvec


y = randn((d + 1)^2)

P = V[:, 1:d+1] # [1, z, ..., z^d]
@test ishermitian(P' * Diagonal(y) * P)
Lam = Hermitian(P' * Diagonal(y) * P)
@show isposdef(Lam)
