
# using Hypatia
using LinearAlgebra
using Test
using Combinatorics


# univariate

deg = 5
m = deg + 1

# basis [1, z, z^2, ..., z^deg]
basis = [x -> x^j for j in 0:deg]

# rand complex points
points = randn(ComplexF64, m)

# vandermonde
V = [b(p) for p in points, b in basis]
@test rank(V) == m
F = qr(V)

# rand real poly coefs
coefs = randn(m)

# values at points given coefs
vals = [dot(coefs, [b(p) for b in basis]) for p in points]
@test vals ≈ V * coefs

# from values at points, recover coefs
test_coefs = F \ vals
@test test_coefs ≈ coefs


# multivariate

n = 3
deg = 4
m = binomial(n + deg, n)

# basis [1, z1, z2, ..., z1^2, z1z2, z2^2, ...]
basis = [(p -> prod(p[i]^a[i] for i in eachindex(a))) for t in 0:deg for a in Combinatorics.multiexponents(n, t)]
@test length(basis) == m

# rand complex points
points = randn(ComplexF64, m, n)

# vandermonde
V = [b(p) for p in eachrow(points), b in basis]
@test rank(V) == m
F = qr(V)

# rand real poly coefs
coefs = randn(m)

# values at points given coefs
vals = [dot(coefs, [b(p) for b in basis]) for p in eachrow(points)]
@test vals ≈ V * coefs

# from values at points, recover coefs
test_coefs = F \ vals
@test test_coefs ≈ coefs


# TODO select from larger sample of points using QR procedure
