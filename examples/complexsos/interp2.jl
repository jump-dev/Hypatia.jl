
using LinearAlgebra
using Test


# univariate
d = 3

# # full vandermonde
# # V_basis = [x -> x^i * conj(x)^j for j in 0:d for i in 0:d] # TODO columns are dependent if not doing j in 0:i
# V_basis = [x -> x^i * conj(x)^j for j in 0:d for i in 0:j]
# U = length(V_basis)
# @show U
# @show div(d * (d + 1), 2)



# V_basis = [z -> z^i * conj(z)^j for j in 0:d for i in 0:j]
U = div((d + 1) * (d + 2), 2)

# V_basis = [z -> z^i * conj(z)^j for j in 0:d for i in 0:d]
# U = (d + 1)^2

# points are randomly sampled
# points = 2 * rand(ComplexF64, U) .- (1 + im)
radii = sqrt.(rand(U))
angles = rand(U) .* 2pi
points = radii .* (cos.(angles) .+ (sin.(angles) .* im))

# # points are the roots of unity
# points = [cospi(2k / U) + sinpi(2k / U) * im for k = 0:(U - 1)]


P = [p^i for p in points, i in 0:d]
@assert rank(P) == d + 1

# # @show points
# V = [b(p) for p in points, b in V_basis]
# # @show rank(V)
# @test rank(V) == U


# make_psd = true
make_psd = false


# rand solution
fh = randn(ComplexF64, d + 1, d + 1)
if make_psd
    F = Hermitian(fh * fh')
else
    F = Hermitian(fh)
end


# values at points given coefs
vals = [sum(F[i+1, j+1] * p^i * conj(p)^j for i in 0:d, j in 0:d) for p in points]
@assert real(vals) ≈ vals
vals = real(vals)
@show vals

# fvec = vec(F)
# @test vals ≈ V * fvec
#
# # from values at points, recover coefs
# test_coefs = V \ vals
# @test test_coefs ≈ fvec
# @show fvec
# @show test_coefs


Lam = Hermitian(P' * Diagonal(vals) * P)
# @show norm(Lam - P' * Diagonal(vals) * P)

@show eigvals(Lam)
@show isposdef(Lam)
@test isposdef(Lam) == isposdef(F)
;
