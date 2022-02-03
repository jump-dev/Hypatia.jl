using ForwardDiff
using LinearAlgebra
using GenericLinearAlgebra

d = 4
W = randn(d, d)
W = W * W'
w = vec(W)
R = randn(d, d)
R += R'
r = vec(R)

################################################################

alpha = rand(d)
# alpha = ones(d)
alpha /= sum(alpha)
rho = reverse(eigvals(W^(-0.5) * R * W^(-0.5)))
mu = dot(alpha, rho)
dev = rho .- mu
function powerm(w)
    W = reshape(w, d, d)
    W = (W + W') / 2
    (vals, vecs) = eigen(Hermitian(W))
    return exp(dot(log.(reverse(vals)), alpha))
end
powerm_dir(t) = powerm(w + t * r)
fd_scnd = ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> powerm_dir(t), s), 0)
fd_third = ForwardDiff.derivative(x -> ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> powerm_dir(t), s), x), 0)
my_scnd = -powerm(w) * (dot(alpha, rho.^2) - dot(alpha, rho)^2)
my_scnd = -powerm(w) * dot(alpha, dev.^2)
my_third = -powerm(w) * (-dot(alpha, rho)^3 + 3 * dot(alpha, rho) * dot(alpha, rho.^2) - 2 * dot(alpha, rho.^3))
my_third = powerm(w) * dot(alpha, 3 * dev.^2 * mu + 2 * dev.^3)
@assert fd_scnd ≈ my_scnd
@assert fd_third ≈ my_third
