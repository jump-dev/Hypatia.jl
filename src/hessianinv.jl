import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const SO = HYP.Solvers
const MO = HYP.Models
const MU = HYP.ModelUtilities
using LinearAlgebra

n = 1
d = 4
domain = MU.FreeDomain(n)
(U, pts, P, PWts, w) = MU.interpolate(domain, d)
L = size(P, 2)
P = Array(qr(P).Q)

x = ones(U) * 2.0

W = Symmetric(P' * Diagonal(x) * P)
@show W
Wi = inv(W)
@show Wi
println()

H = Symmetric((P * Wi * P').^2)
@show H
# H_try = [sum(Wi[k, l] * P[i, k] * P[j, l] for k in 1:L, l in 1:L)^2 for i in 1:U, j in 1:U]
# @show abs.(H - H_try)
println()

Hi = inv(H)
@show Hi
println()





# Hi_try = Symmetric(P * Wi.^2 * P')
# @show Hi_try
# # Hi_try = [sum(sqrt(W[k, l]) / P[i, k] / P[j, l] for k in 1:L, l in 1:L) for i in 1:U, j in 1:U]
# @show abs.(Hi - Hi_try)
#
# PtP = Symmetric(P' * P)
# Lfac = cholesky(PtP).L
# @show P * P'

lambda(y) = P' * Diagonal(y) * P

H_op(y) = diag(P * Wi * lambda(y) * Wi * P')
function Hid()
    U = size(P, 1)
    H = zeros(U, U)
    for i in 1:U
        ei = zeros(U)
        ei[i] = 1.0
        H[:, i] = H_op(ei)
    end
    H
end
# @show Hid()


P * (W * lambda(y) * W) * P'


gradient(y) = -diag(P * (lambda(y) \ P'))
H_opi(y) = diag(P * inv(lambda(gradient(x))) * lambda(y) * inv(lambda(gradient(x))) * P')
# H_opi(y) = diag(P * inv(lambda(1 ./ x)) * lambda(y) * inv(lambda(1 ./ x)) * P')

# PWii = pinv(P * Wi)'
# H_opi(y) = diag(PWii * lambda(y) * PWii')

H_opi(y) =

function Hidi()
    U = size(P, 1)
    H = zeros(U, U)
    for i in 1:U
        ei = zeros(U)
        ei[i] = 1.0
        H[:, i] = H_opi(ei)
    end
    H
end
@show Hidi()





;
