using ForwardDiff
using DiffResults
using LinearAlgebra
import Hypatia
import Random

rt2 = sqrt(2)

function barfun(point)
    u = point[1]
    v = point[2]
    dim = length(point)
    n = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
    W = similar(point, n, n)
    Hypatia.Cones.svec_to_smat!(W, view(point, 3:dim))
    return -log(v * logdet(W / v) - u) - logdet(W) - log(v)
end

Random.seed!(1)
n = 2
u = -5
v = 1
r = rand(n, n)
W = r * r' # Matrix{Float64}(I, n, n)
Wvec = zeros(3)
point = vcat(u, v, Hypatia.Cones.smat_to_svec!(Wvec, W)...)

dim = length(point)

diffres = DiffResults.HessianResult(zeros(dim))
diffres = ForwardDiff.hessian!(diffres, barfun, point)
g = DiffResults.gradient(diffres)
H = DiffResults.hessian(diffres)
@show g, H

L = logdet(W / v)
z = v * L - u
Wi = inv(W)

gu = 1 / z
gv = (n - L) / z - 1 / v
gwmat = -v / z * Wi - Wi
gw = zeros(dim - 2)
gw = Hypatia.Cones.smat_to_svec!(gw, gwmat)
g = [gu, gv, gw...]

Huu = 1 / z / z
Huv = (n - L) / z / z
Huwmat = -(v * Wi) / z / z
Huw = zeros(dim - 2)
Huw = Hypatia.Cones.smat_to_svec!(Huw, Huwmat)


Hvv = (-n + L)^2 / z / z + n / (v * z) + 1 / v^2
Hvwmat = (-n + L) * v * Wi / z / z - Wi / z
Hvw = zeros(dim - 2)
Hvw = Hypatia.Cones.smat_to_svec!(Hvw, Hvwmat)

Hww = zeros((dim - 2), (dim - 2))
k = 1
for i in 1:n, j in 1:i
    k2 = 1
    for i2 in 1:n, j2 in 1:i2
        Hww[k2, k] += Wi[i, j] * Wi[i2, j2] * v^2 / z^2 # this guy's indices should't behave like the other part because he involves only 1 derivative wrt w
        Hww[k2, k] += Wi[i, j2] * Wi[i2, j] * v / z +  Wi[i, j2] * Wi[i2, j]
        # if xor(i == i2, j == j2)
        #     Hww[k2, k] *= sqrt(2)
        # end
        fact = xor(i == j, i2 == j2) ? rt2i : 1.0
        Hww[k2, k] *= rt2i

        # if i == j
        #     if i2 == j2
        #         Hww[k2, k] += abs2(Wi[i2, i]) * v / z +  abs2(Wi[i2, i]) # = abs2(inv_mat[i2, i])
        #     else
        #         Hww[k2, k] += rt2 * (Wi[i2, i] * Wi[j, j2] * v / z +  Wi[i2, i] * Wi[j, j2]) # = rt2 * inv_mat[i2, i] * inv_mat[j, j2]
        #     end
        # else
        #     if i2 == j2
        #         Hww[k2, k] += rt2 * (Wi[i2, i] * Wi[j, j2] * v / z +  Wi[i2, i] * Wi[j, j2]) # = rt2 * inv_mat[i2, i] * inv_mat[j, j2]
        #     else
        #         Hww[k2, k] += (Wi[i2, i] * Wi[j, j2] + Wi[j2, i] * Wi[j, i2]) * v / z +  (Wi[i2, i] * Wi[j, j2] + Wi[j2, i] * Wi[j, i2]) # = inv_mat[i2, i] * inv_mat[j, j2] + inv_mat[j2, i] * inv_mat[j, i2]
        #     end
        # end
        # # if k2 == k
        # #     break
        # # end
        k2 += 1
    end
    global k += 1
end

@show H[3:end, 3:end] ./ Hww


;
