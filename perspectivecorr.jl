using LinearAlgebra
using ForwardDiff
d = 2
dim = d + 2

## careful
# h is phi in the overleaf
# phi is zeta in the overleaf
# ∇p_v is sigma in overleaf
# squig_v is xi in overleaf
# corr is actually -2 * Tau

##
s = 1
h(w) = sum(log, w)
∇h(w) = inv.(w)
∇2h(w) = Diagonal(-inv.(w .^ 2))
∇3h(w) = [(i == j == k) ? 2 / w[i]^3 : 0 for i in 1:d, j in 1:d, k in 1:d]

# s = 1
# h(w) = sum(x -> x * log(x), w)
# ∇h(w) = log.(w) .+ 1
# ∇2h(w) = Diagonal(inv.(w))
# ∇3h(w) = -[(i == j == k) ? 1 / w[i]^2 : 0 for i in 1:d, j in 1:d, k in 1:d]

##
p(v, w) = v * h(w / v)
v = rand()
w =  rand(d)
u = p(v, w) + s * rand() * 10
uvw = vcat(u, v, w)


function f(uvw)
    u = uvw[1]
    v = uvw[2]
    w = uvw[3:end]
    return -log(- s * p(v, w) + s * u) - log(v) - sum(log, w)
end
∇p_v(v, w) = h(w / v) - dot(w, ∇h(w / v)) / v
∇p_w(v, w) = ∇h(w / v)
∇p_vv(v, w) = 1 / v^3 * dot(w, ∇2h(w / v), w)
∇p_wv(v, w) = -1 / v^2 * ∇2h(w / v) * w
∇p_ww(v, w) = 1 / v * ∇2h(w / v)
∇p_vvv(v, w) = -3 / v^4 * dot(w, ∇2h(w / v), w) - 1 / v^5 * sum(∇3h(w / v)[i, j, k] * w[i] * w[j] * w[k] for i in 1:d, j in 1:d, k in 1:d)
∇p_wvv(v, w) = 2 / v^3 * ∇2h(w / v) * w + 1 / v^4 * [sum(∇3h(w / v)[i, j, k] * w[i] * w[j] for i in 1:d, j in 1:d) for k in 1:d]
∇p_wwv(v, w) = -1 / v^2 * ∇2h(w / v) - 1 / v^3 * [sum(∇3h(w / v)[i, j, k] * w[i] for i in 1:d) for j in 1:d, k in 1:d]
∇p_www(v, w) = 1 / v^2 * ∇3h(w / v)

T = zeros(dim, dim, dim)
phi = -(u - p(v, w))
T[1, 1, 1] = 2 * phi^(-3) * s
T[1, 1, 2] = T[1, 2, 1] = T[2, 1, 1] = -2 * phi^(-3) * ∇p_v(v, w)
T[1, 1, 3:end] = T[1, 3:end, 1] = T[3:end, 1, 1] = -2 * phi^(-3) * ∇p_w(v, w)
T[1, 2, 2] = T[2, 1, 2] = T[2, 2, 1] = 2 * phi^(-3) * s * ∇p_v(v, w)^2 - phi^(-2) * s * ∇p_vv(v, w)
T[2, 2, 2] = -2 * phi^(-3) * ∇p_v(v, w)^3 + 3 * phi^(-2) * ∇p_vv(v, w) * ∇p_v(v, w) - phi^(-1) * ∇p_vvv(v, w) - 2 * v^(-3)
T[1, 2, 3:end] = T[2, 1, 3:end] = T[1, 3:end, 2] = T[2, 3:end, 1] = T[3:end, 1, 2] =
    T[3:end, 2, 1] = 2 * phi^(-3) * s * ∇p_v(v, w) * ∇p_w(v, w) - phi^(-2) * s * ∇p_wv(v, w)
T[1, 3:end, 3:end] = T[3:end, 1, 3:end] = T[3:end, 3:end, 1] =
    2 * phi^(-3) * s * ∇p_w(v, w) * ∇p_w(v, w)' - phi^(-2) * s * ∇p_ww(v, w)
T[2, 2, 3:end] = T[2, 3:end, 2] = T[3:end, 2, 2] =
    -2 * phi^(-3) * ∇p_v(v, w)^2 * ∇p_w(v, w) + 2 * phi^(-2) * ∇p_v(v, w) * ∇p_wv(v, w) + phi^(-2) * ∇p_vv(v, w) * ∇p_w(v, w) - phi^(-1) * ∇p_wvv(v, w)
T[3:end, 3:end, 2] = T[3:end, 2, 3:end] = T[2, 3:end, 3:end] =
    -2 * phi^(-3) * ∇p_v(v, w) * ∇p_w(v, w) * ∇p_w(v, w)' +
    phi^(-2) * (∇p_w(v, w) * ∇p_wv(v, w)' + ∇p_wv(v, w) * ∇p_w(v, w)') +
    phi^(-2) * ∇p_ww(v, w) * ∇p_v(v, w) - phi^(-1) * ∇p_wwv(v, w)
T[3:end, 3:end, 3:end] =
    -2 * phi^(-3) * [∇p_w(v, w)[i] * ∇p_w(v, w)[j] * ∇p_w(v, w)[k] for i in 1:d, j in 1:d, k in 1:d] +
    phi^(-2) * [∇p_ww(v, w)[i, j] * ∇p_w(v, w)[k] + ∇p_ww(v, w)[i, k] * ∇p_w(v, w)[j] + ∇p_ww(v, w)[k, j] * ∇p_w(v, w)[i] for i in 1:d, j in 1:d, k in 1:d] -
    phi^(-1) * ∇p_www(v, w) - [(i == j == k) ? 2 / w[i]^3 : 0 for i in 1:d, j in 1:d, k in 1:d]

# fd_T = ForwardDiff.jacobian(x -> ForwardDiff.hessian(f, x), uvw)
# @show reshape(T, dim^2, dim) ./ fd_T

# phi = s * (u - p(v, w))

π = randn()
q = randn()
r = randn(d)
pqr = vcat(π, q, r)

x = randn()
y = randn()
z = randn(d)
xyz = vcat(x, y, z)
# (x, y, z, xyz) = (copy(π), copy(q), copy(r), copy(pqr))

f_dir(point, s, t) = f(point + s * pqr + t * xyz)
fd_corr = ForwardDiff.gradient(
    s2 -> ForwardDiff.derivative(
        s -> ForwardDiff.derivative(
            t -> f_dir(s2, s, t),
            0),
        0),
    uvw)
# fd_corr = ForwardDiff.gradient(x -> ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> f_dir(x, t), s), 0), uvw)

squig = -q / v * w + r
squig_v = squig / v
squig_1 = (-q / v * w + r) / v
squig_2 = (-y / v * w + z) / v
chi = s * (π - q * ∇p_v(v, w) - dot(∇p_w(v, w), r))

chi_1 = π - q * ∇p_v(v, w) - dot(∇p_w(v, w), r)
chi_2 = x - y * ∇p_v(v, w) - dot(∇p_w(v, w), z)


##

(π -  q * ∇p_v(v, w) - dot(∇p_w(v, w), r)) * (x -  y * ∇p_v(v, w) - dot(∇p_w(v, w), z))

my_corr_u =
    2 * phi^(-3) * chi_1 * chi_2 +
    -phi^(-2) * v * dot(squig_1, ∇2h(w / v), squig_2)


# had to do some stuff manually here because tensors are annoying
∇3hwww = sum(∇3h(w / v)[i, j, k] * w[i] * w[j] * w[k] for i in 1:d, j in 1:d, k in 1:d)
∇3hww = [sum(∇3h(w / v)[i, j, k] * w[i] * w[j] for i in 1:d, j in 1:d) for k in 1:d]
∇3hw = [sum(∇3h(w / v)[i, j, k] * w[i] for i in 1:d) for j in 1:d, k in 1:d]

phiichiqv = phi^(-1) * chi_1 + q / v

my_corr_v =
    -my_corr_u * ∇p_v(v, w) +
    s * 2 * phi^(-1) * dot(squig_v, ∇2h(w / v), w / v) * phiichiqv +
    -2 * v^(-3) * q^2 +
    -s * phi^(-1) * dot(squig_v, ∇2h(w / v), squig_v) +
    -phi^(-1) * (
        s * sum(∇3h(w / v)[i, j, k] * w[i] * squig_v[j] * squig_v[k] for i in 1:d, j in 1:d, k in 1:d) / v
        )

phiichiyv = phi^(-1) * chi_2 + y / v

# wrong
my_corr_v =
    -my_corr_u * ∇p_v(v, w) +
    phi^(-1) * dot(squig_1, ∇2h(w / v), w / v) * phiichiyv +
    phi^(-1) * dot(squig_2, ∇2h(w / v), w / v) * phiichiqv +
    -2 * v^(-3) * q * y +
    -phi^(-1) * dot(squig_1, ∇2h(w / v), squig_2) +
    -phi^(-1) * (
        sum(∇3h(w / v)[i, j, k] * w[i] * squig_1[j] * squig_2[k] for i in 1:d, j in 1:d, k in 1:d) / v
        )

# right but out of control
my_corr_v =
    -2 * phi^(-3) * ∇p_v(v, w) * (chi_1 * chi_2 - q * y * ∇p_v(v, w)^2) +
    -phi^(-2) * (
        #  dot(
        #     w / v^2,
        #     ∇2h(w / v),
        #     1 / v * w * (π * y + x * q + -dot(r, ∇p_w(v, w)) * y + -dot(z, ∇p_w(v, w)) * q) +
        #         ∇p_v(v, w) * 2 * (y * r + q * z) +
        #         -x * r +
        #         -π * z +
        #         dot(r, ∇p_w(v, w)) * z +
        #         dot(∇p_w(v, w), z) * r
        #     ) +
        # ∇p_v(v, w) * -dot(r, 1 / v * ∇2h(w / v), z)
        v * dot(q / v^2 * w, ∇2h(w / v), x / v^2 * w) +
        v * dot(y / v^2 * w, ∇2h(w / v), π / v^2 * w) +
        -v * dot(x / v^2 * w, ∇2h(w / v), r / v) +
        -v * dot(π / v^2 * w, ∇2h(w / v), z / v) +
        -v * ∇p_v(v, w) * dot(r / v, ∇2h(w / v), z / v) +
        2 * v * ∇p_v(v, w) * dot(q / v^2 * w, ∇2h(w / v), z / v) +
        2 * v * ∇p_v(v, w) * dot(y / v^2 * w, ∇2h(w / v), r / v) +
        v * dot(r, ∇p_w(v, w)) * dot(w / v^2, ∇2h(w / v), z / v) +
        v * dot(z, ∇p_w(v, w)) * dot(w / v^2, ∇2h(w / v), r / v) +
        -v * dot(r, ∇p_w(v, w)) * dot(w / v^2, ∇2h(w / v), y / v^2 * w) +
        -v * dot(z, ∇p_w(v, w)) * dot(w / v^2, ∇2h(w / v), q / v^2 * w)
    ) +
    -phi^(-1) * dot(r, ∇p_wvv(v, w)) * y +
    -phi^(-1) * dot(r, ∇p_wwv(v, w), z) +
    -phi^(-1) * dot(z, ∇p_wvv(v, w) * q) +
    T[2, 2, 2] * q * y



my_corr_w =
    -∇h(w / v) * my_corr_u +
    -s * 2 * phi^(-1) * ∇2h(w / v) * squig_v * phiichiqv +
    s * phi^(-1) * [dot(squig_v, ∇3h(w / v)[i, :, :], squig_v) for i in 1:d] +  # NOTE when using difference matrix need to double
    -2 * r.^2 ./ w.^3

my_corr_w =
    -∇h(w / v) * my_corr_u +
    -phi^(-1) * ∇2h(w / v) * squig_1 * phiichiyv +
    -phi^(-1) * ∇2h(w / v) * squig_2 * phiichiqv +
    phi^(-1) * [dot(squig_1, ∇3h(w / v)[i, :, :], squig_2) for i in 1:d] +  # NOTE when using difference matrix need to double
    -2 * r .* z ./ w.^3

@show fd_corr[1] / my_corr_u
@show fd_corr[2] / my_corr_v
@show fd_corr[3:end] ./ my_corr_w

# reshape(fd_T * pqr, d + 2, d + 2) * pqr


;
