using ForwardDiff
using LinearAlgebra

n = 1
m = 2 # any m >= n
dim = 1 + n * m

point = rand(dim)
# point = vcat(rand(), vec(Diagonal(rand(m))))
point[1] += m

function barrier(s)
    (u, W) = (s[1], reshape(s[2:end], n, m))
    return -logdet(cholesky!(Symmetric(abs2(u) * I - W * W'))) + (n - 1) * log(u)
end

H = Symmetric(ForwardDiff.hessian(barrier, point))
Hi = Symmetric(inv(H))

u = point[1]
W = reshape(point[2:end], n, m)
Z = Symmetric(abs2(u) * I - W * W')
Zi = Symmetric(inv(Z))

T = Symmetric(abs2(u) * I + W * W')
Ti = Symmetric(inv(T))
sigma = (1 - n) / u^2 + 2 * tr(Ti)
ZiW = Zi * W

2*kron(Zi, Zi)
4*kron(ZiW, ZiW') + 2*kron(Zi, Diagonal(ones(2)))
H[2:end, 2:end]



rho(i, j) = (i == j ? 1 : sqrt(2))
gkron(i, j, k, l, X, Y) = X[i, k] * Y[j, l]
akron(i, j, k, l, X, Y) = X[i, l] * Y[k, j] + X[k, j] * Y[i, l]
skron(i, j, k, l, M, N) = (M[i, k] * N[j, l] + M[i, l] * N[j, k] + N[i, k] * M[j, l] + N[i, l] * M[j, k]) * rho(i, j) * rho(k, l) / 4



# Huu = (2 * u * tr(Z))^2 / det(Z)^2 - (2 * tr(Z) + 8 * u^2) / det(Z) - 1 / u^2
# Hiuu = u^2 * (det(Z) + 2 * u^2 * sum(abs2, W)) / (4 * u^4 - det(Z))

Huu = 2 * tr(T * Zi^2) - (n - 1) / u^2
H[1,1]
HUW = -4 * u * Zi * Zi * W
H[1,2:end]


# akroninv(i, j, k, l, X, Y) = X[i, j] * Y[k, l] + X[k, l] * Y[i, j]

Hww = zeros(dim - 1, dim - 1)
Hiww = zeros(dim - 1, dim - 1)
ZiW = Zi * W
# ZiWi = pinv(ZiW)'
global idx1 = 1
for j1 in 1:m, i1 in 1:n
    idx2 = 1
    for j2 in 1:m, i2 in 1:n
        # Hww[idx1, idx2] = ZiW[, i2] + ZiW[j2, j1]
        # Hiww[idx1, idx2] = 0.25 * akron(i1, j1, i2, j2, ZiWi, ZiWi)
        # Hww[idx1, idx2] +=
            # 2 * gkron(i1, j1, i2, j2, Zi, W' * Zi * W + I) +
            # 1 * akron(i1, j1, i2, j2, Zi * W, Zi * W)
            # 2 * gkron(i1, j1, i2, j2, Zi, I) +
            # 2 * gkron(i1, j1, i2, j2, Zi, W' * Zi * W) +
            # 1 * akron(i1, j1, i2, j2, Zi * W, Zi * W)
        idx2 += 1
        # @show akron(i1, j1, i2, j2, inv(Zi * W), inv(Zi * W))
        # @show akron(i1, j1, i2, j2, Zi * W, Zi * W)
        # @show gkron(i1, j1, i2, j2, Zi, W' * Zi * W)
        # @show akron(i1, j1, i2, j2, Zi * W, Zi * W)
        # @show gkron(i1, j1, i2, j2, Zi, I)
        # @show skron(i1, j1, i2, j2, Zi, I)
        # println()
    end
    global idx1 += 1
end

Hww
invHww = inv(Hww)
# Hiww
Hww * Hiww


Hww
H[2:end, 2:end]







Hiuu = inv(sigma)
Hi[1,1]
HIUW = 2 * u / sigma * Ti * W
Hi[1,2:end]
# HiWW = 4 * u^2 / sigma * kron(Ti * W) + kron ...
# Hi[2:end,2:end]

ZTiZ = Zi * T * Zi
uTiW = 2 * u * Ti * W
# uTiW = 2 * u * vec(Ti * W)
# Hiww = uTiW * uTiW' / sigma
Hiww = zeros(dim - 1, dim - 1)
global idx1 = 1
for j1 in 1:m, i1 in 1:n
    idx2 = 1
    for j2 in 1:m, i2 in 1:n
        Hiww[idx1, idx2] +=
            Hiuu * akron(i1, j1, i2, j2, uTiW, uTiW) #+
            # + 0.5 * akron(i1, j1, i2, j2, uTiW, uTiW)
            # 0.5 * gkron(i1, j1, i2, j2, Z, Ti * Z)
            # + 0.5 * akron(i1, j1, i2, j2, uTiW, uTiW)
        idx2 += 1
    end
    global idx1 += 1
end
@show Hiww
# @show vec(uTiW) * vec(uTiW)' / sigma
# @show Hiww ./ Hi[2:end, 2:end]
Hi[2:end, 2:end]


ZTiZ = Z * Ti * Z / 2
uTiW = 2 * u * Ti * W
Hiww = [
    # uTiW[i,j] * uTiW[k,l] / sigma +
    dot(uTiW[i,:], uTiW[k,:]) / sigma +
    #akron(i, j, k, l, uTiW, uTiW) / sigma
    (j == l ? ZTiZ[i,k] : 0)
    for i in 1:n, j in 1:m, k in 1:n, l in 1:m
    ]
Hiww
Hi[2:end, 2:end]




# note T * Z = Z * T = u^4 I - (W * W')^2
# note Ti * Zi = Zi * Ti = (T * Z)^-1



# H prod try
arr = rand(dim)
# arr = [0.0, 1.0, 0.0]

pro = zero(arr)
temp_W = zero(W)
@views arr_w = arr[2:end]
@views pro_w = pro[2:end]
temp_W[:] .= arr_w

pro[1] = Huu * arr[1] + dot(HUW, temp_W)
pro_w .= vec(HUW) * arr[1]

tmpnn = temp_W * W'
# tmpnn = tmpnn + tmpnn' - 2 * u * arr[1] * I
tmpnm = Hermitian(tmpnn + tmpnn', :U) * Zi * W + temp_W
tmpnm = 2 * Zi * tmpnm
pro_w .+= vec(tmpnm)

H * arr
pro

# tmpnn = temp_W * W'
# tmpnn = tmpnn + tmpnn' - 2 * u * arr[1] * I
# tmpnm = Hermitian(tmpnn, :U) * Zi * W + temp_W
# tmpnm .*= 2
# tmpnm = Zi * tmpnm
# pro_w .= vec(tmpnm)

# pro_w .+= 4 * dot(Zi * W, temp_W) * vec(Zi * W)
# pro_w .+= vec(2 * Zi) .* temp_W
# pro_w .+= vec(2 * Zi * T * Zi * temp_W)






# Hi prod try
arr = rand(dim)
# arr = [0.0, 1.0, 0.0]

pro = zero(arr)
temp_W = zero(W)
@views arr_w = arr[2:end]
@views pro_w = pro[2:end]
temp_W[:] .= arr_w

pro[1] = Hiuu * arr[1] + dot(HIUW, temp_W)
pro_w .= vec(HIUW) * arr[1]

pro_w .+= 4u^2 / sigma * dot(Ti * W, temp_W) * vec(Ti * W)
pro_w .+= vec(Z^2 * Ti * temp_W / 2)
# pro_w .+= dot(W, temp_W) * vec(W)
# pro_w .+= vec(Z * temp_W / 2)

# tmpnn = temp_W * W'
# tmpnm = Hermitian(tmpnn + tmpnn', :U) * Ti * W + temp_W
# tmpnm = 2 * u * Ti * tmpnm
# pro_w .+= vec(tmpnm)

# tmpnn = temp_W * W' * Ti
# # tmpnn = tmpnn + tmpnn' - 2 * u * arr[1] * I
# tmpnm = 4 * u^2 / sigma * Hermitian(tmpnn + tmpnn', :U) * Ti * W
# tmpnm2 = Z^2 * Ti * temp_W / 2
# pro_w .+= vec(tmpnm + tmpnm2)
# # pro_w .+= vec(tmpnm2)
# # pro_w .+= 4u^2 / sigma * dot(Ti * W, temp_W) * vec(Ti * W)

Hi * arr
pro

Hi * arr ./ pro



# # works for n = 1
# # pro_w .+= dot(W, temp_W) * vec(W)
# # pro_w .+= vec(Z / 2 * temp_W)
# # works for diagonal elements
# pro_w .+= 4u^2 / sigma * dot(Ti * W, temp_W) * vec(Ti * W)
# pro_w .+= vec(Z * Ti * Z / 2 * temp_W)
# # pro_w .+= tr(Z * Ti * Z) / 2 * vec(temp_W)


# temp_Z = temp_W * HIUW'
# temp_Z2 = temp_Z + temp_Z' # ... diag
# temp_W
# mul!(temp_Z, temp_W, W')
# temp_Z2 = temp_Z + temp_Z' # ... diag
# mul!(temp_W, Hermitian(temp_Z2, :U), Ti * W, 2 * u / sigma, 2)
# temp_W2 = Ti * temp_W
# pro_w .= vec(temp_W2)





# D = Symmetric(H[2:end, 2:end], :U)
# z = H[1, 2:end] #-4 * u * vec(Zi ^ 2 * W)
# u = vcat(-1, D \ z)
# a = H[1, 1]
# rho = a - z' * inv(D) * z
# Hitry = u * u' / rho
# Hitry[2:end, 2:end] .+= inv(D)
#
#
#
# (Hww \ Huw) / -sigma = Hiuw
# (Hww \ (-4 * u * vec(Zi * Zi * W)) / -sigma = 2 * u / sigma * vec(Ti * W)
# (Hww \ (2 * vec(Zi * Zi * W)) = vec(Ti * W)
#
#
#
# hijkl = Zi[l, j] * (W' * Zi * W)[i, k] + (Zi * W)[l, i] * (Zi * W)[j, k]






gkron(i, j, k, l, X, Y) = X[i, k] * Y[j, l]
akron(i, j, k, l, X, Y) = X[i, l] * Y[k, j] + X[k, j] * Y[i, l]



uTiW = 2 * u * Ti * W
Hiww = vec(uTiW) * vec(uTiW)' / sigma
Hiww = vec(W) * vec(W)'

# Hiww = zeros(dim - 1, dim - 1)
global idx1 = 1
for j1 in 1:m, i1 in 1:n
    idx2 = 1
    for j2 in 1:m, i2 in 1:n
        Hiww[idx1, idx2] +=
            # 0.5 * gkron(i1, j1, i2, j2, Z, I) +
            # 0.5 * gkron(i1, j1, i2, j2, Z, -W' * Ti * W) +
            # -0.25 * akron(i1, j1, i2, j2, W, Z * Ti * W)
            0.5 * gkron(i1, j1, i2, j2, Z, inv(W' * Ti * W + I)) +
            -0.25 * akron(i1, j1, i2, j2, pinv(Zi * W), pinv(Zi * W))
            # -0.25 * akron(i1, j1, i2, j2, W, Z * Ti * W)
        # Hiww[idx1, idx2] += 0.5 * gkron(i1, j1, i2, j2, Z, I)
        idx2 += 1
    end
    global idx1 += 1
end

Hiww
invHww
# Hi[2:end, 2:end]


# sigma = (1 - n) / u^2 + 2 * tr(Ti)


# 2 * u^2 * Ti - I == Z * Ti

X = zeros(dim - 1, dim - 1);
global idx1 = 1
for j1 in 1:m, i1 in 1:n
    idx2 = 1
    for j2 in 1:m, i2 in 1:n
        X[idx1, idx2] = 2 * gkron(i1, j1, i2, j2, Zi, W' * Zi * W + I)
        idx2 += 1
    end
    global idx1 += 1
end

Y = zeros(dim - 1, dim - 1);
global idx1 = 1
for j1 in 1:m, i1 in 1:n
    idx2 = 1
    for j2 in 1:m, i2 in 1:n
        Y[idx1, idx2] = akron(i1, j1, i2, j2, Zi * W, Zi * W)
        idx2 += 1
    end
    global idx1 += 1
end

Xi = zeros(dim - 1, dim - 1);
global idx1 = 1
for j1 in 1:m, i1 in 1:n
    idx2 = 1
    for j2 in 1:m, i2 in 1:n
        Xi[idx1, idx2] = 0.5 * gkron(i1, j1, i2, j2, Z, inv(W' * Zi * W + I))
        idx2 += 1
    end
    global idx1 += 1
end

Yi = zeros(dim - 1, dim - 1);
global idx1 = 1
for j1 in 1:m, i1 in 1:n
    idx2 = 1
    for j2 in 1:m, i2 in 1:n
        Yi[idx1, idx2] = 0.25 * akron(i1, j1, i2, j2, pinv(Zi * W)', pinv(Zi * W)')
        idx2 += 1
    end
    global idx1 += 1
end


inv(X + Y)
inv(X) * inv(inv(X) + inv(Y)) * inv(Y)
Xi * inv(Xi + Yi) * Yi




Y = zeros(dim - 1, dim - 1);
global idx1 = 1
for j1 in 1:m, i1 in 1:n
    idx2 = 1
    for j2 in 1:m, i2 in 1:n
        Y[idx1, idx2] = akron(i1, j1, i2, j2, Zi * W, Zi * W)
        idx2 += 1
    end
    global idx1 += 1
end
Y

Y = zeros(dim - 1, dim - 1);
global idx1 = 1
for j1 in 1:m, i1 in 1:n
    idx2 = 1
    for j2 in 1:m, i2 in 1:n
        Y[idx1, idx2] = akron(i1, j1, i2, j2, Ti * W, Ti * W)
        idx2 += 1
    end
    global idx1 += 1
end
Y
