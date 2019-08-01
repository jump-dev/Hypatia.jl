using DelimitedFiles, CSV, Hypatia

LHS = CSV.read("lhs.csv", header = false);
LHS = convert(Matrix, LHS)
b = CSV.read("rhs.csv", header = false);
b = convert(Matrix, b)
A = Symmetric(LHS, :U)

# all in Float64
F = cholesky(A)
@assert isposdef(F)
x = F \ b
@show norm(b - A * x) # 2.0596501028708477e-6

# cholesky in Float64, ldiv in BF
b_bf = BigFloat.(b)
b_tmp = b_bf[:, i]
x = BigFloat.(F.U) \ (BigFloat.(F.L) \ b_tmp)
@show norm(b_tmp - BigFloat.(A) * x) # blah e-6

A_bf = BigFloat.(A)
F_bf = cholesky(A_bf)
x_bf = F_bf \ b_bf
@show norm(b_bf - A_bf * x_bf) # blah...e-4922

x_cast = Float64.(x_bf)
@show norm(b - A * x_cast) # 1.2663985543670767e-6
@show norm(BigFloat.(b) - BigFloat.(A) * BigFloat.(x_cast)) # blah e-7

i = 1
bi = b[:, i]
bi_scaled = bi / 1e9
A_scaled = A / 1e9
F = cholesky(Symmetric(A_scaled))
xi = F \ bi_scaled
# xi .*= 1e9
@show norm(bi - A * xi) / norm(A * xi)

@show norm(bi_scaled - A_scaled * xi)

b_bfi = BigFloat.(bi)
F_bf = cholesky(A_bf)
x_bfi = F_bf \ b_bfi
@show norm(b_bfi - A_bf * x_bfi) # blah...e-4922

x_casti = Float64.(x_bfi)
@show norm(bi - A * x_casti) # 1.2663985543670767e-6
@show norm(BigFloat.(bi) - BigFloat.(A) * BigFloat.(x_casti)) # blah e-7

y = zeros(size(bi, 1), 1)
A_tmp = copy(A.data)
bi_tmp = zeros(size(bi, 1), 1)
bi_tmp[:] .= bi
@time (_, S) = Hypatia.hyp_posvx!(y, A_tmp, bi_tmp)

b .= 1
S = I
y = cholesky(Symmetric(S * A * S)) \ (S * b)
@show norm(b - A * S * y)
y = bunchkaufman(Symmetric(S * A * S)) \ (S * b)
@show norm(b - A * S * y)



@show norm(bi - A * y)
