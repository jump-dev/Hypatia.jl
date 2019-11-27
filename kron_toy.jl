using LinearAlgebra

function my_kron(x, y)
    (mx, nx) = size(x)
    (my, ny) = size(y)
    mxmy = mx * my
    nxny = nx * ny
    prod = zeros(mxmy, nxny)
    for xi in 1:mx, yi in 1:my, xj in 1:nx, yj in 1:ny
        prod[(xi - 1) * my + yi, (xj - 1) * ny + yj] = x[xi, xj] * y[yi, yj]
    end
    return prod
end

# x and y need to be the same size. looks like khatri-rao product?
function my_kron_tr(x, y)
    (mx, nx) = size(x)
    (my, ny) = size(y)
    # mxmy = mx * my
    # nxny = nx * ny
    mxnx = mx * nx
    myny = my * ny
    prod = zeros(mxnx, myny)
    for xi in 1:mx, yi in 1:my, xj in 1:nx, yj in 1:ny
        prod[(xj - 1) * mx + xi, (yj - 1) * mx + yi] = x[yi, xj] * y[xi, yj]
    end
    return prod
end

println(my_kron(Matrix(I, 3, 3), randn(2, 2)))
A = randn(4, 5)
B = randn(6, 7)
@assert my_kron(A, B) ≈ kron(A, B)
X = randn(7, 5)
@assert my_kron(A, B) * vec(X) ≈ vec(B * X * A')
A = randn(6, 6)
B = randn(6, 6)
A = A * A'
B = B * B'
C = my_kron(A, B)
@assert cholesky(C).L ≈ my_kron(cholesky(A).L, cholesky(B).L)

println(my_kron_tr(Matrix(I, 3, 3), randn(3, 3)))
println(my_kron_tr([1 2; 3 4], [5 6; 7 8]))
println(my_kron_tr([1 2; 3 4], [1 2; 3 4]))

A = randn(4, 5)
B = randn(4, 5)
X = randn(4, 5)
@assert my_kron_tr(A, A) * vec(X) ≈ vec(A * X' * A)
@assert my_kron_tr(A, B) * vec(X) ≈ vec(B * X' * A)
A = randn(6, 6)
B = randn(6, 6)
A = A * A'
B = B * B'
C = my_kron_tr(A, A)


W = B
u = opnorm(W) + 2
Z = abs2(u) * I - W * W'
Zi = inv(Z)
T = Zi * W
my_term = my_kron_tr(T, T)
@show isposdef(my_term)



;
