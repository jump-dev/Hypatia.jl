using IterativeSolvers, CSV, Preconditioners, SparseArrays
using AlgebraicMultigrid
using Krylov

A = CSV.read("lhs.csv", header = false)
A = convert(Matrix, A)
b = CSV.read("rhs.csv", header = false)
b = convert(Matrix, b)
b = b[:, 1]

A = sparse(A)
ml = ruge_stuben(Symmetric(A, :L))
p = aspreconditioner(ml)
# (xi, log) = minres!(b, Symmetric(A, :L), b, log = true, verbose = true, Pl = p)
(xi, stats) = Krylov.minres(Symmetric(A, :L), b)

A = sparse(A)
ml = ruge_stuben(A)
p = aspreconditioner(ml)
(xi, log) = gmres(A, b, log = true, restart = size(A, 2), verbose = true, Pl = p)
xe = A \ b
xi ./ xe
# (xi, log) = gmres!(xe, A, b, log = true)


using IterativeSolvers, AlgebraicMultigrid, Random
Random.seed!(1)
A = sprandn(50, 50, 0.1)
x = randn(50)
b = A * x
ml = ruge_stuben(A)
p = aspreconditioner(ml)
(y, log) = gmres(A, b, log = true, restart = size(A, 2), verbose = true, Pl = p, tol = 1e-9)
# julia> log
# Converged after 13 iterations.
# julia> log[:resnorm][end]
# 0.5156281490822161
(y, log) = gmres(A, b, log = true, restart = size(A, 2), verbose = true, tol = 1e-9)
# julia> log
# Converged after 50 iterations.
# julia> log[:resnorm][end]
# 3.184916424132057e-15




norm(b - A * y)


xe = A \ b
xi ./ xe
