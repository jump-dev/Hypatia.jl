using IterativeSolvers, CSV, Preconditioners, SparseArrays
using AlgebraicMultigrid
using Krylov
using LinearOperators

LHS = CSV.read("lhs.csv", header = false)
LHS = convert(Matrix, LHS)
b = CSV.read("rhs.csv", header = false)
b = convert(Matrix, b)
b = b[:, 1]
x = zeros(size(b))
@show cond(Symmetric(LHS, :L))

# size(solver.model.A) = (1, 9)
# size(solver.model.G) = (28, 9)
n = 9; p = 1; q = 28
ill_cond_block = LHS[(n + 1):end, (n + 1):end]
AG = LHS[(n + 1):end, 1:n]
W = I * 10
preconditioner = [
    W   zeros(n, p + q);
    zeros(p + q, n)   ill_cond_block + AG * (W \ AG');
    ]
@show cond(preconditioner \ Symmetric(LHS, :L))

LHS = sparse(LHS)
ml = ruge_stuben(Symmetric(LHS, :L))
p = aspreconditioner(ml)
(xi, log) = minres!(x, Symmetric(LHS, :L), b, log = true, verbose = true)
# (xi, stats) = Krylov.minres(Symmetric(LHS, :L), b, M = LinearOperator(preconditioner))

precLHS = sparse(preconditioner \ Symmetric(LHS, :L))
pecb = preconditioner \ b
x = zeros(size(b))
minres!(x, precLHS, pecb, log = true, verbose = true)

LHS = sparse(LHS)
ml = ruge_stuben(LHS)
p = aspreconditioner(ml)
(xi, log) = gmres(LHS, b, log = true, restart = size(LHS, 2), verbose = true, Pl = p)
xe = LHS \ b
xi ./ xe
# (xi, log) = gmres!(xe, LHS, b, log = true)


using IterativeSolvers, AlgebraicMultigrid, Random
Random.seed!(1)
LHS = sprandn(50, 50, 0.1)
x = randn(50)
b = LHS * x
ml = ruge_stuben(LHS)
p = aspreconditioner(ml)
(y, log) = gmres(LHS, b, log = true, restart = size(LHS, 2), verbose = true, Pl = p, tol = 1e-9)
# julia> log
# Converged after 13 iterations.
# julia> log[:resnorm][end]
# 0.5156281490822161
(y, log) = gmres(LHS, b, log = true, restart = size(LHS, 2), verbose = true, tol = 1e-9)
# julia> log
# Converged after 50 iterations.
# julia> log[:resnorm][end]
# 3.184916424132057e-15




norm(b - LHS * y)


xe = LHS \ b
xi ./ xe
