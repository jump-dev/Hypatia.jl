using IterativeSolvers, CSV, Preconditioners, SparseArrays
using AlgebraicMultigrid
using Krylov
using LinearOperators
using DelimitedFiles
using LinearAlgebra
import Krylov


LHS = CSV.read("lhs.csv", header = false)
LHS = convert(Matrix, LHS)
b = CSV.read("rhs.csv", header = false)
b = convert(Matrix, b)
i = 1
b = b[:, i]
prevsol = CSV.read("prevsol.csv", header = false)
prevsol = convert(Matrix, prevsol)
prevsol = prevsol[:, i]
A = Symmetric(LHS, :L)
(x, hist) = minres(A, b, log = true, reorth = true)
norm(b - A * x)

# maxels = maximum(A, dims = 2)
# maxels = map(x -> iszero(x) ? 1.0 : x, maxels)
# p = Diagonal(maxels[:])
# maxels = maximum(A, dims = 1)
# maxels = map(x -> iszero(x) ? 1.0 : x, maxels)
# c = Diagonal(maxels[:])
# n = 4; pq = 11;
# ill_cond_block = Symmetric(A[(n + 1):end, (n + 1):end])
# AG = A[(n + 1):end, 1:n]
# W = I
# preconditioner = Symmetric([
#     W   zeros(n, pq);
#     zeros(pq, n)   ill_cond_block + AG * AG';
#     ])
# y = Krylov.minres(A, b, verbose = true)


#=
# size(solver.model.A) = (1, 9)
# size(solver.model.G) = (28, 9)
n = 19; p = 1; q = 58
ill_cond_block = LHS[(n + 1):end, (n + 1):end]
AG = LHS[(n + 1):end, 1:n]
W = I
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
for n in [50, 500, 5_000, 50_000, 100_000]
    println(n)
    Random.seed!(1)
    LHS = sprandn(n, n, 0.1)
    x = randn(n)
    b = LHS * x
    # ml = ruge_stuben(LHS)
    # p = aspreconditioner(ml)
    # (y, log) = gmres(LHS, b, log = true, restart = size(LHS, 2), verbose = true, Pl = p, tol = 1e-9)
    d = diag(LHS)
    d = map(di -> (iszero(di) ? 1.0 : di), d)
    LHSp = Diagonal(d) \ LHS
    bp = Array(Diagonal(d)) \ b
    t2 = @timed (_, log) = gmres(LHS, b, log = true, restart =20, verbose = false);
    @show log.isconverged
    println(t2[2], t2[3])
    t1 = @timed (_, log) = gmres(LHS, b, log = true, restart = size(LHS, 2), verbose = false);
    @show log.isconverged
    println(t1[2], t1[3])
end

=#


function equilibrators(A::AbstractMatrix{T}) where {T}
    abs1(x::Real) = abs(x)
    abs1(x::Complex) = abs(real(x)) + abs(imag(x))
    m,n = size(A)
    R = zeros(T,m)
    C = zeros(T,n)
    @inbounds for j=1:n
        R .= max.(R,view(A,:,j))
    end
    @inbounds for i=1:m
        if R[i] > 0
            R[i] = T(2)^floor(Int,log2(R[i]))
        end
    end
    R .= 1 ./ R
    @inbounds for i=1:m
        C .= max.(C,R[i] * view(A,i,:))
    end
    @inbounds for j=1:n
        if C[j] > 0
            C[j] = T(2)^floor(Int,log2(C[j]))
        end
    end
    C .= 1 ./ C
    R,C
end

# no warm starts
(n, p, q) = (270, 1, 960)
tol = 1e-8
restart_freq = div(n + p + q, 4)
maxiter = restart_freq * 5
open("itertry.csv", "w") do iterf
    println(iterf, "cond,direct,,minres,,,," *
        "gmres,,,,gmresrestart,,,," *
        "minresprecon,,,," *
        # "minreskry,,,,greskry,,,," *
        # "diprecongmres,,,,gemresmanual,,,,minresmanual,,,,newprecongmres,,,"
        "")
    rhs = vec(readdlm("rhs.csv"))
    idx = 0
    for f in readdir("lhs")
        idx += 1
        println(idx)

        lhs = convert(Matrix, CSV.read(joinpath("lhs", f), header = false))
        L = Symmetric(lhs, :L)
        print(iterf, 0, ",")

        # direct
        xi = zeros(length(rhs))
        t = @timed begin
            F = bunchkaufman(Symmetric(lhs, :L), true, check = true)
            ldiv!(xi, F, rhs)
        end
        print(iterf, "$(t[2]),$(t[3]),")

        # minres
        xi = zeros(length(rhs))
        t = @timed minres!(xi, L, rhs, tol = tol)
        isconverged = (norm(rhs - L * xi) < 1e-5)
        print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi)),")

        # unrestarted gmres
        xi = zeros(length(rhs))
        t = @timed gmres!(xi, L, rhs, tol = tol, restart = size(L, 2))
        isconverged = (norm(rhs - L * xi) < 1e-5)
        print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi)),")

        # restarted gmres
        xi = zeros(length(rhs))
        t = @timed gmres!(xi, L, rhs, tol = tol, restart = restart_freq, maxiter = maxiter)
        isconverged = (norm(rhs - L * xi) < 1e-5)
        print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi)),")


        # t = @timed (xi, _) = Krylov.minres(L, rhs, atol = tol, rtol = tol)
        # isconverged = (norm(rhs - L * xi) < 1e-5)
        # print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi)),")
        #
        # t = @timed (xi, _) = Krylov.dqgmres(L, rhs, atol = tol, rtol = tol)
        # isconverged = (norm(rhs - L * xi) < 1e-5)
        # print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi)),")

        # (R, C) =  equilibrators(L)
        # lprecond = cholesky(Array(Diagonal(map(ri -> isfinite(ri) ? inv(ri) : 1.0, R)))) # ugh preconditioners so messy
        # rprecond = cholesky(Array(Diagonal(map(ci -> isfinite(ci) ? inv(ci) : 1.0, C))))
        # xi = zeros(length(rhs))
        # t = @timed gmres!(xi, L, rhs, tol = tol, restart = restart_freq, Pl = lprecond, Pr = rprecond, maxiter = maxiter)
        # isconverged = (norm(rhs - L * xi) < 1e-5)
        # print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi)),")
        #
        tcond = @timed begin
            ill_cond_block = lhs[(n + 1):end, (n + 1):end]
            AG = lhs[(n + 1):end, 1:n]
            W = I
            preconditioner = [
                W   zeros(n, p + q);
                zeros(p + q, n)   ill_cond_block + AG * (W \ AG');
                ]
            Lp = preconditioner \ L
            rhsp = preconditioner \ rhs
        end
        @show (tcond[2], tcond[3])

        # preconditioned minres
        xi = zeros(length(rhs))
        t = @timed minres!(xi, Lp, rhsp, tol = tol, maxiter = maxiter)
        isconverged = (norm(rhs - L * xi) < 1e-5)
        print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi)),")


        # xi = zeros(length(rhs))
        # t = @timed gmres!(xi, Lp, rhsp, tol = tol, restart = restart_freq, maxiter = maxiter)
        # isconverged = (norm(rhs - L * xi) < 1e-5)
        # print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi)),")
        #
        # xi = zeros(length(rhs))
        # t = @timed minres!(xi, Lp, rhsp, tol = tol)
        # isconverged = (norm(rhs - L * xi) < 1e-5)
        # print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi)),")
        #
        # xi = zeros(length(rhs))
        # t = @timed gmres!(xi, L, rhs, tol = tol, restart = restart_freq, Pl = lu(preconditioner), maxiter = maxiter)
        # isconverged = (norm(rhs - L * xi) < 1e-5)
        # print(iterf, "$isconverged,$(t[2]),$(t[3]),$(norm(rhs - L * xi))\n")

        print(iterf, "\n")
    end
end

# using BenchmarkTools, IterativeSolvers
# n = 100
# A = randn(n, n)
# A = A + A'
# x = randn(n)
# b = A * x
# sol = zeros(n)
#
# t1 = @timed ldiv!(sol, lu(A), b)
#
# sol = zeros(n)
# t2 = @timed minres!(sol, A, b)
#
# sol = zeros(n)
# t3 = @timed gmres!(sol, A, b)
