using IterativeSolvers, CSV, Preconditioners, SparseArrays
using AlgebraicMultigrid
using Krylov
using LinearOperators

#=

LHS = CSV.read("lhs.csv", header = false)
LHS = convert(Matrix, LHS)
b = CSV.read("rhs.csv", header = false)
b = convert(Matrix, b)
b = b[:, 1]
x = zeros(size(b))
@show cond(Symmetric(LHS, :L))

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
(n, p, q) = (22, 1, 64)
tol = 1e-8
open("itertry.csv", "w") do iterf
    println(iterf, "cond,minres,,gmres,,gmresrestart,,minreskry,,greskry,,diprecon,,newpreconrestart,")
    rhs = vec(readdlm("rhs.csv"))
    for f in readdir("lhs")
        lhs = convert(Matrix, CSV.read(joinpath("lhs", f), header = false))
        L = Symmetric(lhs, :L)
        print(iterf, 0, ",")

        (xi, log) = minres!(zeros(length(rhs)), L, rhs, log = true, tol = tol)
        print(iterf, log.isconverged, ",")
        print(iterf, norm(rhs - L * xi), ",")

        (xi, log) = gmres!(zeros(length(rhs)), L, rhs, log = true, tol = tol, restart = size(lhs, 2))
        print(iterf, log.isconverged, ",")
        print(iterf, norm(rhs - L * xi), ",")

        (xi, log) = gmres!(zeros(length(rhs)), L, rhs, log = true, tol = tol, restart = div(size(lhs, 2), 2))
        print(iterf, log.isconverged, ",")
        print(iterf, norm(rhs - L * xi), ",")

        (xi, log) = Krylov.minres(L, rhs, atol = tol, rtol = tol)
        print(iterf, log.solved, ",")
        print(iterf, norm(rhs - L * xi), ",")

        (xi, log) = Krylov.dqgmres(L, rhs, atol = tol, rtol = tol)
        print(iterf, log.solved, ",")
        print(iterf, norm(rhs - L * xi), ",")

        (R, C) =  equilibrators(L)
        lprecond = cholesky(Array(Diagonal(map(ri -> isfinite(ri) ? inv(ri) : 1.0, R)))) # ugh preconditioners so messy
        rprecond = cholesky(Array(Diagonal(map(ci -> isfinite(ci) ? inv(ci) : 1.0, C))))
        (xi, log) = gmres!(zeros(length(rhs)), L, rhs, log = true, tol = tol, restart = size(lhs, 2), Pl = lprecond, Pr = rprecond)
        print(iterf, log.isconverged, ",")
        print(iterf, norm(rhs - L * xi), ",")

        ill_cond_block = lhs[(n + 1):end, (n + 1):end]
        AG = lhs[(n + 1):end, 1:n]
        W = I
        preconditioner = [
            W   zeros(n, p + q);
            zeros(p + q, n)   ill_cond_block + AG * (W \ AG');
            ]
        # lhsp = preconditioner \ Symmetric(lhs, :L)
        # rhsp = preconditioner \ rhs
        (xi, log) = gmres!(zeros(length(rhs)), L, rhs, log = true, tol = tol, restart = div(size(L, 2), 2), Pl = lu(preconditioner))
        print(iterf, log.isconverged, ",")
        print(iterf, norm(rhs - L * xi), "\n")
    end
end
