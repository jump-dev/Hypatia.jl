using SparseArrays, LinearAlgebra, SuiteSparse, Printf

n_range = [400, 800, 1500, 5000, 8000]
p_range = [1, 10, 100, 500, 1000, 5000]
q_range = [400, 800, 1500, 5000, 8000]
f_range = [1e-5, 1e-3, 1e-2, 1e-1]

for n in n_range, p in p_range, q in q_range, fa in f_range, fg in f_range
    A = sprandn(p, n, fa)
    G = sprandn(q, n, fg)
    F = qr(sparse(A'))
    Q = F.Q
    GQ2 = (Matrix(G) * Q)[:, (p + 1):end]
    M = sparse(GQ2' * GQ2)
    @printf("n = %3d p = %3d q = %3d fa = %3g fg = %3g \n", n, p, q, fa, fg)
    @assert issparse(M)
end
