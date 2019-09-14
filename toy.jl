using SparseArrays, Random, LinearAlgebra, Pardiso
n = 100;
A = sprandn(n, n, 0.1);
B = A * randn(n, 2);
X = zeros(n, 2)

ps = PardisoSolver()
# At = get_matrix(ps, A, :T)
set_phase!(ps, Pardiso.ANALYSIS)
pardiso(ps, X, A, B)
set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
pardiso(ps, X, A, B)
@show norm(B - A' * X)

B = A * randn(n, 2);
set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
pardiso(ps, X, A, B)
@show norm(B - A' * X)


set_phase!(ps, Pardiso.RELEASE_ALL)
pardiso(ps, X, A, B)



using SparseArrays, Pardiso, LinearAlgebra
n = 100
A = sprandn(n, n, 0.1)
B = A * rand(n, 2)
B_old = copy(B)
X = zeros(n, 2)
ps = PardisoSolver()
pardisoinit(ps)
set_iparm!(ps, 1, 1)
set_iparm!(ps, 12, 1)
set_iparm!(ps, 6, 1)
pardiso(ps, X, A, B)
@show norm(B_old - A * B)
set_phase!(ps, Pardiso.RELEASE_ALL)
pardiso(ps)
