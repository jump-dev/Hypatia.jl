# This is an example script demonstrating how PARDISO works on a small,
# sparse, real symmetric matrix. It computes the m solutions X to the
# collection of linear systems
#
#    A * X = B
#
# using the PARDISO solver, where A is a symmetric n x n matrix, B is an
# n x m matrix, and X is another n x m matrix.
using Pardiso
using SparseArrays
using Random
using Printf
using Test

verbose = true
n       = 4  # The number of equations.
m       = 3  # The number of right-hand sides.
A = sparse([ 1. 0 -2  3
             0  5  1  2
            -2  1  4 -7
             3  2 -7  5 ])
B = rand(n,m)
ps = PardisoSolver()
if verbose
    set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
end
set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
pardisoinit(ps)
fix_iparm!(ps, :N)
A_pardiso = get_matrix(ps, A, :N)
set_phase!(ps, Pardiso.ANALYSIS)
set_perm!(ps, randperm(n))
pardiso(ps, A_pardiso, B)
@printf("The factors have %d nonzero entries.\n", get_iparm(ps, 18))
set_phase!(ps, Pardiso.NUM_FACT)
pardiso(ps, A_pardiso, B)
set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)

julia> X = similar(B)
malloc(): invalid size (unsorted)

signal (6): Aborted
in expression starting at REPL[24]:1




julia> get_iparm(ps, 23)
free(): corrupted unsorted chunks

signal (6): Aborted
in expression starting at none:0


# @printf("The matrix has %d positive and %d negative eigenvalues.\n",
#         get_iparm(ps, 22), get_iparm(ps, 23))

# Compute the solutions X using the symbolic factorization.
# X = similar(B) # Solution is stored in X
# pardiso(ps, X, A_pardiso, B)
# @printf("PARDISO performed %d iterative refinement steps.\n", get_iparm(ps, 7))
#
# # Compute the residuals.
# R = maximum(abs.(A*X - B))
# @printf("The maximum residual for the solution X is %0.3g.\n", R)
# @test R < 1e-10
#
# # Free the PARDISO data structures.
# set_phase!(ps, Pardiso.RELEASE_ALL)
# pardiso(ps)
