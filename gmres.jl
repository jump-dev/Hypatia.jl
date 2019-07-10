# copyright

import Base: iterate
using Printf
export gmres, gmres!

struct ArnoldiDecomp{T, matT}
    A::matT
    V::Matrix{T} # Orthonormal basis vectors
    H::Matrix{T} # Hessenberg matrix
end

ArnoldiDecomp(A::matT, order::Int, T::Type) where {matT} = ArnoldiDecomp{T, matT}(
    A,
    zeros(T, size(A, 1), order + 1),
    zeros(T, order + 1, order)
)

mutable struct Residual{numT, resT}
    current::resT # Current, absolute, preconditioned residual
    accumulator::resT # Used to compute the residual on the go
    nullvec::Vector{numT} # Vector in the null space of H to compute residuals
    β::resT # the initial residual
end

Residual(order, T::Type) = Residual{T, real(T)}(
    one(real(T)),
    one(real(T)),
    ones(T, order + 1),
    one(real(T))
)

mutable struct GMRESIterable{preclT, precrT, solT, rhsT, vecT, arnoldiT <: ArnoldiDecomp, residualT <: Residual, resT <: Real}
    Pl::preclT
    Pr::precrT
    x::solT
    b::rhsT
    Ax::vecT # Some room to work in.

    arnoldi::arnoldiT
    residual::residualT

    mv_products::Int
    restart::Int
    k::Int
    maxiter::Int
    reltol::resT
    β::resT

    V
end

converged(g::GMRESIterable) = g.residual.current ≤ g.reltol

start(::GMRESIterable) = 0

done(g::GMRESIterable, iteration::Int) = iteration ≥ g.maxiter || converged(g)

function iterate(g::GMRESIterable, iteration::Int=start(g))
    if done(g, iteration) return nothing end

    # Arnoldi step: expand
    expand!(g.arnoldi, g.Pl, g.Pr, g.k, g.Ax)
    g.mv_products += 1

    # Orthogonalize V[:, k + 1] w.r.t. V[:, 1 : k]
    g.arnoldi.H[g.k + 1, g.k] = orthogonalize_and_normalize!(
        view(g.arnoldi.V, :, 1 : g.k),
        view(g.arnoldi.V, :, g.k + 1),
        view(g.arnoldi.H, 1 : g.k, g.k)
    )

    # Implicitly computes the residual
    update_residual!(g.residual, g.arnoldi, g.k)

    g.k += 1

    # Computation of x only at the end of the iterations
    # and at restart.
    if g.k == g.restart + 1 || done(g, iteration + 1)

        # Solve the projected problem Hy = β * e1 in the least-squares sense
        rhs = solve_least_squares!(g.arnoldi, g.β, g.k)

        # And improve the solution x ← x + Pr \ (V * y)
        update_solution!(g.x, view(rhs, 1 : g.k - 1), g.arnoldi, g.Pr, g.k, g.Ax)

        g.k = 1

        # Restart when not done.
        if !done(g, iteration)

            # Set the first basis vector
            g.β = init!(g.arnoldi, g.x, g.b, g.Pl, g.Ax)

            # And initialize the residual
            init_residual!(g.residual, g.β)

            g.mv_products += 1
        end
    end

    g.residual.current, iteration + 1
end

function gmres_iterable!(x, A, b;
    Pl = Identity(),
    Pr = Identity(),
    tol = sqrt(eps(real(eltype(b)))),
    restart::Int = min(20, size(A, 2)),
    maxiter::Int = size(A, 2),
    initially_zero::Bool = false
)
    T = eltype(x)

    # Approximate solution
    arnoldi = ArnoldiDecomp(A, restart, T)
    residual = Residual(restart, T)
    mv_products = initially_zero ? 1 : 0

    # Workspace vector to reduce the # allocs.
    Ax = similar(x)
    residual.current = init!(arnoldi, x, b, Pl, Ax, initially_zero = initially_zero)
    init_residual!(residual, residual.current)

    reltol = tol * residual.current

    GMRESIterable(Pl, Pr, x, b, Ax,
        arnoldi, residual,
        mv_products, restart, 1, maxiter, reltol, residual.current, zeros(size(Ax))
    )
end

"""
    gmres(A, b; kwargs...) -> x, [history]
Same as [`gmres!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
gmres(A, b; kwargs...) = gmres!(zerox(A, b), A, b; initially_zero = true, kwargs...)

"""
    gmres!(x, A, b; kwargs...) -> x, [history]
Solves the problem ``Ax = b`` with restarted GMRES.
# Arguments
- `x`: Initial guess, will be updated in-place;
- `A`: linear operator;
- `b`: right-hand side.
## Keywords
- `initially_zero::Bool`: If `true` assumes that `iszero(x)` so that one
  matrix-vector product can be saved when computing the initial
  residual vector;
- `tol`: relative tolerance;
- `restart::Int = min(20, size(A, 2))`: restarts GMRES after specified number of iterations;
- `maxiter::Int = size(A, 2)`: maximum number of inner iterations of GMRES;
- `Pl`: left preconditioner;
- `Pr`: right preconditioner;
- `log::Bool`: keep track of the residual norm in each iteration;
- `verbose::Bool`: print convergence information during the iterations.
# Return values
**if `log` is `false`**
- `x`: approximate solution.
**if `log` is `true`**
- `x`: approximate solution;
- `history`: convergence history.
"""
function gmres!(x, A, b;
  Pl = Identity(),
  Pr = Identity(),
  tol = sqrt(eps(real(eltype(b)))),
  restart::Int = min(20, size(A, 2)),
  maxiter::Int = size(A, 2),
  log::Bool = false,
  initially_zero::Bool = false,
  verbose::Bool = false
)
    history = ConvergenceHistory(partial = !log, restart = restart)
    history[:tol] = tol
    log && reserve!(history, :resnorm, maxiter)

    iterable = gmres_iterable!(x, A, b; Pl = Pl, Pr = Pr, tol = tol, maxiter = maxiter, restart = restart, initially_zero = initially_zero)

    verbose && @printf("=== gmres ===\n%4s\t%4s\t%7s\n","rest","iter","resnorm")

    for (iteration, residual) = enumerate(iterable)
        if log
            nextiter!(history)
            history.mvps = iterable.mv_products
            push!(history, :resnorm, residual)
        end

        verbose && @printf("%3d\t%3d\t%1.2e\n", 1 + div(iteration - 1, restart), 1 + mod(iteration - 1, restart), residual)
    end

    verbose && println()
    setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (x, history) : x, iterable.arnoldi.V
end

function update_residual!(r::Residual, arnoldi::ArnoldiDecomp, k::Int)
    # Cheaply computes the current residual
    r.nullvec[k + 1] = -conj(dot(view(r.nullvec, 1 : k), view(arnoldi.H, 1 : k, k)) / arnoldi.H[k + 1, k])
    r.accumulator += abs2(r.nullvec[k + 1])
    r.current = r.β / √r.accumulator
end

function init!(arnoldi::ArnoldiDecomp{T}, x, b, Pl, Ax; initially_zero::Bool = false) where {T}
    # Initialize the Krylov subspace with the initial residual vector
    # This basically does V[1] = Pl \ (b - A * x) and then normalize

    first_col = view(arnoldi.V, :, 1)

    copyto!(first_col, b)

    # Potentially save one MV product
    if !initially_zero
        mul!(Ax, arnoldi.A, x)
        first_col .-= Ax
    end

    ldiv!(Pl, first_col)

    # Normalize
    β = norm(first_col)
    first_col .*= inv(β)
    β
end

@inline function init_residual!(r::Residual{numT, resT}, β) where {numT,resT}
    r.accumulator = one(resT)
    r.β = β
end

function solve_least_squares!(arnoldi::ArnoldiDecomp{T}, β, k::Int) where {T}
    # Compute the least-squares solution to Hy = β e1 via Given's rotations
    rhs = zeros(T, k)
    rhs[1] = β

    H = FastHessenberg(view(arnoldi.H, 1 : k, 1 : k - 1))
    ldiv!(H, rhs)

    rhs
end

function update_solution!(x, y, arnoldi::ArnoldiDecomp{T}, Pr::Identity, k::Int, Ax) where {T}
    # Update x ← x + V * y

    # TODO: find the SugarBLAS alternative
    BLAS.gemv!('N', one(T), view(arnoldi.V, :, 1 : k - 1), y, one(T), x)
end

function update_solution!(x, y, arnoldi::ArnoldiDecomp{T}, Pr, k::Int, Ax) where {T}
    # Computing x ← x + Pr \ (V * y) and use Ax as a work space
    mul!(Ax, view(arnoldi.V, :, 1 : k - 1), y)
    ldiv!(Pr, Ax)
    x .+= Ax
end

function expand!(arnoldi::ArnoldiDecomp, Pl::Identity, Pr::Identity, k::Int, Ax)
    # Simply expands by A * v without allocating
    mul!(view(arnoldi.V, :, k + 1), arnoldi.A, view(arnoldi.V, :, k))
end

function expand!(arnoldi::ArnoldiDecomp, Pl, Pr::Identity, k::Int, Ax)
    # Expands by Pl \ (A * v) without allocating
    nextV = view(arnoldi.V, :, k + 1)
    mul!(nextV, arnoldi.A, view(arnoldi.V, :, k))
    ldiv!(Pl, nextV)
end

function expand!(arnoldi::ArnoldiDecomp, Pl, Pr, k::Int, Ax)
    # Expands by Pl \ (A * (Pr \ v)). Avoids allocation by using Ax.
    nextV = view(arnoldi.V, :, k + 1)
    ldiv!(nextV, Pr, view(arnoldi.V, :, k))
    mul!(Ax, arnoldi.A, nextV)
    copyto!(nextV,  Ax)
    ldiv!(Pl, nextV)
end

export DGKS, ClassicalGramSchmidt, ModifiedGramSchmidt
export orthogonalize_and_normalize!

abstract type OrthogonalizationMethod end
struct DGKS <: OrthogonalizationMethod end
struct ClassicalGramSchmidt <: OrthogonalizationMethod end
struct ModifiedGramSchmidt <: OrthogonalizationMethod end

# Default to MGS, good enough for solving linear systems.
@inline orthogonalize_and_normalize!(V::StridedMatrix{T}, w::StridedVector{T}, h::StridedVector{T}) where {T} = orthogonalize_and_normalize!(V, w, h, ModifiedGramSchmidt)

function orthogonalize_and_normalize!(V::StridedMatrix{T}, w::StridedVector{T}, h::StridedVector{T}, ::Type{DGKS}) where {T}
    # Orthogonalize using BLAS-2 ops
    mul!(h, adjoint(V), w)
    BLAS.gemv!('N', -one(T), V, h, one(T), w)
    nrm = norm(w)

    # Constant used by ARPACK.
    η = one(real(T)) / √2

    projection_size = norm(h)

    # Repeat as long as the DGKS condition is satisfied
    # Typically this condition is true only once.
    while nrm < η * projection_size
        correction = adjoint(V)*w
        projection_size = norm(correction)
        # w = w - V * correction
        BLAS.gemv!('N', -one(T), V, correction, one(T), w)
        h .+= correction
        nrm = norm(w)
    end

    # Normalize; note that we already have norm(w).
    w .*= inv(nrm)

    nrm
end

function orthogonalize_and_normalize!(V::StridedMatrix{T}, w::StridedVector{T}, h::StridedVector{T}, ::Type{ClassicalGramSchmidt}) where {T}
    # Orthogonalize using BLAS-2 ops
    mul!(h, adjoint(V), w)
    BLAS.gemv!('N', -one(T), V, h, one(T), w)
    nrm = norm(w)

    # Normalize
    w .*= inv(nrm)

    nrm
end

function orthogonalize_and_normalize!(V::StridedVector{Vector{T}}, w::StridedVector{T}, h::StridedVector{T}, ::Type{ModifiedGramSchmidt}) where {T}
    # Orthogonalize using BLAS-1 ops
    for i = 1 : length(V)
        h[i] = dot(V[i], w)
        w .-= h[i] .* V[i]
    end

    # Normalize
    nrm = norm(w)
    w .*= inv(nrm)

    nrm
end

function orthogonalize_and_normalize!(V::StridedMatrix{T}, w::StridedVector{T}, h::StridedVector{T}, ::Type{ModifiedGramSchmidt}) where {T}
    # Orthogonalize using BLAS-1 ops and column views.
    for i = 1 : size(V, 2)
        column = view(V, :, i)
        h[i] = dot(column, w)
        w .-= h[i] .* column
    end

    nrm = norm(w)
    w .*= inv(nrm)

    nrm
end

import LinearAlgebra: Givens, givensAlgorithm
import LinearAlgebra.ldiv!

struct FastHessenberg{T<:AbstractMatrix}
    H::T # H is assumed to be Hessenberg of size (m + 1) × m
end

@inline Base.size(H::FastHessenberg, args...) = size(H.H, args...)

"""
Solve Hy = rhs for a non-square Hessenberg matrix.
Note that `H` is also modified as is it converted
to an upper triangular matrix via Given's rotations
"""
function ldiv!(H::FastHessenberg, rhs)
    # Implicitly computes H = QR via Given's rotations
    # and then computes the least-squares solution y to
    # |Hy - rhs| = |QRy - rhs| = |Ry - Q'rhs|

    width = size(H, 2)

    # Hessenberg -> UpperTriangular; also apply to r.h.s.
    @inbounds for i = 1 : width
        c, s, _ = givensAlgorithm(H.H[i, i], H.H[i + 1, i])

        # Skip the first sub-diagonal since it'll be zero by design.
        H.H[i, i] = c * H.H[i, i] + s * H.H[i + 1, i]

        # Remaining columns
        @inbounds for j = i + 1 : width
            tmp = -conj(s) * H.H[i, j] + c * H.H[i + 1, j]
            H.H[i, j] = c * H.H[i, j] + s * H.H[i + 1, j]
            H.H[i + 1, j] = tmp
        end

        # Right hand side
        tmp = -conj(s) * rhs[i] + c * rhs[i + 1]
        rhs[i] = c * rhs[i] + s * rhs[i + 1]
        rhs[i + 1] = tmp
    end

    # Solve the upper triangular problem.
    U = UpperTriangular(view(H.H, 1 : width, 1 : width))
    ldiv!(U, view(rhs, 1 : width))
    nothing
end
