#=
Copyright 2018, Chris Coey and contributors

caches for various linear system solvers, for precomputation and memory allocation
# TODO put functions for each cache type into separate files in a new folder

=#

abstract type LinSysCache end


#=
CVXOPT method: solve two symmetric linear systems and combine solutions
QR + Cholesky factorization (direct) solver
(1) eliminate equality constraints via QR of A'
(2) solve reduced system by cholesky
|0  A' G'| * |ux| = |bx|
|A  0  0 |   |uy|   |by|
|G  0  M |   |uz|   |bz|
where M = -I (for initial iterate only) or M = -Hi/mu (Hi is Hessian inverse)
=#
mutable struct QRCholCache <: LinSysCache
    Q2
    RiQ1
    HG
    GHG
    GHGQ2
    Q2GHGQ2
    bxGHbz
    Q1x
    rhs
    Q2div
    Q2x
    GHGxi
    HGxi
    x1
    y1
    z1

    function QRCholCache(
        Q1::AbstractMatrix{Float64},
        Q2::AbstractMatrix{Float64},
        Ri::AbstractMatrix{Float64},
        G::AbstractMatrix{Float64},
        n::Int,
        p::Int,
        q::Int,
        )

        L = new()
        nmp = n - p
        L.Q2 = Q2
        L.RiQ1 = Ri*Q1'
        L.HG = Matrix{Float64}(undef, q, n) # TODO don't enforce dense on some
        L.GHG = Matrix{Float64}(undef, n, n)
        L.GHGQ2 = Matrix{Float64}(undef, n, nmp)
        L.Q2GHGQ2 = Matrix{Float64}(undef, nmp, nmp)
        L.bxGHbz = Vector{Float64}(undef, n)
        L.Q1x = Vector{Float64}(undef, n)
        L.rhs = Vector{Float64}(undef, n)
        L.Q2div = Vector{Float64}(undef, nmp)
        L.Q2x = Vector{Float64}(undef, n)
        L.GHGxi = Vector{Float64}(undef, n)
        L.HGxi = Vector{Float64}(undef, q)
        L.x1 = Vector{Float64}(undef, n)
        L.y1 = Vector{Float64}(undef, p)
        L.z1 = Vector{Float64}(undef, q)

        return L
    end
end

function QRCholCache(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    )

    (n, p, q) = (length(c), length(b), length(h))

    # TODO use dispatch, not manually checking issparse. implement the necessary functions in Julia
    if issparse(A)
        println("Julia is currently missing some sparse matrix methods that could improve performance; Hypatia may perform better if A is loaded as a dense matrix")
        # TODO currently using dense Q1, Q2, R - probably some should be sparse
        F = qr(sparse(A'))
        @assert length(F.prow) == n
        @assert length(F.pcol) == p
        @assert istriu(F.R)

        Q = F.Q*Matrix(1.0I, n, n)
        Q1 = zeros(n, p)
        Q1[F.prow, F.pcol] = Q[:, 1:p]
        Q2 = zeros(n, n-p)
        Q2[F.prow, :] = Q[:, p+1:n]
        Ri = zeros(p, p)
        Ri[F.pcol, F.pcol] = inv(UpperTriangular(F.R))
    else
        F = qr(A')
        @assert istriu(F.R)

        Q = F.Q*Matrix(1.0I, n, n)
        Q1 = Q[:, 1:p]
        Q2 = Q[:, p+1:n]
        Ri = inv(UpperTriangular(F.R))
        @assert norm(A'*Ri - Q1) < 1e-8 # TODO delete later
    end

    # # check rank conditions
    # # TODO rank for qr decomp should be implemented in Julia - see https://github.com/JuliaLang/julia/blob/f8b52dab77415a22d28497f48407aca92fbbd4c3/stdlib/LinearAlgebra/src/qr.jl#L895
    # if rank(A) < p # TODO change to rank(F)
    #     error("A matrix is not full-row-rank; some primal equalities may be redundant or inconsistent")
    # end
    # if rank(vcat(A, G)) < n
    #     error("[A' G'] is not full-row-rank; some dual equalities may be redundant (i.e. primal variables can be removed) or inconsistent")
    # end

    return QRCholCache(Q1, Q2, Ri, G, n, p, q)
end

#
function solvesinglelinsys!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    L::QRCholCache,
    )

    (Q2, HG, GHG, GHGQ2, Q2GHGQ2) = (L.Q2, L.HG, L.GHG, L.GHGQ2, L.Q2GHGQ2)

    # solve one symmetric system
    # use QR + cholesky method from CVXOPT
    # (1) eliminate equality constraints via QR of A'
    # (2) solve reduced system by cholesky
    # |0  A' G'  | * |ux| = | bx|
    # |A  0  0   |   |uy|   |-by|
    # |G  0 -H^-1|   |uz|   |-bz|

    # factorize Q2' * G' * mu*H * G * Q2
    mul!(HG, H, G)
    mul!(GHG, G', HG)
    mul!(GHGQ2, GHG, Q2)
    mul!(Q2GHGQ2, Q2', GHGQ2)

    # bunchkaufman allocates more than cholesky, but doesn't fail when approximately quasidefinite (TODO could try LDL instead)
    # TODO does it matter that F could be either type?
    F = cholesky!(Symmetric(Q2GHGQ2), check=false)
    if !issuccess(F)
        # verbose && # TODO pass this in
        println("linear system matrix was not positive definite")
        mul!(Q2GHGQ2, Q2', GHGQ2)
        F = bunchkaufman!(Symmetric(Q2GHGQ2))
    end

    @. rhs_ty *= -1.0
    @. rhs_tz *= -1.0
    solvereducedlinsys!(rhs_tx, rhs_ty, rhs_tz, F, G, L)

    return nothing
end

#
function solvedoublelinsys!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    rhs_ts::Vector{Float64},
    rhs_kap::Float64,
    rhs_tau::Float64,
    mu::Float64,
    tau::Float64,
    H::AbstractMatrix{Float64},
    c::Vector{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    L::QRCholCache,
    )

    (Q2, HG, GHG, GHGQ2, Q2GHGQ2, x1, y1, z1) = (L.Q2, L.HG, L.GHG, L.GHGQ2, L.Q2GHGQ2, L.x1, L.y1, L.z1)

    # solve two symmetric systems and combine the solutions
    # use QR + cholesky method from CVXOPT
    # (1) eliminate equality constraints via QR of A'
    # (2) solve reduced system by cholesky
    # |0  A' G'  | * |ux| = |bx|
    # |A  0  0   |   |uy|   |by|
    # |G  0 -H^-1|   |uz|   |bz|

    # A' = [Q1 Q2] * [R1; 0]
    # factorize Q2' * G' * H * G * Q2
    mul!(HG, H, G)
    mul!(GHG, G', HG)
    mul!(GHGQ2, GHG, Q2)
    mul!(Q2GHGQ2, Q2', GHGQ2)

    # bunchkaufman allocates more than cholesky, but doesn't fail when approximately quasidefinite (TODO could try LDL instead)
    # TODO does it matter that F could be either type?
    F = cholesky!(Symmetric(Q2GHGQ2), check=false)
    if !issuccess(F)
        # verbose && # TODO pass this in
        println("linear system matrix was not positive definite")
        mul!(Q2GHGQ2, Q2', GHGQ2)
        F = bunchkaufman!(Symmetric(Q2GHGQ2))
    end

    # (x2, y2, z2) = (rhs_tx, -rhs_ty, -H*rhs_ts - rhs_tz)
    @. rhs_ty *= -1.0
    @. rhs_tz *= -1.0
    if !iszero(rhs_ts)
        mul!(z1, H, rhs_ts)
        @. rhs_tz -= z1
    end
    solvereducedlinsys!(rhs_tx, rhs_ty, rhs_tz, F, G, L)

    # (x1, y1, z1) = (-c, b, H*h)
    @. x1 = -c
    @. y1 = b
    mul!(z1, H, h)
    solvereducedlinsys!(x1, y1, z1, F, G, L)

    # combine
    dir_tau = (rhs_tau + rhs_kap + dot(c, rhs_tx) + dot(b, rhs_ty) + dot(h, rhs_tz))/(mu/tau/tau - dot(c, x1) - dot(b, y1) - dot(h, z1))
    @. rhs_tx += dir_tau*x1
    @. rhs_ty += dir_tau*y1
    @. rhs_tz += dir_tau*z1
    mul!(z1, G, rhs_tx)
    @. rhs_ts = -z1 + h*dir_tau - rhs_ts
    dir_kap = -dot(c, rhs_tx) - dot(b, rhs_ty) - dot(h, rhs_tz) - rhs_tau

    return (dir_kap, dir_tau)
end

# calculate solution to reduced symmetric linear system
function solvereducedlinsys!(
    xi::Vector{Float64},
    yi::Vector{Float64},
    zi::Vector{Float64},
    F,
    G::AbstractMatrix{Float64},
    L::QRCholCache,
    )

    (Q2, RiQ1, HG, GHG, bxGHbz, Q1x, rhs, Q2div, Q2x, GHGxi, HGxi) = (L.Q2, L.RiQ1, L.HG, L.GHG, L.bxGHbz, L.Q1x, L.rhs, L.Q2div, L.Q2x, L.GHGxi, L.HGxi)

    # bxGHbz = bx + G'*Hbz
    mul!(bxGHbz, G', zi)
    @. bxGHbz += xi
    # Q1x = Q1*Ri'*by
    mul!(Q1x, RiQ1', yi)
    # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
    mul!(rhs, GHG, Q1x)
    @. rhs = bxGHbz - rhs
    mul!(Q2div, Q2', rhs)
    ldiv!(F, Q2div)
    mul!(Q2x, Q2, Q2div)
    # xi = Q1x + Q2x
    @. xi = Q1x + Q2x
    # yi = Ri*Q1'*(bxGHbz - GHG*xi)
    mul!(GHGxi, GHG, xi)
    @. bxGHbz -= GHGxi
    mul!(yi, RiQ1, bxGHbz)
    # zi = HG*xi - Hbz
    mul!(HGxi, HG, xi)
    @. zi = HGxi - zi

    return nothing
end
