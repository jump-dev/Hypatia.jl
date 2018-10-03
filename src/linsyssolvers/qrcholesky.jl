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
    c
    b
    G
    h
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
        c::Vector{Float64},
        A::AbstractMatrix{Float64},
        b::Vector{Float64},
        G::AbstractMatrix{Float64},
        h::Vector{Float64},
        Q1::AbstractMatrix{Float64},
        Q2::AbstractMatrix{Float64},
        Ri::AbstractMatrix{Float64},
        )

        L = new()
        (n, p, q) = (length(c), length(b), length(h))
        nmp = n - p
        L.c = c
        L.b = b
        L.G = G
        L.h = h
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

    return QRCholCache(c, A, b, G, h, Q1, Q2, Ri)
end

# solve system for x, y, z
function solvelinsys3!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    H::AbstractMatrix{Float64},
    L::QRCholCache,
    )

    F = helpfactorize!(H, L)
    @. rhs_ty *= -1.0
    @. rhs_tz *= -1.0
    helplinsys!(rhs_tx, rhs_ty, rhs_tz, F, L)

    return nothing
end

# solve system for x, y, z, s, kap, tau
function solvelinsys6!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    rhs_ts::Vector{Float64},
    rhs_kap::Float64,
    rhs_tau::Float64,
    mu::Float64,
    tau::Float64,
    H::AbstractMatrix{Float64},
    L::QRCholCache,
    )

    # solve two symmetric systems and combine the solutions
    F = helpfactorize!(H, L)

    # (x2, y2, z2) = (rhs_tx, -rhs_ty, -H*rhs_ts - rhs_tz)
    @. rhs_ty *= -1.0
    @. rhs_tz *= -1.0
    if !iszero(rhs_ts)
        mul!(L.z1, H, rhs_ts)
        @. rhs_tz -= L.z1
    end
    helplinsys!(rhs_tx, rhs_ty, rhs_tz, F, L)

    # (x1, y1, z1) = (-c, b, H*h)
    @. L.x1 = -L.c
    @. L.y1 = L.b
    mul!(L.z1, H, L.h)
    helplinsys!(L.x1, L.y1, L.z1, F, L)

    # combine
    dir_tau = (rhs_tau + rhs_kap + dot(L.c, rhs_tx) + dot(L.b, rhs_ty) + dot(L.h, rhs_tz))/(mu/tau/tau - dot(L.c, L.x1) - dot(L.b, L.y1) - dot(L.h, L.z1))
    @. rhs_tx += dir_tau*L.x1
    @. rhs_ty += dir_tau*L.y1
    @. rhs_tz += dir_tau*L.z1
    mul!(L.z1, L.G, rhs_tx)
    @. rhs_ts = -L.z1 + L.h*dir_tau - rhs_ts
    dir_kap = -dot(L.c, rhs_tx) - dot(L.b, rhs_ty) - dot(L.h, rhs_tz) - rhs_tau

    return (dir_kap, dir_tau)
end

# calculate solution to reduced symmetric linear system
function helplinsys!(
    xi::Vector{Float64},
    yi::Vector{Float64},
    zi::Vector{Float64},
    F,
    L::QRCholCache,
    )

    # bxGHbz = bx + G'*Hbz
    mul!(L.bxGHbz, L.G', zi)
    @. L.bxGHbz += xi
    # Q1x = Q1*Ri'*by
    mul!(L.Q1x, L.RiQ1', yi)
    # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
    mul!(L.rhs, L.GHG, L.Q1x)
    @. L.rhs = L.bxGHbz - L.rhs
    mul!(L.Q2div, L.Q2', L.rhs)
    ldiv!(F, L.Q2div)
    mul!(L.Q2x, L.Q2, L.Q2div)
    # xi = Q1x + Q2x
    @. xi = L.Q1x + L.Q2x
    # yi = Ri*Q1'*(bxGHbz - GHG*xi)
    mul!(L.GHGxi, L.GHG, xi)
    @. L.bxGHbz -= L.GHGxi
    mul!(yi, L.RiQ1, L.bxGHbz)
    # zi = HG*xi - Hbz
    mul!(L.HGxi, L.HG, xi)
    @. zi = L.HGxi - zi

    return nothing
end

# calculate solution to reduced symmetric linear system
function helpfactorize!(
    H::AbstractMatrix{Float64},
    L::QRCholCache,
    )

    # factorize Q2' * G' * H * G * Q2
    mul!(L.HG, H, L.G)
    mul!(L.GHG, L.G', L.HG)
    mul!(L.GHGQ2, L.GHG, L.Q2)
    mul!(L.Q2GHGQ2, L.Q2', L.GHGQ2)

    # bunchkaufman allocates more than cholesky, but doesn't fail when approximately quasidefinite (TODO could try LDL instead)
    # TODO does it matter that F could be either type?
    F = cholesky!(Symmetric(L.Q2GHGQ2), check=false)
    if !issuccess(F)
        # verbose && # TODO pass this in
        println("linear system matrix was not positive definite")
        mul!(L.Q2GHGQ2, L.Q2', L.GHGQ2)
        F = bunchkaufman!(Symmetric(L.Q2GHGQ2))
    end

    return F
end
