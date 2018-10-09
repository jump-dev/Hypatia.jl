#=
CVXOPT method: solve two symmetric linear systems and combine solutions
QR plus either Cholesky factorization or iterative conjugate gradients method
(1) eliminate equality constraints via QR of A'
(2) solve reduced system by Cholesky or iterative method
|0  A' G'| * |ux| = |bx|
|A  0  0 |   |uy|   |by|
|G  0  M |   |uz|   |bz|
where M = -I (for initial iterate only) or M = -Hi (Hi is Hessian inverse, pre-scaled by 1/mu)
=#
mutable struct QRSymmCache <: LinSysCache
    # TODO can remove some of the intermediary prealloced arrays after github.com/JuliaLang/julia/issues/23919 is resolved
    useiterative
    cone
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
    # for iterative only
    cgstate
    lprecond
    Q2sol

    function QRSymmCache(
        c::Vector{Float64},
        A::AbstractMatrix{Float64},
        b::Vector{Float64},
        G::AbstractMatrix{Float64},
        h::Vector{Float64},
        cone::Cone,
        Q2::AbstractMatrix{Float64},
        RiQ1::AbstractMatrix{Float64};
        useiterative::Bool = false,
        )

        L = new()
        (n, p, q) = (length(c), length(b), length(h))
        nmp = n - p
        L.useiterative = useiterative
        L.cone = cone
        L.c = c
        L.b = b
        L.G = G
        L.h = h
        L.Q2 = Q2
        L.RiQ1 = RiQ1
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
        if useiterative
            cgu = zeros(nmp)
            L.cgstate = IterativeSolvers.CGStateVariables{Float64, Vector{Float64}}(cgu, similar(cgu), similar(cgu))
            L.lprecond = IterativeSolvers.Identity()
            L.Q2sol = zeros(nmp)
        end

        return L
    end
end

QRSymmCache(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone;
    useiterative::Bool = false,
    ) = error("to use a QRSymmCache for linear system solves, the data must be preprocessed and Q2 and RiQ1 must be passed into the QRSymmCache constructor")

# solve system for x, y, z
function solvelinsys3!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    mu::Float64,
    L::QRSymmCache;
    identityH::Bool = false,
    )

    F = helplhs!(mu, L, identityH=identityH)
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
    L::QRSymmCache,
    )

    # solve two symmetric systems and combine the solutions
    F = helplhs!(mu, L)

    # (x2, y2, z2) = (rhs_tx, -rhs_ty, -H*rhs_ts - rhs_tz)
    @. rhs_ty *= -1.0
    @. rhs_tz *= -1.0
    if !iszero(rhs_ts)
        calcHarr!(L.z1, rhs_ts, L.cone)
        @. rhs_tz -= mu*L.z1
    end
    helplinsys!(rhs_tx, rhs_ty, rhs_tz, F, L)

    # (x1, y1, z1) = (-c, b, H*h)
    @. L.x1 = -L.c
    @. L.y1 = L.b
    calcHarr!(L.z1, L.h, L.cone)
    @. L.z1 *= mu
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
    L::QRSymmCache,
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

    if L.useiterative
        # TODO use previous solution from same pred/corr (requires knowing if in pred or corr step, and having two Q2sol objects)
        # TODO allow tol to be loose in early iterations and tighter later (maybe depend on mu)
        IterativeSolvers.cg!(L.Q2sol, F, L.Q2div, maxiter=10000, tol=1e-13, statevars=L.cgstate, Pl=L.lprecond, verbose=false)
        @. L.Q2div = L.Q2sol
    else
        ldiv!(F, L.Q2div)
    end

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

# calculate or factorize LHS of symmetric positive definite linear system
function helplhs!(
    mu::Float64,
    L::QRSymmCache;
    identityH::Bool = false,
    )

    # Q2' * G' * H * G * Q2
    if identityH
        @. L.HG = L.G
    else
        calcHarr!(L.HG, L.G, L.cone)
        @. L.HG *= mu
    end
    mul!(L.GHG, L.G', L.HG)
    mul!(L.GHGQ2, L.GHG, L.Q2)
    mul!(L.Q2GHGQ2, L.Q2', L.GHGQ2)

    if L.useiterative
        return Symmetric(L.Q2GHGQ2)
    else
        # bunchkaufman allocates more than cholesky, but doesn't fail when approximately quasidefinite
        # TODO does it matter that F could be either type?
        F = cholesky!(Symmetric(L.Q2GHGQ2), Val(true), check=false)
        if !isposdef(F)
            # verbose && # TODO pass this in
            mul!(L.Q2GHGQ2, L.Q2', L.GHGQ2)
            F = bunchkaufman!(Symmetric(L.Q2GHGQ2), true, check=false) # TODO remove allocs, need to use low-level functions
            if !issuccess(F)
                error("linear system matrix was not positive definite")
            end
        end
        # F = bunchkaufman!(Symmetric(L.Q2GHGQ2)) # TODO remove allocs, need to use low-level functions
        return F
    end
end
