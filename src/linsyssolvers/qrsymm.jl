#=
CVXOPT method: solve two symmetric linear systems and combine solutions
QR plus either Cholesky factorization or iterative conjugate gradients method
(1) eliminate equality constraints via QR of A'
(2) solve reduced symmetric system by Cholesky or iterative method
=#
mutable struct QRSymmCache <: LinSysCache
    # TODO can remove some of the intermediary prealloced arrays after github.com/JuliaLang/julia/issues/23919 is resolved
    useiterative
    userefine

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
    Q2divcopy
    Q2x
    GHGxi
    HGxi

    zi
    z1
    z2
    yi
    xi

    lsferr
    lsberr
    lswork
    lsiwork
    lsAF
    lsS

    # for iterative only
    # cgstate
    # lprecond
    # Q2sol

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
        userefine::Bool = false,
        )
        @assert !useiterative # TODO disabled for now

        L = new()
        (n, p, q) = (length(c), length(b), length(h))
        nmp = n - p

        L.useiterative = useiterative
        L.userefine = userefine

        L.cone = cone
        L.c = c
        L.b = b
        L.G = G
        L.h = h
        L.Q2 = Q2
        L.RiQ1 = RiQ1

        # L.GQ2 = G*Q2 # TODO calculate H*(G*Q2)
        L.HG = Matrix{Float64}(undef, q, n) # TODO don't enforce dense on some
        L.GHG = Matrix{Float64}(undef, n, n)
        L.GHGQ2 = Matrix{Float64}(undef, n, nmp)
        L.Q2GHGQ2 = Matrix{Float64}(undef, nmp, nmp)
        L.bxGHbz = Matrix{Float64}(undef, n, 2)
        L.Q1x = Matrix{Float64}(undef, n, 2)
        L.rhs = Matrix{Float64}(undef, n, 2)
        L.Q2div = Matrix{Float64}(undef, nmp, 2)
        L.Q2divcopy = Matrix{Float64}(undef, nmp, 2)
        L.Q2x = Matrix{Float64}(undef, n, 2)
        L.GHGxi = Matrix{Float64}(undef, n, 2)
        L.HGxi = Matrix{Float64}(undef, q, 2)

        L.zi = Matrix{Float64}(undef, q, 2)
        L.z1 = view(L.zi, :, 1)
        L.z2 = view(L.zi, :, 2)
        L.yi = Matrix{Float64}(undef, p, 2)
        L.xi = Matrix{Float64}(undef, n, 2)

        L.lsferr = Vector{Float64}(undef, 2)
        L.lsberr = Vector{Float64}(undef, 2)
        L.lswork = Vector{Float64}(undef, 3*nmp)
        L.lsiwork = Vector{Float64}(undef, nmp)
        L.lsAF = Matrix{Float64}(undef, nmp, nmp)
        L.lsS = Vector{Float64}(undef, nmp)

        # if useiterative
        #     cgu = zeros(nmp)
        #     L.cgstate = IterativeSolvers.CGStateVariables{Float64, Vector{Float64}}(cgu, similar(cgu), similar(cgu))
        #     L.lprecond = IterativeSolvers.Identity()
        #     L.Q2sol = zeros(nmp)
        # end

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
    userefine::Bool = false,
    ) = error("to use a QRSymmCache for linear system solves, the data must be preprocessed and Q2 and RiQ1 must be passed into the QRSymmCache constructor")

# solve two symmetric systems and combine the solutions for x, y, z, s, kap, tau
# TODO update math description
# TODO use in-place mul-add when available in Julia, see https://github.com/JuliaLang/julia/issues/23919
function solvelinsys6!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    rhs_kap::Float64,
    rhs_ts::Vector{Float64},
    rhs_tau::Float64,
    mu::Float64,
    tau::Float64,
    L::QRSymmCache,
    )
    invmu = inv(mu)

    # calculate or factorize LHS of symmetric positive definite linear system
    # TODO multiply H by G*Q2 (lower dimension than G)
    for k in eachindex(L.cone.prmtvs)
        if L.cone.prmtvs[k].usedual
            calcHiarr_prmtv!(view(L.HG, L.cone.idxs[k], :), view(L.G, L.cone.idxs[k], :), L.cone.prmtvs[k])
            @. @views L.HG[L.cone.idxs[k], :] *= invmu
        else
            calcHarr_prmtv!(view(L.HG, L.cone.idxs[k], :), view(L.G, L.cone.idxs[k], :), L.cone.prmtvs[k])
            @. @views L.HG[L.cone.idxs[k], :] *= mu
        end
    end
    mul!(L.GHG, L.G', L.HG)
    mul!(L.GHGQ2, L.GHG, L.Q2)
    mul!(L.Q2GHGQ2, L.Q2', L.GHGQ2)

    (zi, z1, z2, yi, xi) = (L.zi, L.z1, L.z2, L.yi, L.xi)
    @. yi[:,1] = L.b
    @. yi[:,2] = -rhs_ty
    @. xi[:,1] = -L.c
    @. xi[:,2] = rhs_tx

    # calculate z2
    @. z2 = -rhs_tz
    for k in eachindex(L.cone.prmtvs)
        if L.cone.prmtvs[k].usedual
            @. @views z1[L.cone.idxs[k]] = z2[L.cone.idxs[k]] - rhs_ts[L.cone.idxs[k]]
            calcHiarr_prmtv!(view(z2, L.cone.idxs[k]), view(z1, L.cone.idxs[k]), L.cone.prmtvs[k])
            @. @views z2[L.cone.idxs[k]] *= invmu
        elseif !iszero(rhs_ts[L.cone.idxs[k]]) # TODO rhs_ts = 0 for correction steps, so can just check if doing correction
            calcHarr_prmtv!(view(z1, L.cone.idxs[k]), view(rhs_ts, L.cone.idxs[k]), L.cone.prmtvs[k])
            @. @views z2[L.cone.idxs[k]] -= mu*z1[L.cone.idxs[k]]
            # @. @views z2[L.cone.idxs[k]] -= z1[L.cone.idxs[k]]
        end
    end

    # calculate z1 TODO don't need this if h is zero (can check once when creating cache)
    for k in eachindex(L.cone.prmtvs)
        if L.cone.prmtvs[k].usedual
            calcHiarr_prmtv!(view(z1, L.cone.idxs[k]), view(L.h, L.cone.idxs[k]), L.cone.prmtvs[k])
            @. @views z1[L.cone.idxs[k]] *= invmu
        else
            calcHarr_prmtv!(view(z1, L.cone.idxs[k]), view(L.h, L.cone.idxs[k]), L.cone.prmtvs[k])
            @. @views z1[L.cone.idxs[k]] *= mu
        end
    end

    # bxGHbz = bx + G'*Hbz
    mul!(L.bxGHbz, L.G', zi)
    @. L.bxGHbz += xi
    # Q1x = Q1*Ri'*by
    mul!(L.Q1x, L.RiQ1', yi)
    # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
    mul!(L.rhs, L.GHG, L.Q1x)
    @. L.rhs = L.bxGHbz - L.rhs
    mul!(L.Q2divcopy, L.Q2', L.rhs)

    # posdef = posv!('U', L.Q2GHGQ2, L.Q2div) # for no iterative refinement or equilibration
    if size(L.Q2div, 1) > 0
        posdef = hypatia_posvx!(L.Q2div, L.Q2GHGQ2, L.Q2divcopy, L.lsferr, L.lsberr, L.lswork, L.lsiwork, L.lsAF, L.lsS)
        if !posdef
            println("linear system matrix was not positive definite")
            mul!(L.Q2GHGQ2, L.Q2', L.GHGQ2)
            L.Q2GHGQ2 += 1e-4I
            F = bunchkaufman!(Symmetric(L.Q2GHGQ2), true, check=false)
            if !issuccess(F)
                error("could not fix failure of positive definiteness; terminating")
            end
            mul!(L.Q2div, L.Q2', L.rhs)
            ldiv!(F, L.Q2div)
        end
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

    # combine
    @views dir_tau = (rhs_tau + rhs_kap + dot(L.c, xi[:,2]) + dot(L.b, yi[:,2]) + dot(L.h, z2))/(mu/tau/tau - dot(L.c, xi[:,1]) - dot(L.b, yi[:,1]) - dot(L.h, z1))
    @. @views rhs_tx = xi[:,2] + dir_tau*xi[:,1]
    @. @views rhs_ty = yi[:,2] + dir_tau*yi[:,1]
    @. rhs_tz = z2 + dir_tau*z1
    mul!(z1, L.G, rhs_tx)
    @. rhs_ts = -z1 + L.h*dir_tau - rhs_ts
    dir_kap = -dot(L.c, rhs_tx) - dot(L.b, rhs_ty) - dot(L.h, rhs_tz) - rhs_tau

    return (dir_kap, dir_tau)
end


using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc

# call LAPACK dposvx function (compare to dposv and dposvxx)
# performs equilibration and iterative refinement
# TODO not currently available in LinearAlgebra.LAPACK but should contribute
function hypatia_posvx!(
    X::Matrix{Float64},
    A::Matrix{Float64},
    B::Matrix{Float64},
    ferr,
    berr,
    work,
    iwork,
    AF,
    S,
    )
    n = size(A, 1)
    @assert n == size(A, 2) == size(B, 1)

    lda = stride(A, 2)
    nrhs = size(B, 2)
    ldb = stride(B, 2)
    rcond = Ref{Float64}()

    fact = 'E'
    uplo = 'U'
    equed = 'Y'

    info = Ref{BlasInt}()

    ccall((@blasfunc(dposvx_), Base.liblapack_name), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
        Ref{UInt8}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
        Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}),
        fact, uplo, n, nrhs, A, lda, AF, lda, equed, S, B,
        ldb, X, n, rcond, ferr, berr, work, iwork, info)

    if info[] != 0 && info[] != n+1
        # @warn("failure to solve linear system (posvx status $(info[]))")
        return false
    end
    return true
end

# call LAPACK dsysvx function
# performs equilibration and iterative refinement
# TODO not currently available in LinearAlgebra.LAPACK but should contribute
function hypatia_sysvx!(
    X::Matrix{Float64},
    A::Matrix{Float64},
    B::Matrix{Float64},
    ferr,
    berr,
    work,
    iwork,
    AF,
    S,
    )
    n = size(A, 1)
    @assert n == size(A, 2) == size(B, 1)

    lda = stride(A, 2)
    nrhs = size(B, 2)
    ldb = stride(B, 2)
    rcond = Ref{Float64}()

    fact = 'E'
    uplo = 'U'
    equed = 'Y'

    info = Ref{BlasInt}()

    ccall((@blasfunc(dsysvx_), Base.liblapack_name), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
        Ref{UInt8}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
        Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}),
        fact, uplo, n, nrhs, A, lda, AF, lda, equed, S, B,
        ldb, X, n, rcond, ferr, berr, work, iwork, info)

    if info[] != 0 && info[] != n+1
        # @warn("failure to solve linear system (posvx status $(info[]))")
        return false
    end
    return true
end
