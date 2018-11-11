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
        userefine::Bool = false,
        )
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
        # L.GQ2 = G*Q2
        L.HG = Matrix{Float64}(undef, q, n) # TODO don't enforce dense on some
        L.GHG = Matrix{Float64}(undef, n, n)
        L.GHGQ2 = Matrix{Float64}(undef, n, nmp)
        L.Q2GHGQ2 = Matrix{Float64}(undef, nmp, nmp)
        L.bxGHbz = Matrix{Float64}(undef, n, 2)
        L.Q1x = Matrix{Float64}(undef, n, 2)
        L.rhs = Matrix{Float64}(undef, n, 2)
        L.Q2div = Matrix{Float64}(undef, nmp, 2)
        L.Q2x = Matrix{Float64}(undef, n, 2)
        L.GHGxi = Matrix{Float64}(undef, n, 2)
        L.HGxi = Matrix{Float64}(undef, q, 2)
        # L.bxGHbz = Vector{Float64}(undef, n)
        # L.Q1x = Vector{Float64}(undef, n)
        # L.rhs = Vector{Float64}(undef, n)
        # L.Q2div = Vector{Float64}(undef, nmp)
        # L.Q2x = Vector{Float64}(undef, n)
        # L.GHGxi = Vector{Float64}(undef, n)
        # L.HGxi = Vector{Float64}(undef, q)
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
    userefine::Bool = false,
    ) = error("to use a QRSymmCache for linear system solves, the data must be preprocessed and Q2 and RiQ1 must be passed into the QRSymmCache constructor")

# # solve system for x, y, z
# function solvelinsys3!(
#     rhs_tx::Vector{Float64},
#     rhs_ty::Vector{Float64},
#     rhs_tz::Vector{Float64},
#     mu::Float64,
#     L::QRSymmCache;
#     identityH::Bool = false,
#     )
#     F = helplhs!(mu, L, identityH=identityH)
#     @. rhs_ty *= -1.0
#     @. rhs_tz *= -1.0
#     helplinsys!(rhs_tx, rhs_ty, rhs_tz, F, L)
#
#     return nothing
# end

# solve two symmetric systems and combine the solutions for x, y, z, s, kap, tau
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
    @assert !L.useiterative
    # Q2' * G' * H * G * Q2 # TODO update math description
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




    # calculate z2 TODO update math description
    z2 = -rhs_tz
    for k in eachindex(L.cone.prmtvs)
        if L.cone.prmtvs[k].usedual
            @. @views  L.z1[L.cone.idxs[k]] = z2[L.cone.idxs[k]] - rhs_ts[L.cone.idxs[k]]
            calcHiarr_prmtv!(view(z2, L.cone.idxs[k]), view(L.z1, L.cone.idxs[k]), L.cone.prmtvs[k])
            @. @views z2[L.cone.idxs[k]] *= invmu
        elseif !iszero(rhs_ts[L.cone.idxs[k]]) # TODO rhs_ts = 0 for correction steps, so can just check if doing correction
            calcHarr_prmtv!(view(L.z1, L.cone.idxs[k]), view(rhs_ts, L.cone.idxs[k]), L.cone.prmtvs[k])
            @. @views z2[L.cone.idxs[k]] -= mu*L.z1[L.cone.idxs[k]]
        end
    end

    # calculate z1 TODO don't need this if h is zero (can check once when creating cache)
    for k in eachindex(L.cone.prmtvs)
        if L.cone.prmtvs[k].usedual
            calcHiarr_prmtv!(view(L.z1, L.cone.idxs[k]), view(L.h, L.cone.idxs[k]), L.cone.prmtvs[k])
            @. @views L.z1[L.cone.idxs[k]] *= invmu
        else
            calcHarr_prmtv!(view(L.z1, L.cone.idxs[k]), view(L.h, L.cone.idxs[k]), L.cone.prmtvs[k])
            @. @views L.z1[L.cone.idxs[k]] *= mu
        end
    end

    # xyz12 = [-L.c rhs_tx; L.b -rhs_ty; L.z1 z2]
    xi = [-L.c rhs_tx]
    yi = [L.b -rhs_ty]
    zi = [L.z1 z2]



    # bxGHbz = bx + G'*Hbz
    mul!(L.bxGHbz, L.G', zi)
    @. L.bxGHbz += xi
    # Q1x = Q1*Ri'*by
    mul!(L.Q1x, L.RiQ1', yi)
    # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
    mul!(L.rhs, L.GHG, L.Q1x)
    @. L.rhs = L.bxGHbz - L.rhs
    mul!(L.Q2div, L.Q2', L.rhs)


    # TODO use posvx or posvxx (has iterative refinement, equilibration)
    posdef = posv!('U', L.Q2GHGQ2, L.Q2div)
    if !posdef
        @warn("linear system matrix was not positive definite")
        # TODO improve recovery method for making LHS positive definite
        mul!(L.Q2GHGQ2, L.Q2', L.GHGQ2)
        L.Q2GHGQ2 += 1e-4I
        posdef = posv!('U', L.Q2GHGQ2, L.Q2div)
        if !posdef
            error("could not fix failure of positive definiteness; terminating")
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
    @views dir_tau = (rhs_tau + rhs_kap + dot(L.c, xi[:,2]) + dot(L.b, yi[:,2]) + dot(L.h, zi[:,2]))/(mu/tau/tau - dot(L.c, xi[:,1]) - dot(L.b, yi[:,1]) - dot(L.h, zi[:,1]))
    @. @views rhs_tx = xi[:,2] + dir_tau*xi[:,1]
    @. @views rhs_ty = yi[:,2] + dir_tau*yi[:,1]
    @. @views rhs_tz = zi[:,2] + dir_tau*zi[:,1]
    mul!(L.z1, L.G, rhs_tx)
    @. rhs_ts = -L.z1 + L.h*dir_tau - rhs_ts
    dir_kap = -dot(L.c, rhs_tx) - dot(L.b, rhs_ty) - dot(L.h, rhs_tz) - rhs_tau

    return (dir_kap, dir_tau)
end


using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc

function posv!(uplo::AbstractChar, A::Matrix{Float64}, B::Matrix{Float64})
    n = size(A, 1)
    @assert n == size(B, 1)

    info = Ref{BlasInt}()
    ccall(
        (@blasfunc(dposv_), Base.liblapack_name), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}),
        uplo, n, size(B, 2), A, max(1, stride(A, 2)), B, max(1, stride(B, 2)), info
        )

    if info[] < 0
        throw(ArgumentError("invalid argument #$(-info[]) to LAPACK call"))
    end
    return (info[] == 0)
end

# performs equilibration and iterative refinement (posvxx goes even further?)
# not currently available in LinearAlgebra.LAPACK but should contribute
function posvx!(fact::AbstractChar, uplo::AbstractChar, A::Matrix{Float64}, AF::Matrix{Float64}, equed::AbstractChar, S::Vector{Float64}, B::Matrix{Float64}, X::Matrix{Float64}, work, iwork)
    n = size(A, 1)
    @assert n == size(A, 2) == size(B, 1) == size(AF, 1) == size(AF, 2)

    info = Ref{BlasInt}()
    ccall(
        (@blasfunc(dposvx_), Base.liblapack_name), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
        Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}),

        fact, uplo, n, size(B, 2), A, max(1, stride(A, 2)), AF, max(1, stride(AF, 2)), equed, 
        S, B, max(1, stride(B, 2)), X, max(1, stride(X, 2)), rcond, info
        )

    if info[] < 0
        throw(ArgumentError("invalid argument #$(-info[]) to LAPACK call"))
    end
    return (info[] == 0)
end

# see examples at https://github.com/jagot/PowerLAPACK.jl/blob/master/src/lapack.jl https://github.com/JuliaLang/julia/blob/c50aaeacc5c702f3a772c57ddacda0a80c2aafeb/stdlib/LinearAlgebra/src/lapack.jl#L4-L6 https://github.com/andreasnoack/GenericLinearAlgebra.jl/blob/master/src/lapack.jl


posvx!('E', 'U', A, AF, 'Y', B, S)



#
# *       SUBROUTINE DPOSVX( FACT, UPLO, N, NRHS, A, LDA, AF, LDAF, EQUED,
# *                          S, B, LDB, X, LDX, RCOND, FERR, BERR, WORK,
# *                          IWORK, INFO )
# *
# *       .. Scalar Arguments ..
# *       CHARACTER          EQUED, FACT, UPLO
# *       INTEGER            INFO, LDA, LDAF, LDB, LDX, N, NRHS
# *       DOUBLE PRECISION   RCOND
# *       ..
# *       .. Array Arguments ..
# *       INTEGER            IWORK( * )
# *       DOUBLE PRECISION   A( LDA, * ), AF( LDAF, * ), B( LDB, * ),
# *      $                   BERR( * ), FERR( * ), S( * ), WORK( * ),
# *      $                   X( LDX, * )




#
# # solve two symmetric systems and combine the solutions for x, y, z, s, kap, tau
# function solvelinsys6!(
#     rhs_tx::Vector{Float64},
#     rhs_ty::Vector{Float64},
#     rhs_tz::Vector{Float64},
#     rhs_kap::Float64,
#     rhs_ts::Vector{Float64},
#     rhs_tau::Float64,
#     mu::Float64,
#     tau::Float64,
#     L::QRSymmCache,
#     )
#     # solve two symmetric systems and combine the solutions
#     invmu = inv(mu)
#     F = helplhs!(mu, invmu, L)
#
#     # (x2, y2, z2) = (rhs_tx, -rhs_ty, -H*rhs_ts - rhs_tz) # TODO update math description
#     @. rhs_ty *= -1.0
#     @. rhs_tz *= -1.0
#     for k in eachindex(L.cone.prmtvs)
#         if L.cone.prmtvs[k].usedual
#             @. @views  L.z1[L.cone.idxs[k]] = rhs_tz[L.cone.idxs[k]] - rhs_ts[L.cone.idxs[k]]
#             calcHiarr_prmtv!(view(rhs_tz, L.cone.idxs[k]), view(L.z1, L.cone.idxs[k]), L.cone.prmtvs[k])
#             @. @views rhs_tz[L.cone.idxs[k]] *= invmu
#         elseif !iszero(rhs_ts[L.cone.idxs[k]]) # TODO rhs_ts = 0 for correction steps, so can just check if doing correction
#             calcHarr_prmtv!(view(L.z1, L.cone.idxs[k]), view(rhs_ts, L.cone.idxs[k]), L.cone.prmtvs[k])
#             @. @views rhs_tz[L.cone.idxs[k]] -= mu*L.z1[L.cone.idxs[k]]
#         end
#     end
#     helplinsys!(rhs_tx, rhs_ty, rhs_tz, F, L)
#
#     # (x1, y1, z1) = (-c, b, H*h) # TODO update math description
#     @. L.x1 = -L.c
#     @. L.y1 = L.b
#     # TODO don't need this if h is zero (can check once when creating cache)
#     for k in eachindex(L.cone.prmtvs)
#         if L.cone.prmtvs[k].usedual
#             calcHiarr_prmtv!(view(L.z1, L.cone.idxs[k]), view(L.h, L.cone.idxs[k]), L.cone.prmtvs[k])
#             @. @views L.z1[L.cone.idxs[k]] *= invmu
#         else
#             calcHarr_prmtv!(view(L.z1, L.cone.idxs[k]), view(L.h, L.cone.idxs[k]), L.cone.prmtvs[k])
#             @. @views L.z1[L.cone.idxs[k]] *= mu
#         end
#     end
#     helplinsys!(L.x1, L.y1, L.z1, F, L)
#
#     # combine
#     dir_tau = (rhs_tau + rhs_kap + dot(L.c, rhs_tx) + dot(L.b, rhs_ty) + dot(L.h, rhs_tz))/(mu/tau/tau - dot(L.c, L.x1) - dot(L.b, L.y1) - dot(L.h, L.z1))
#     @. rhs_tx += dir_tau*L.x1
#     @. rhs_ty += dir_tau*L.y1
#     @. rhs_tz += dir_tau*L.z1
#     mul!(L.z1, L.G, rhs_tx)
#     @. rhs_ts = -L.z1 + L.h*dir_tau - rhs_ts
#     dir_kap = -dot(L.c, rhs_tx) - dot(L.b, rhs_ty) - dot(L.h, rhs_tz) - rhs_tau
#
#     return (dir_kap, dir_tau)
# end
#
# # calculate solution to reduced symmetric linear system
# function helplinsys!(
#     xi::Vector{Float64},
#     yi::Vector{Float64},
#     zi::Vector{Float64},
#     F,
#     L::QRSymmCache,
#     )
#     # bxGHbz = bx + G'*Hbz
#     mul!(L.bxGHbz, L.G', zi)
#     @. L.bxGHbz += xi
#     # Q1x = Q1*Ri'*by
#     mul!(L.Q1x, L.RiQ1', yi)
#     # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
#     mul!(L.rhs, L.GHG, L.Q1x)
#     @. L.rhs = L.bxGHbz - L.rhs
#     mul!(L.Q2div, L.Q2', L.rhs)
#
#     if L.useiterative
#         # TODO use previous solution from same pred/corr (requires knowing if in pred or corr step, and having two Q2sol objects)
#         # TODO allow tol to be loose in early iterations and tighter later (maybe depend on mu)
#         IterativeSolvers.cg!(L.Q2sol, F, L.Q2div, maxiter=10000, tol=1e-13, statevars=L.cgstate, Pl=L.lprecond, verbose=false)
#         @. L.Q2div = L.Q2sol
#     else
#         ldiv!(F, L.Q2div)
#     end
#
#     mul!(L.Q2x, L.Q2, L.Q2div)
#     # xi = Q1x + Q2x
#     @. xi = L.Q1x + L.Q2x
#     # yi = Ri*Q1'*(bxGHbz - GHG*xi)
#     mul!(L.GHGxi, L.GHG, xi)
#     @. L.bxGHbz -= L.GHGxi
#     mul!(yi, L.RiQ1, L.bxGHbz)
#     # zi = HG*xi - Hbz
#     mul!(L.HGxi, L.HG, xi)
#     @. zi = L.HGxi - zi
#
#     return nothing
# end
#
# # calculate or factorize LHS of symmetric positive definite linear system
# function helplhs!(
#     mu::Float64,
#     invmu::Float64,
#     L::QRSymmCache;
#     identityH::Bool = false,
#     )
#     # Q2' * G' * H * G * Q2 # TODO update math description
#     if identityH
#         @. L.HG = L.G
#     else
#         for k in eachindex(L.cone.prmtvs)
#             if L.cone.prmtvs[k].usedual
#                 calcHiarr_prmtv!(view(L.HG, L.cone.idxs[k], :), view(L.G, L.cone.idxs[k], :), L.cone.prmtvs[k])
#                 @. @views L.HG[L.cone.idxs[k], :] *= invmu
#             else
#                 calcHarr_prmtv!(view(L.HG, L.cone.idxs[k], :), view(L.G, L.cone.idxs[k], :), L.cone.prmtvs[k])
#                 @. @views L.HG[L.cone.idxs[k], :] *= mu
#             end
#         end
#     end
#     mul!(L.GHG, L.G', L.HG)
#     mul!(L.GHGQ2, L.GHG, L.Q2)
#     mul!(L.Q2GHGQ2, L.Q2', L.GHGQ2)
#
#     if L.useiterative
#         return Symmetric(L.Q2GHGQ2)
#     else
#         # TODO remove allocs from bunch-kaufman by using lower-level functions
#         F = bunchkaufman!(Symmetric(L.Q2GHGQ2), true, check=false)
#         if !issuccess(F)
#             @warn("linear system matrix was not positive definite")
#             # TODO improve recovery method for making LHS positive definite
#             mul!(L.Q2GHGQ2, L.Q2', L.GHGQ2)
#             L.Q2GHGQ2 += 1e-4I
#             F = bunchkaufman!(Symmetric(L.Q2GHGQ2), true, check=false)
#             if !issuccess(F)
#                 error("could not fix failure of positive definiteness; terminating")
#             end
#         end
#         return F
#     end
# end
