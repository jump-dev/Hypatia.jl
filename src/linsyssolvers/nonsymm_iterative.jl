#=
solve a nonsymmetric linear system
use a method from IterativeSolvers.jl
=#

# TODO incomplete

mutable struct NonSymmIterCache <: LinSysCache

    function NonSymmIterCache(
        A::AbstractMatrix{Float64},
        G::AbstractMatrix{Float64},
        n::Int,
        p::Int,
        q::Int,
        )

        L = new()



        return L
    end
end

function NonSymmIterCache(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    )

    (n, p, q) = (length(c), length(b), length(h))


    return NonSymmIterCache(A, G, n, p, q)
end

#
function solvesinglelinsys!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    L::NonSymmIterCache,
    )

    # solve one symmetric system
    # |0  A' G'  | * |ux| = | bx|
    # |A  0  0   |   |uy|   |-by|
    # |G  0 -H^-1|   |uz|   |-bz|

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
    L::NonSymmIterCache,
    )

    # solve two symmetric systems and combine the solutions

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
    L::NonSymmIterCache,
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
