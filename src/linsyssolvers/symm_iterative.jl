#=
solve two symmetric linear systems and combine solutions
use a method from IterativeSolvers.jl to solve each symmetric system
=#

# TODO incomplete

mutable struct SymmIterCache <: LinSysCache

    function SymmIterCache(
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

function SymmIterCache(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    )

    (n, p, q) = (length(c), length(b), length(h))


    return SymmIterCache(A, G, n, p, q)
end

#
function solvesinglelinsys!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    L::SymmIterCache,
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
    L::SymmIterCache,
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
    L::SymmIterCache,
    )

    # |0  A' G'  | * |ux| = | bx|
    # |A  0  0   |   |uy|   |-by|
    # |G  0 -H^-1|   |uz|   |-bz|

    # solve second equation for ux
    # A*ux = -by
    # => ux = A\-by
    # TODO this won't change if by is constant?

    # solve third equation for uz
    # G*ux - H^-1*uz = -bz
    # => H^-1*uz = bz + G*ux
    # => uz = H*(bz + G*ux)

    # solve first equation for uy
    # A'*uy + G'*uz = bx
    # => A'*uy = bx - G'*uz
    # => uy = A'\(bx - G'*uz)

    return nothing
end
