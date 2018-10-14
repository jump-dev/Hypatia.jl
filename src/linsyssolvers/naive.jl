#=
naive method that simply performs one high-dimensional linear system solve
TODO currently only does dense operations, needs to work for sparse
=#
mutable struct NaiveCache <: LinSysCache
    cone
    c
    A
    b
    G
    h
    LHS3
    LHS3copy
    rhs3
    LHS6
    LHS6copy
    rhs6
    tyk
    tzk
    tkk
    tsk
    ttk

    function NaiveCache(
        c::Vector{Float64},
        A::AbstractMatrix{Float64},
        b::Vector{Float64},
        G::AbstractMatrix{Float64},
        h::Vector{Float64},
        cone::Cone,
        )

        (n, p, q) = (length(c), length(b), length(h))
        L = new()
        L.cone = cone
        L.c = c
        L.A = A
        L.b = b
        L.G = G
        L.h = h
        L.tyk = n + 1
        L.tzk = L.tyk + p
        L.tkk = L.tzk + q
        L.tsk = L.tkk + 1
        L.ttk = L.tsk + q
        # tx ty tz
        L.LHS3 = [
            zeros(n,n)  A'          G';
            A           zeros(p,p)  zeros(p,q);
            G           zeros(q,p)  Matrix(-1.0I,q,q);
            ]
        L.LHS3copy = similar(L.LHS3)
        L.rhs3 = zeros(L.tkk-1)
        # tx ty tzp tzd kap tsp tsd tau
        L.LHS6 = [
            zeros(n,n)  A'          G'                zeros(n)  zeros(n,q)         c;
            -A          zeros(p,p)  zeros(p,q)        zeros(p)  zeros(p,q)         b;
            zeros(q,n)  zeros(q,p)  Matrix(1.0I,q,q)  zeros(q)  Matrix(1.0I,q,q)   zeros(q);
            zeros(1,n)  zeros(1,p)  zeros(1,q)        1.0       zeros(1,q)         1.0;
            -G          zeros(q,p)  zeros(q,q)        zeros(q)  Matrix(-1.0I,q,q)  h;
            -c'         -b'         -h'               -1.0      zeros(1,q)         0.0;
            ]
        L.LHS6copy = similar(L.LHS6)
        L.rhs6 = zeros(L.ttk)

        return L
    end
end

# # solve system for x, y, z
# function solvelinsys3!(
#     rhs_tx::Vector{Float64},
#     rhs_ty::Vector{Float64},
#     rhs_tz::Vector{Float64},
#     mu::Float64,
#     L::NaiveCache;
#     identityH::Bool = false,
#     )
#
#     rhs = L.rhs3
#     rhs[1:L.tyk-1] = rhs_tx
#     @. rhs[L.tyk:L.tzk-1] = -rhs_ty
#     @. rhs[L.tzk:L.tkk-1] = -rhs_tz
#
#     @. L.LHS3copy = L.LHS3
#     @assert identityH
#     # TODO update for prim or dual cones
#     # if !identityH
#     #     for k in eachindex(L.cone.prms)
#     #         idxs = L.tzk - 1 .+ L.cone.idxs[k]
#     #         dim = dimension(L.cone.prms[k])
#     #         calcHiarr_prm!(view(L.LHS3copy, idxs, idxs), Matrix(-inv(mu)*I, dim, dim), L.cone.prms[k])
#     #     end
#     # end
#
#     F = bunchkaufman!(Symmetric(L.LHS3copy))
#     ldiv!(F, rhs)
#
#     @. @views begin
#         rhs_tx = rhs[1:L.tyk-1]
#         rhs_ty = rhs[L.tyk:L.tzk-1]
#         rhs_tz = rhs[L.tzk:L.tkk-1]
#     end
#
#     return nothing
# end

# solve system for x, y, z, s, kap, tau
function solvelinsys6!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    rhs_ts::Vector{Float64},
    rhs_kap::Float64, # TODO reorder kap and tau to match 6x6 system order
    rhs_tau::Float64,
    mu::Float64,
    tau::Float64,
    L::NaiveCache,
    )

    rhs = L.rhs6
    rhs[1:L.tyk-1] = rhs_tx
    rhs[L.tyk:L.tzk-1] = rhs_ty
    rhs[L.tzk:L.tkk-1] = rhs_tz
    rhs[L.tkk] = rhs_kap
    rhs[L.tsk:L.ttk-1] = rhs_ts
    rhs[end] = rhs_tau

    # TODO don't use Matrix(mu*I, dim, dim) because it allocates and is slow
    @. L.LHS6copy = L.LHS6
    L.LHS6copy[L.tkk, end] = mu/tau/tau # TODO note in CVXOPT coneprog doc, there is no rescaling by tau, they to kap*dtau + tau*dkap = -rhskap
    for k in eachindex(L.cone.prms)
        dim = dimension(L.cone.prms[k])
        if L.cone.useduals[k]
            calcHarr_prm!(view(L.LHS6copy, L.tzk - 1 .+ L.cone.idxs[k], L.tzk - 1 .+ L.cone.idxs[k]), Matrix(mu*I, dim, dim), L.cone.prms[k])
        else
            calcHarr_prm!(view(L.LHS6copy, L.tzk - 1 .+ L.cone.idxs[k], L.tsk - 1 .+ L.cone.idxs[k]), Matrix(mu*I, dim, dim), L.cone.prms[k])
        end
    end

    F = qr!(L.LHS6copy)
    ldiv!(F, rhs)

    @. @views begin
        rhs_tx = rhs[1:L.tyk-1]
        rhs_ty = rhs[L.tyk:L.tzk-1]
        rhs_tz = rhs[L.tzk:L.tkk-1]
        rhs_ts = rhs[L.tsk:L.ttk-1]
    end
    dir_kap = rhs[L.tkk]
    dir_tau = rhs[end]

    return (dir_kap, dir_tau)
end
