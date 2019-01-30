#=
Copyright 2018, Chris Coey and contributors

functions and types for linear objective conic problems of the form:

primal (over x,s):
```
  min  c'x :          duals
    b - Ax == 0       (y)
    h - Gx == s in K  (z)
```
dual (over z,y):
```
  max  -b'y - h'z :      duals
    c + A'y + G'z == 0   (x)
                z in K*  (s)
```
where K is a convex cone defined as a Cartesian product of recognized primitive cones, and K* is its dual cone.

The primal-dual optimality conditions are:
```
         b - Ax == 0
         h - Gx == s
  c + A'y + G'z == 0
            s'z == 0
              s in K
              z in K*
```
=#

mutable struct LinearObjConic <: Model
    n::Int
    p::Int
    q::Int
    c::Vector{Float64}
    A::AbstractMatrix{Float64}
    b::Vector{Float64}
    G::AbstractMatrix{Float64}
    h::Vector{Float64}
    cones::Vector{Cones.Cone}
    cone_idxs::Vector{UnitRange{Int}}
    nu::Float64

    # initial_x::Vector{Float64}
    # initial_y::Vector{Float64}
    # initial_z::Vector{Float64}
    # initial_s::Vector{Float64}

    function LinearObjConic(c, A, b, G, h, cones, cone_idxs)
        model = new()
        model.n = length(c)
        model.p = length(b)
        model.q = length(h)
        model.c = c
        model.A = A
        model.b = b
        model.G = G
        model.h = h
        model.cones = cones
        model.cone_idxs = cone_idxs
        model.nu = isempty(cones) ? 0.0 : sum(Cones.get_nu, cones)
        return model
    end
end

# TODO check model data consistency function

# TODO preprocess function - maybe wraps a model

#
# function preprocess_data(
#     c::Vector{Float64},
#     A::AbstractMatrix{Float64},
#     b::Vector{Float64},
#     G::AbstractMatrix{Float64};
#     tol::Float64 = 1e-13, # presolve tolerance
#     useQR::Bool = false, # returns QR fact of A' for use in a QR-based linear system solver
#     )
#     (n, p) = (length(c), length(b))
#     q = size(G, 1)
#
#     # NOTE (pivoted) QR factorizations are usually rank-revealing but may be unreliable, see http://www.math.sjsu.edu/~foster/rankrevealingcode.html
#     # rank of a matrix is number of nonzero diagonal elements of R
#
#     # preprocess dual equality constraints
#     dukeep = 1:n
#     AG = vcat(A, G)
#
#     # get pivoted QR # TODO when Julia has a unified QR interface, replace this
#     if issparse(AG)
#         AGF = qr(AG, tol=tol)
#     else
#         AGF = qr(AG, Val(true))
#     end
#     AGR = AGF.R
#     AGrank = 0
#     for i in 1:size(AGR, 1) # TODO could replace this with rank(AF) when available for both dense and sparse
#         if abs(AGR[i, i]) > tol
#             AGrank += 1
#         end
#     end
#
#     if AGrank < n
#         if issparse(AG)
#             dukeep = AGF.pcol[1:AGrank]
#             AGQ1 = Matrix{Float64}(undef, p + q, AGrank)
#             AGQ1[AGF.prow, :] = AGF.Q * Matrix{Float64}(I, p + q, AGrank) # TODO could eliminate this allocation
#         else
#             dukeep = AGF.p[1:AGrank]
#             AGQ1 = AGF.Q * Matrix{Float64}(I, p + q, AGrank) # TODO could eliminate this allocation
#         end
#         AGRiQ1 = UpperTriangular(AGR[1:AGrank, 1:AGrank]) \ AGQ1'
#
#         A1 = A[:, dukeep]
#         G1 = G[:, dukeep]
#         c1 = c[dukeep]
#
#         if norm(AG' * AGRiQ1' * c1 - c, Inf) > tol
#             error("some dual equality constraints are inconsistent")
#         end
#
#         A = A1
#         G = G1
#         c = c1
#         println("removed $(n - AGrank) out of $n dual equality constraints")
#         n = AGrank
#     end
#
#     if p == 0
#         # no primal equality constraints to preprocess
#         # TODO use I instead of dense for Q2
#         return (c, A, b, G, 1:0, dukeep, Matrix{Float64}(I, n, n), Matrix{Float64}(I, 0, n))
#     end
#
#     # preprocess primal equality constraints
#     # get pivoted QR # TODO when Julia has a unified QR interface, replace this
#     if issparse(A)
#         AF = qr(sparse(A'), tol=tol)
#     else
#         AF = qr(A', Val(true))
#     end
#     AR = AF.R
#     Arank = 0
#     for i in 1:size(AR, 1) # TODO could replace this with rank(AF) when available for both dense and sparse
#         if abs(AR[i, i]) > tol
#             Arank += 1
#         end
#     end
#
#     if !useQR && Arank == p
#         # no primal equalities to remove and QR of A' not needed
#         return (c, A, b, G, 1:p, dukeep, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0))
#     end
#
#     # using QR of A' (requires reordering rows) and/or some primal equalities are dependent
#     if issparse(A)
#         prkeep = AF.pcol[1:Arank]
#         AQ = Matrix{Float64}(undef, n, n)
#         AQ[AF.prow, :] = AF.Q * Matrix{Float64}(I, n, n) # TODO could eliminate this allocation
#     else
#         prkeep = AF.p[1:Arank]
#         AQ = AF.Q * Matrix{Float64}(I, n, n) # TODO could eliminate this allocation
#     end
#     AQ2 = AQ[:, Arank+1:n]
#     ARiQ1 = UpperTriangular(AR[1:Arank, 1:Arank]) \ AQ[:, 1:Arank]'
#
#     A1 = A[prkeep, :]
#     b1 = b[prkeep]
#
#     if Arank < p
#         # some dependent primal equalities, so check if they are consistent
#         x1 = ARiQ1' * b1
#         if norm(A * x1 - b, Inf) > tol
#             error("some primal equality constraints are inconsistent")
#         end
#         println("removed $(p - Arank) out of $p primal equality constraints")
#     end
#
#     A = A1
#     b = b1
#
#     return (c, A, b, G, prkeep, dukeep, AQ2, ARiQ1)
# end




# eliminate s
s_sol = copy(z_rhs)

z_rhs3 = copy(z_rhs)
for k in eachindex(cone.cones)
    sview = view(srhs, cone.idxs[k])
    zview = view(z_rhs3, cone.idxs[k])
    if cone.cones[k].use_dual # G*x - mu*H*z = z_rhs - srhs
        zview .-= sview
    else # G*x - (mu*H)\z = z_rhs - (mu*H)\srhs
        calcHiarr!(sview, cone.cones[k])
        @. zview -= sview / mu
    end
end

(x_sol, y_sol, z_sol) = solve_linear(3 rhs)

# TODO get s back
# G*x + s = z_rhs
s_sol .-= G * x
