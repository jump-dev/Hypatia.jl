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

mutable struct RawLinearModel <: LinearModel
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

    initial_point::Point
    final_point::Point

    function RawLinearModel(c::Vector{Float64}, A::AbstractMatrix{Float64}, b::Vector{Float64}, G::AbstractMatrix{Float64}, h::Vector{Float64}, cones::Vector{<:Cones.Cone}, cone_idxs::Vector{UnitRange{Int}})
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

        # get initial point
        point = Point(similar(c), similar(b), similar(h), similar(h), cones, cone_idxs)
        set_initial_cone_point(point, model.cones)
        if !isempty(point.y)
            # solve for y as least squares solution to A'y = -c - G'z
            # TODO reduce allocs
            y_rhs = -c - G' * point.z
            point.y .= (issparse(A) ? sparse(A') : A') \ y_rhs
        end
        if !isempty(point.x)
            # solve for x as least squares solution to Ax = b, Gx = h - s
            # TODO reduce allocs
            x_rhs = vcat(b, h - point.s)
            point.x .= vcat(A, G) \ x_rhs
        end
        model.initial_point = point

        return model
    end
end

# TODO check model data consistency function

mutable struct PreprocessedLinearModel <: LinearModel
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

    n_raw::Int
    p_raw::Int
    x_keep_idxs::AbstractVector{Int}
    y_keep_idxs::AbstractVector{Int}
    Ap_R
    Ap_Q1
    Ap_Q2

    initial_point::Point
    final_point::Point

    function PreprocessedLinearModel(c::Vector{Float64}, A::AbstractMatrix{Float64}, b::Vector{Float64}, G::AbstractMatrix{Float64}, h::Vector{Float64}, cones::Vector{<:Cones.Cone}, cone_idxs::Vector{UnitRange{Int}}; tol_QR::Float64 = 1e-13)
        model = new()
        n = length(c)
        p = length(b)
        q = length(h)
        model.n_raw = n
        model.p_raw = p
        model.q = q
        model.h = h
        model.cones = cones
        model.cone_idxs = cone_idxs
        model.nu = isempty(cones) ? 0.0 : sum(Cones.get_nu, cones)

        # TODO could optionally rescale rows of [A, b] and [G, h] and [A', G', c] and variables

        # get initial point and preprocess
        point = Point(similar(c), similar(b), similar(h), similar(h), cones, cone_idxs)
        set_initial_cone_point(point, model.cones)

        # NOTE (pivoted) QR factorizations are usually rank-revealing but may be unreliable, see http://www.math.sjsu.edu/~foster/rankrevealingcode.html

        # solve for x as least squares solution to Ax = b, Gx = h - s
        if !isempty(point.x)
            # get pivoted QR # TODO when Julia has a unified QR interface, replace this
            AG = vcat(A, G)
            if issparse(AG)
                AG_fact = qr(AG, tol = tol_QR)
            else
                AG_fact = qr(AG, Val(true))
            end
            AG_R = AG_fact.R

            # TODO could replace this with rank(Ap_fact) when available for both dense and sparse
            AG_rank = 0
            for i in 1:size(AG_R, 1) # TODO could replace this with rank(AG_fact) when available for both dense and sparse
                if abs(AG_R[i, i]) > tol_QR
                    AG_rank += 1
                end
            end

            point.x = AG_fact \ vcat(b, h - point.s)

            if AG_rank == n
                # no dual equalities to remove
                x_keep_idxs = 1:n
            else
                # TODO optimize all below
                if issparse(AG)
                    x_keep_idxs = AG_fact.pcol[1:AG_rank]
                    AG_Q1 = Matrix{Float64}(undef, p + q, AG_rank)
                    # AG_Q1[AG_fact.prow, :] = AG_fact.Q * Matrix{Float64}(I, p + q, AG_rank) # TODO could eliminate this allocation
                    AG_Q1[AG_fact.prow, :] = Matrix(AG_fact.Q)
                else
                    x_keep_idxs = AG_fact.p[1:AG_rank]
                    # AG_Q1 = AG_fact.Q * Matrix{Float64}(I, p + q, AG_rank) # TODO could eliminate this allocation
                    AG_Q1 = Matrix(AG_fact.Q)
                end
                AG_R = UpperTriangular(AG_R[1:AG_rank, 1:AG_rank])

                c_sub = c[x_keep_idxs]
                yz_sub = AG_Q1 * (AG_R' \ c_sub)
                if norm(AG' * yz_sub - c, Inf) > tol_QR
                    error("some dual equality constraints are inconsistent")
                end
                println("removed $(n - AG_rank) out of $n dual equality constraints")
                c = c_sub
                A = A[:, x_keep_idxs]
                G = G[:, x_keep_idxs]
                n = AG_rank
                point.x = point.x[x_keep_idxs]
            end
        else
            x_keep_idxs = Int[]
        end

        # solve for y as least squares solution to A'y = -c - G'z
        if !isempty(point.y)
            # get pivoted QR # TODO when Julia has a unified QR interface, replace this
            if issparse(A)
                Ap_fact = qr(sparse(A'), tol = tol_QR)
            else
                Ap_fact = qr(A', Val(true))
            end
            Ap_R = Ap_fact.R

            # TODO could replace this with rank(Ap_fact) when available for both dense and sparse
            Ap_rank = 0
            for i in 1:size(Ap_R, 1)
                if abs(Ap_R[i, i]) > tol_QR
                    Ap_rank += 1
                end
            end

            point.y = Ap_fact \ (-c - G' * point.z) # TODO remove allocs

            # TODO optimize all below
            if issparse(A)
                y_keep_idxs = Ap_fact.pcol[1:Ap_rank]
                A_Q = Matrix{Float64}(undef, n, n)
                A_Q[Ap_fact.prow, :] = Ap_fact.Q * Matrix{Float64}(I, n, n) # TODO could eliminate this allocation
            else
                y_keep_idxs = Ap_fact.p[1:Ap_rank]
                A_Q = Ap_fact.Q * Matrix{Float64}(I, n, n) # TODO could eliminate this allocation
            end
            Ap_Q1 = A_Q[:, 1:Ap_rank]
            Ap_Q2 = A_Q[:, (Ap_rank + 1):n]
            Ap_R = UpperTriangular(Ap_R[1:Ap_rank, 1:Ap_rank]) # TODO could eliminate this allocation

            b_sub = b[y_keep_idxs]
            if Ap_rank < p
                # some dependent primal equalities, so check if they are consistent
                x_sub = Ap_Q1 * (Ap_R' \ b_sub)
                if norm(A * x_sub - b, Inf) > tol_QR
                    error("some primal equality constraints are inconsistent")
                end
                println("removed $(p - Ap_rank) out of $p primal equality constraints")
            end
            A = A[y_keep_idxs, :]
            b = b_sub
            p = Ap_rank
            point.y = point.y[y_keep_idxs]

            model.Ap_R = Ap_R
            model.Ap_Q1 = Ap_Q1
            model.Ap_Q2 = Ap_Q2
        else
            y_keep_idxs = Int[]
        end

        model.n = n
        model.p = p
        model.c = c
        model.A = A
        model.b = b
        model.G = G
        model.x_keep_idxs = x_keep_idxs
        model.y_keep_idxs = y_keep_idxs

        model.initial_point = point

        return model
    end
end

function set_initial_cone_point(point, cones)
    for k in eachindex(cones)
        cone_k = cones[k]
        primal_k = point.primal_views[k]
        Cones.set_initial_point(primal_k, cone_k)
        Cones.load_point(cone_k, primal_k)
        @assert Cones.check_in_cone(cone_k)
        point.dual_views[k] .= -Cones.grad(cone_k)
    end
    return
end

# function x_unprocess(x_processed::Vector{Float64}, model::PreprocessedLinearModel)
#     x = zeros(model.n_raw)
#     x[model.x_keep_idxs] = x_processed
#     return x
# end
#
# function y_unprocess(y_processed::Vector{Float64}, model::PreprocessedLinearModel)
#     y = zeros(model.p_raw)
#     y[model.y_keep_idxs] = y_processed
#     return y
# end
#
# x_unprocess(x::Vector{Float64}, model::RawLinearModel) = x
# y_unprocess(y::Vector{Float64}, model::RawLinearModel) = y
