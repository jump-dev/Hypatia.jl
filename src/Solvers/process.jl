#=
preprocessing and initial point finding functions for interior point algorithms
=#

MatrixyAG = Union{AbstractMatrix, UniformScaling}

# delete later, affects qr. see https://github.com/JuliaLang/julia/pull/40623
if VERSION < v"1.7.0-DEV.1188"
    const ColumnNorm = Val{true}
end

# rescale the rows and columns of the conic data to get an equivalent conic problem
function rescale_data(solver::Solver{T}) where {T <: Real}
    solver.rescale || return false
    model = solver.model
    (c, A, b, G, h) = (model.c, model.A, model.b, model.G, model.h)
    (isa(A, MatrixyAG) && isa(G, MatrixyAG)) || return false

    minval = sqrt(eps(T))
    maxabsmin(v::AbstractVecOrMat) = mapreduce(abs, max, v; init = minval)
    maxabsmincol(v::UniformScaling, ::Int) = max(abs(v.λ), minval)
    maxabsmincol(v::AbstractMatrix, j::Int) = maxabsmin(view(v, :, j))
    maxabsminrow(v::UniformScaling, ::Int) = max(abs(v.λ), minval)
    maxabsminrow(v::AbstractMatrix, i::Int) = maxabsmin(view(v, i, :))
    maxabsminrows(v::UniformScaling, ::UnitRange{Int}) = max(abs(v.λ), minval)
    maxabsminrows(v::AbstractMatrix, rows::UnitRange{Int}) =
        maxabsmin(view(v, rows, :))

    @inbounds solver.c_scale = c_scale = T[sqrt(max(abs(c[j]),
        maxabsmincol(A, j), maxabsmincol(G, j))) for j in eachindex(c)]
    @inbounds solver.b_scale = b_scale = T[sqrt(max(abs(b[i]),
        maxabsminrow(A, i))) for i in eachindex(b)]

    h_scale = solver.h_scale = ones(T, model.q)
    for (k, cone) in enumerate(model.cones)
        idxs = model.cone_idxs[k]
        if cone isa Cones.Nonnegative
            for i in idxs
                @inbounds h_scale[i] = sqrt(max(abs(h[i]), maxabsminrow(G, i)))
            end
        else
            # TODO store single scale value only?
            @inbounds @views h_scale[idxs] .= sqrt(
                max(maxabsmin(h[idxs]), maxabsminrows(G, idxs)))
        end
    end

    c_diag = Diagonal(c_scale)
    model.c = c_diag \ c
    model.A = A / c_diag
    model.G = G / c_diag
    b_diag = Diagonal(b_scale)
    model.b = b_diag \ b
    ldiv!(b_diag, model.A)
    h_diag = Diagonal(h_scale)
    model.h = h_diag \ h
    ldiv!(h_diag, model.G)

    return true
end

# optionally preprocess dual equalities and solve for x as least squares
# solution to Ax = b, Gx = h - s
function find_initial_x(
    solver::Solver{T},
    init_s::Vector{T},
    ) where {T <: Real}
    if solver.status != SolveCalled
        return zeros(T, 0)
    end
    model = solver.model
    n = model.n
    if iszero(n) # x is empty (no primal variables)
        solver.x_keep_idxs = Int[]
        return zeros(T, 0)
    end
    p = model.p
    q = model.q
    A = model.A
    G = model.G
    solver.x_keep_idxs = 1:n

    rhs = vcat(model.b, model.h - init_s)

    # indirect method
    if solver.init_use_indirect || !isa(A, MatrixyAG) || !isa(G, MatrixyAG)
        # TODO pick lsqr or lsmr
        if iszero(p)
            AG = G
        else
            linmap(M::UniformScaling) = LinearMaps.LinearMap(M, n)
            linmap(M) = LinearMaps.LinearMap(M)
            AG = vcat(linmap(A), linmap(G))
        end
        init_x = IterativeSolvers.lsqr(AG, rhs)
        return init_x
    end

    # direct method
    if iszero(p)
        # A is empty
        if issparse(G)
            AG = G
        elseif G isa Matrix{T}
            AG = copy(G)
        else
            AG = Matrix(G)
        end
    else
        AG = vcat(A, G)
    end
    if issparse(AG)
        if !(T <: Float64)
            @warn("using dense factorization of [A; G] in preprocessing and " *
                "initial point finding because sparse factorization for number " *
                "type $T is not supported by SuiteSparse packages")
            AG_fact = qr!(Matrix(AG), ColumnNorm())
        else
            AG_fact = qr(AG, tol = solver.init_tol_qr)
        end
    else
        AG_fact = qr!(AG, ColumnNorm())
    end
    AG_rank = get_rank_est(AG_fact, solver.init_tol_qr)

    if !solver.preprocess || (AG_rank == n)
        if AG_rank < n
            @warn("some dual equalities appear to be dependent " *
            "(possibly inconsistent); try using preprocess = true")
        end
        init_x = AG_fact \ rhs
        return init_x
    end

    # preprocess dual equalities
    col_piv = (AG_fact isa QRPivoted{T, Matrix{T}}) ? AG_fact.p : AG_fact.pcol
    x_keep_idxs = col_piv[1:AG_rank]
    AG_R = UpperTriangular(AG_fact.R[1:AG_rank, 1:AG_rank])

    c_sub = model.c[x_keep_idxs]
    # yz_sub = AG_fact.Q * vcat((AG_R' \ c_sub), zeros(p + q - AG_rank))
    yz_sub = zeros(T, p + q)
    yz_sub1 = view(yz_sub, 1:AG_rank)
    copyto!(yz_sub1, c_sub)
    ldiv!(AG_R', yz_sub1)
    lmul!(AG_fact.Q, yz_sub)
    if !(AG_fact isa QRPivoted{T, Matrix{T}})
        yz_sub = yz_sub[AG_fact.rpivinv]
    end
    @views residual = norm(A' * yz_sub[1:p] + G' *
        yz_sub[(p + 1):end] - model.c, Inf)
    if residual > solver.init_tol_qr
        if solver.verbose
            println("some dual equality constraints are inconsistent " *
            "(residual $residual, tolerance $(solver.init_tol_qr))")
        end
        solver.status = DualInconsistent
        return zeros(T, 0)
    end
    if solver.verbose
        println("$(n - AG_rank) of $n dual equality constraints are dependent")
    end

    # modify solver.model to remove/reorder some primal variables x
    model.c = c_sub
    model.A = A[:, x_keep_idxs]
    model.G = G[:, x_keep_idxs]
    model.n = AG_rank
    solver.x_keep_idxs = x_keep_idxs

    # init_x = AG_R \ ((AG_fact.Q' * vcat(b, h - point.s))[1:AG_rank])
    temp = vcat(model.b, model.h - init_s)
    lmul!(AG_fact.Q', temp)
    init_x = temp[1:model.n]
    ldiv!(AG_R, init_x)

    return init_x
end

# optionally preprocess primal equalities and solve for y as least squares
# solution to A'y = -c - G'z
function find_initial_y(
    solver::Solver{T},
    init_z::Vector{T},
    reduce::Bool,
    ) where {T <: Real}
    if solver.status != SolveCalled
        return zeros(T, 0)
    end
    model = solver.model
    p = model.p
    if iszero(p) # y is empty (no primal variables)
        solver.y_keep_idxs = Int[]
        solver.Ap_R = UpperTriangular(zeros(T, 0, 0))
        solver.Ap_Q = I
        return zeros(T, 0)
    end
    n = model.n
    q = model.q
    A = model.A
    solver.y_keep_idxs = 1:p

    if !reduce || !isa(A, MatrixyAG)
        # rhs = -c - G' * point.z
        rhs = copy(model.c)
        mul!(rhs, model.G', init_z, -1, -1)

        # indirect method
        if solver.init_use_indirect
            # TODO pick lsqr or lsmr
            init_y = IterativeSolvers.lsqr(A', rhs)
            return init_y
        end
    end

    # factorize A'
    if issparse(A)
        if !(T <: Float64)
            @warn("using dense factorization of A' in preprocessing and initial " *
                "point finding because sparse factorization for number type $T " *
                "is not supported by SuiteSparse packages")
            Ap_fact = qr!(Matrix(A'), ColumnNorm())
        else
            Ap_fact = qr(sparse(A'), tol = solver.init_tol_qr)
        end
    else
        Ap_fact = qr!(Matrix(A'), ColumnNorm())
    end
    Ap_rank = get_rank_est(Ap_fact, solver.init_tol_qr)

    if !reduce && !solver.preprocess
        if Ap_rank < p
            @warn("some primal equalities appear to be dependent " *
            "(possibly inconsistent); try using preprocess = true")
        end
        init_y = Ap_fact \ rhs
        return init_y
    end

    # preprocess dual equalities
    Ap_R = UpperTriangular(Ap_fact.R[1:Ap_rank, 1:Ap_rank])
    col_piv = (Ap_fact isa QRPivoted{T, Matrix{T}}) ? Ap_fact.p : Ap_fact.pcol
    y_keep_idxs = col_piv[1:Ap_rank]
    Ap_Q = Ap_fact.Q

    b_sub = model.b[y_keep_idxs]
    if Ap_rank < p
        # some dependent primal equalities, so check if they are consistent
        # x_sub = Ap_Q * vcat((Ap_R' \ b_sub), zeros(n - Ap_rank))
        x_sub = zeros(T, n)
        x_sub1 = view(x_sub, 1:Ap_rank)
        copyto!(x_sub1, b_sub)
        ldiv!(Ap_R', x_sub1)
        lmul!(Ap_Q, x_sub)

        if !(Ap_fact isa QRPivoted{T, Matrix{T}})
            x_sub = x_sub[Ap_fact.rpivinv]
        end
        residual = norm(A * x_sub - model.b, Inf)
        if residual > solver.init_tol_qr
            if solver.verbose
                println("some primal equality constraints are inconsistent " *
                "(residual $residual, tolerance $(solver.init_tol_qr))")
            end
            solver.status = PrimalInconsistent
            return zeros(T, 0)
        end
        if solver.verbose
            println("$(p - Ap_rank) of $p primal equality constraints " *
                "are dependent")
        end
    end

    if reduce && isa(model.G, MatrixyAG)
        # remove all primal equalities by making A and b empty with
        # n = n0 - p0 and p = 0
        # TODO improve efficiency
        # TODO avoid calculating GQ1 explicitly if possible
        # recover original-space solution using:
        # x0 = Q * [(R' \ b0), x]
        # y0 = R \ (-cQ1' - GQ1' * z0)
        if !(Ap_fact isa QRPivoted{T, Matrix{T}})
            row_piv = Ap_fact.prow
            model.c = model.c[row_piv]
            model.G = model.G[:, row_piv]
            solver.reduce_row_piv_inv = Ap_fact.rpivinv
        else
            solver.reduce_row_piv_inv = Int[]
        end

        Q1_idxs = 1:Ap_rank
        Q2_idxs = (Ap_rank + 1):n

        # [cQ1 cQ2] = c0' * Q
        cQ = model.c' * Ap_Q
        cQ1 = solver.reduce_cQ1 = cQ[Q1_idxs]
        cQ2 = cQ[Q2_idxs]
        # c = cQ2
        model.c = cQ2
        model.n = length(model.c)
        # offset = offset0 + cQ1 * (R' \ b0)
        Rpib0 = solver.reduce_Rpib0 = ldiv!(Ap_R', b_sub)
        # solver.Rpib0 = Rpib0 # TODO
        model.obj_offset += dot(cQ1, Rpib0)

        # [GQ1 GQ2] = G0 * Q
        # very inefficient method used for sparse G * QRSparseQ
        # see https://github.com/JuliaLang/julia/issues/31124#issuecomment-501540818
        if model.G isa UniformScaling
            side = size(Ap_Q, 1)
            G_mul = Matrix{T}(model.G, side, side)
        elseif !isa(model.G, Matrix)
            G_mul = Matrix{T}(model.G)
        else
            G_mul = model.G
        end
        GQ = G_mul * Ap_Q
        GQ1 = solver.reduce_GQ1 = GQ[:, Q1_idxs]
        GQ2 = GQ[:, Q2_idxs]
        # h = h0 - GQ1 * (R' \ b0)
        mul!(model.h, GQ1, Rpib0, -1, true)

        # G = GQ2
        model.G = GQ2

        # A and b empty
        model.p = 0
        model.A = zeros(T, 0, model.n)
        model.b = zeros(T, 0)
        solver.reduce_Ap_R = Ap_R
        solver.reduce_Ap_Q = Ap_Q

        solver.reduce_y_keep_idxs = y_keep_idxs
        solver.Ap_R = UpperTriangular(zeros(T, 0, 0))
        solver.Ap_Q = I

        return zeros(T, 0)
    end

    # init_y = Ap_R \ ((Ap_fact.Q' * (-c - G' * point.z))[1:Ap_rank])
    temp = copy(model.c)
    mul!(temp, model.G', init_z, true, true)
    lmul!(Ap_fact.Q', temp)
    init_y = temp[1:Ap_rank]
    init_y .*= -1
    ldiv!(Ap_R, init_y)

    # modify solver.model to remove/reorder some dual variables y
    if !(Ap_fact isa QRPivoted{T, Matrix{T}})
        row_piv = Ap_fact.prow
        model.A = A[y_keep_idxs, row_piv]
        model.c = model.c[row_piv]
        model.G = model.G[:, row_piv]
        solver.x_keep_idxs = solver.x_keep_idxs[row_piv]
    else
        model.A = A[y_keep_idxs, :]
    end
    model.b = b_sub
    model.p = Ap_rank
    solver.y_keep_idxs = y_keep_idxs
    solver.Ap_R = Ap_R
    solver.Ap_Q = Ap_Q

    return init_y
end

# (pivoted) QR factorizations are usually rank-revealing but may be unreliable
# see http://www.math.sjsu.edu/~foster/rankrevealingcode.html
# TODO could replace this with rank(qr_fact) when available for both dense and sparse
get_rank_est(qr_fact::SuiteSparse.SPQR.QRSparse, init_tol_qr::Real) =
    rank(qr_fact)

function get_rank_est(qr_fact::QRPivoted, init_tol_qr::Real)
    factors = qr_fact.factors
    rank_est = 0
    for i in diagind(factors)
        if abs(factors[i]) > init_tol_qr
            rank_est += 1
        end
    end
    return rank_est
end

# postprocess the interior point and save in result point
function postprocess(solver::Solver{T}) where {T <: Real}
    point = solver.point
    result = solver.result
    if in(solver.status, (PrimalInfeasible, DualInfeasible))
        tau = true
    else
        tau = point.tau[]
        if tau <= 0
            result.vec .= NaN
            return
        end
    end

    # finalize s,z
    @. result.s = point.s / tau
    @. result.z = point.z / tau

    # finalize x
    if solver.preprocess && !iszero(solver.orig_model.n) && !any(isnan, point.x)
        # unpreprocess solver's solution
        if solver.reduce && !iszero(solver.orig_model.p)
            # unreduce solver's solution
            # x = Q * [(R' \ b0), x]
            xa = zeros(T, solver.orig_model.n - length(solver.reduce_Rpib0))
            @. @views xa[solver.x_keep_idxs] = point.x / tau
            if in(solver.status, (PrimalInfeasible, DualInfeasible))
                Rpib0 = zeros(T, length(solver.reduce_Rpib0))
            else
                Rpib0 = solver.reduce_Rpib0
            end
            xb = vcat(Rpib0, xa)
            lmul!(solver.reduce_Ap_Q, xb)
            if isempty(solver.reduce_row_piv_inv)
                result.x .= xb
            else
                @. @views result.x = xb[solver.reduce_row_piv_inv]
            end
        else
            @. @views result.x[solver.x_keep_idxs] = point.x / tau
        end
    else
        @. result.x = point.x / tau
    end

    # finalize y
    if solver.preprocess && !iszero(solver.orig_model.p) && !any(isnan, point.y)
        # unpreprocess solver's solution
        if solver.reduce
            # unreduce solver's solution
            # y = R \ (-cQ1' - GQ1' * z)
            ya = solver.reduce_GQ1' * result.z
            if !in(solver.status, (PrimalInfeasible, DualInfeasible))
                ya .+= solver.reduce_cQ1
            end
            @views ldiv!(solver.reduce_Ap_R,
                ya[1:length(solver.reduce_y_keep_idxs)])
            @. @views result.y[solver.reduce_y_keep_idxs] = -ya
        else
            @. @views result.y[solver.y_keep_idxs] = point.y / tau
        end
    else
        @. result.y = point.y / tau
    end

    if solver.used_rescaling
        # unscale result
        result.s .*= solver.h_scale
        result.z ./= solver.h_scale
        result.y ./= solver.b_scale
        result.x ./= solver.c_scale
    end

    return
end
