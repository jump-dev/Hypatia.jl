#=
Copyright 2018, Chris Coey and contributors

interior point algorithms based on homogeneous self dual embedding
=#

function solve(solver::Solver{T}) where {T <: Real}
    @assert solver.status == :Loaded
    solver.status = :SolveCalled
    start_time = time()
    solver.num_iters = 0
    solver.solve_time = NaN
    solver.timer = TimerOutput()

    # preprocess and find initial point
    @timeit solver.timer "initialize" begin
        @timeit solver.timer "init_cone" point = solver.point = initialize_cone_point(solver.orig_model.cones, solver.orig_model.cone_idxs)

        @timeit solver.timer "find_point" solver.preprocess ? preprocess_find_initial_point(solver) : find_initial_point(solver)
        solver.status != :SolveCalled && return solver
        model = solver.model

        solver.tau = one(T)
        solver.kap = one(T)
        calc_mu(solver)
        if isnan(solver.mu) || abs(one(T) - solver.mu) > sqrt(eps(T))
            @warn("initial mu is $(solver.mu) but should be 1 (this could indicate a problem with cone barrier oracles)")
        end
        Cones.load_point.(model.cones, point.primal_views)
    end

    # setup iteration helpers
    solver.x_residual = similar(model.c)
    solver.y_residual = similar(model.b)
    solver.z_residual = similar(model.h)
    solver.x_norm_res_t = NaN
    solver.y_norm_res_t = NaN
    solver.z_norm_res_t = NaN
    solver.x_norm_res = NaN
    solver.y_norm_res = NaN
    solver.z_norm_res = NaN

    solver.primal_obj_t = NaN
    solver.dual_obj_t = NaN
    solver.primal_obj = NaN
    solver.dual_obj = NaN
    solver.gap = NaN
    solver.rel_gap = NaN
    solver.x_feas = NaN
    solver.y_feas = NaN
    solver.z_feas = NaN

    solver.x_conv_tol = inv(max(one(T), norm(model.c)))
    solver.y_conv_tol = inv(max(one(T), norm(model.b)))
    solver.z_conv_tol = inv(max(one(T), norm(model.h)))
    solver.prev_is_slow = false
    solver.prev2_is_slow = false
    solver.prev_gap = NaN
    solver.prev_rel_gap = NaN
    solver.prev_x_feas = NaN
    solver.prev_y_feas = NaN
    solver.prev_z_feas = NaN

    solver.prev_aff_alpha = one(T)
    solver.prev_gamma = one(T)
    solver.prev_alpha = one(T)
    solver.z_temp = similar(model.h)
    solver.s_temp = similar(model.h)
    solver.primal_views = [view(Cones.use_dual(model.cones[k]) ? solver.z_temp : solver.s_temp, model.cone_idxs[k]) for k in eachindex(model.cones)]
    solver.dual_views = [view(Cones.use_dual(model.cones[k]) ? solver.s_temp : solver.z_temp, model.cone_idxs[k]) for k in eachindex(model.cones)]
    if !solver.use_infty_nbhd
        solver.nbhd_temp = [Vector{T}(undef, length(model.cone_idxs[k])) for k in eachindex(model.cones)]
    end
    solver.cones_infeas = trues(length(model.cones))
    solver.cones_loaded = trues(length(model.cones))

    @timeit solver.timer "setup_stepper" load(solver.stepper, solver)
    @timeit solver.timer "setup_system" load(solver.system_solver, solver)

    # iterate from initial point
    solver.keep_iterating = true
    while solver.keep_iterating
        @timeit solver.timer "calc_res" calc_residual(solver)

        @timeit solver.timer "calc_conv" calc_convergence_params(solver)

        @timeit solver.timer "print_iter" solver.verbose && print_iteration_stats(solver)

        @timeit solver.timer "check_conv" check_convergence(solver) && break

        if solver.num_iters == solver.iter_limit
            solver.verbose && println("iteration limit reached; terminating")
            solver.status = :IterationLimit
            break
        end
        if time() - start_time >= solver.time_limit
            solver.verbose && println("time limit reached; terminating")
            solver.status = :TimeLimit
            break
        end

        @timeit solver.timer "step" step(solver.stepper, solver)
        solver.num_iters += 1
    end

    # calculate result and iteration statistics and finish
    point.x ./= solver.tau
    point.y ./= solver.tau
    point.z ./= solver.tau
    point.s ./= solver.tau
    Cones.load_point.(solver.model.cones, point.primal_views)

    solver.solve_time = time() - start_time

    # free memory used by some system solvers
    free_memory(solver.system_solver)

    solver.verbose && println("\nstatus is $(solver.status) after $(solver.num_iters) iterations and $(trunc(solver.solve_time, digits=3)) seconds\n")

    return solver
end

function initialize_cone_point(cones::Vector{Cones.Cone{T}}, cone_idxs::Vector{UnitRange{Int}}) where {T <: Real}
    q = isempty(cones) ? 0 : sum(Cones.dimension, cones)
    point = Models.Point(T[], T[], Vector{T}(undef, q), Vector{T}(undef, q), cones, cone_idxs)

    for k in eachindex(cones)
        cone_k = cones[k]
        Cones.setup_data(cone_k)
        primal_k = point.primal_views[k]
        Cones.set_initial_point(primal_k, cone_k)
        Cones.load_point(cone_k, primal_k)
        @assert Cones.is_feas(cone_k)
        g = Cones.grad(cone_k)
        @. point.dual_views[k] = -g
    end

    return point
end

const sparse_QR_reals = Float64

# TODO optionally use row and col scaling on AG and apply to c, b, h
# TODO precondition iterative methods
# TODO pick default tol for iter methods
function find_initial_point(solver::Solver{T}) where {T <: Real}
    model = solver.model = solver.orig_model
    A = model.A
    G = model.G
    n = model.n
    p = model.p
    q = model.q
    point = solver.point

    # solve for x as least squares solution to Ax = b, Gx = h - s
    @timeit solver.timer "init_x" if !iszero(n)
        rhs = vcat(model.b, model.h - point.s)
        if solver.init_use_indirect
            # use iterative solvers method TODO pick lsqr or lsmr
            AG = BlockMatrix{T}(p + q, n, [A, G], [1:p, (p + 1):(p + q)], [1:n, 1:n])
            @timeit solver.timer "lsqr_solve" point.x = IterativeSolvers.lsqr(AG, rhs)
        else
            AG = vcat(A, G)
            if issparse(AG) && !(T <: sparse_QR_reals)
                # TODO alternative fallback is to convert sparse{T} to sparse{Float64} and do the sparse LU
                if solver.init_use_fallback
                    @warn("using dense factorization of [A; G] in initial point finding because sparse factorization for number type $T is not supported by SuiteSparse packages")
                    AG = Matrix(AG)
                else
                    error("sparse factorization for number type $T is not supported by SuiteSparse packages, so Hypatia cannot find an initial point")
                end
            end
            @timeit solver.timer "qr_fact" AG_fact = issparse(AG) ? qr(AG) : qr!(AG)

            AG_rank = get_rank_est(AG_fact, solver.init_tol_qr)
            if AG_rank < n
                @warn("some dual equalities appear to be dependent; try using preprocess = true")
            end

            @timeit solver.timer "qr_solve" point.x = AG_fact \ rhs
        end
    end

    # solve for y as least squares solution to A'y = -c - G'z
    @timeit solver.timer "init_y" if !iszero(p)
        rhs = -model.c - G' * point.z
        if solver.init_use_indirect
            # use iterative solvers method TODO pick lsqr or lsmr
            @timeit solver.timer "lsqr_solve" point.y = IterativeSolvers.lsqr(A', rhs)
        else
            if issparse(A) && !(T <: sparse_QR_reals)
                # TODO alternative fallback is to convert sparse{T} to sparse{Float64} and do the sparse LU
                if solver.init_use_fallback
                    @warn("using dense factorization of A' in initial point finding because sparse factorization for number type $T is not supported by SuiteSparse packages")
                    @timeit solver.timer "qr_fact" Ap_fact = qr!(Matrix(A'))
                else
                    error("sparse factorization for number type $T is not supported by SuiteSparse packages, so Hypatia cannot find an initial point")
                end
            else
                @timeit solver.timer "qr_fact" Ap_fact = issparse(A) ? qr(sparse(A')) : qr!(Matrix(A'))
            end

            Ap_rank = get_rank_est(Ap_fact, solver.init_tol_qr)
            if Ap_rank < p
                @warn("some primal equalities appear to be dependent; try using preprocess = true")
            end

            @timeit solver.timer "qr_solve" point.y = Ap_fact \ rhs
        end
    end

    return point
end

# reduce model by removing all primal equalities
function preprocess_find_initial_point(solver::Solver{T}) where {T <: Real}
    orig_model = solver.orig_model
    A = copy(orig_model.A)
    b = copy(orig_model.b)
    G = copy(orig_model.G)
    h = copy(orig_model.h)
    obj_offset = orig_model.obj_offset
    n = length(c)
    p = length(b)
    q = orig_model.q

    # remove all primal equalities by making A and b empty with n = n0 - p0 and p = 0:
    Q1_idxs = 1:Ap_rank
    Q2_idxs = (Ap_rank + 1):n
    # TODO improve efficiency, and maybe move some of the transformations above to here
    # NOTE avoid calculating GQ1 explicitly if possible
    # [cQ1 cQ2] = c0' * Q
    cQ = orig_model.c' * Ap_Q
    cQ1 = cQ[Q1_idxs]
    cQ2 = cQ[Q2_idxs]
    # c = cQ2
    c = cQ2
    # offset = offset0 + cQ1 * (R' \ b0)
    Rpib0 = Ap_R' \ b
    obj_offset += dot(cQ1, Rpib0)
    # [GQ1 GQ2] = G0 * Q
    GQ = G * Ap_Q
    GQ1 = GQ[:, Q1_idxs]
    GQ2 = GQ[:, Q2_idxs]
    # h = h0 - GQ1 * (R' \ b0)
    h -= GQ1 * Rpib0
    # G = GQ2
    G = GQ2

    # A and b empty
    n = length(c)
    p = 0
    A = zeros(T, 0, n)
    b = zeros(T, 0)
    solver.Ap_R = UpperTriangular(zeros(T, 0, 0))
    solver.Ap_Q = I

    point.y = zeros(T, 0)
    point.x =

    # recover original-space solution using:
    # x0 = Q * [(R' \ b0), x]
    # y0 = R \ (-cQ1' - GQ1' * z0)
    # TODO save what's needed

    solver.model = Models.Model{T}(c, A, b, G, h, orig_model.cones, obj_offset = obj_offset)
end

# preprocess and get initial point
function preprocess_find_initial_point(solver::Solver{T}) where {T <: Real}
    orig_model = solver.orig_model
    c = copy(orig_model.c)
    A = copy(orig_model.A)
    b = copy(orig_model.b)
    G = copy(orig_model.G)
    n = length(c)
    p = length(b)
    q = orig_model.q
    point = solver.point

    # solve for x as least squares solution to Ax = b, Gx = h - s
    @timeit solver.timer "preproc_x" if !iszero(n)
        # get pivoted QR # TODO when Julia has a unified QR interface, replace this
        AG = vcat(A, G)
        if issparse(AG) && !(T <: sparse_QR_reals)
            if solver.init_use_fallback
                @warn("using dense factorization of [A; G] in preprocessing and initial point finding because sparse factorization for number type $T is not supported by SuiteSparse packages")
                AG = Matrix(AG)
            else
                error("sparse factorization for number type $T is not supported by SuiteSparse packages, so Hypatia cannot preprocess and find an initial point")
            end
        end
        @timeit solver.timer "qr_fact" AG_fact = issparse(AG) ? qr(AG, tol = solver.init_tol_qr) : qr(AG, Val(true))

        AG_rank = get_rank_est(AG_fact, solver.init_tol_qr)
        if solver.reduce
            @assert AG_rank == n # TODO cannot preprocess dual equalities if have used reduction, since reduction stores matrices needed for transformation to original space at end of solve
        end

        if AG_rank == n
            # no dual equalities to remove
            x_keep_idxs = 1:n
            @timeit solver.timer "qr_solve" point.x = AG_fact \ vcat(b, orig_model.h - point.s)
        else
            col_piv = (AG_fact isa QRPivoted{T, Matrix{T}}) ? AG_fact.p : AG_fact.pcol
            x_keep_idxs = col_piv[1:AG_rank]
            AG_R = UpperTriangular(AG_fact.R[1:AG_rank, 1:AG_rank])

            c_sub = c[x_keep_idxs]
            @timeit solver.timer "residual" begin
                # yz_sub = AG_fact.Q * vcat((AG_R' \ c_sub), zeros(p + q - AG_rank))
                yz_sub = zeros(p + q)
                yz_sub1 = view(yz_sub, 1:AG_rank)
                copyto!(yz_sub1, c_sub)
                ldiv!(AG_R', yz_sub1)
                lmul!(AG_fact.Q, yz_sub)
            end
            if !(AG_fact isa QRPivoted{T, Matrix{T}})
                yz_sub = yz_sub[AG_fact.rpivinv]
            end
            residual = norm(AG' * yz_sub - c, Inf)
            if residual > solver.init_tol_qr
                solver.verbose && println("some dual equality constraints are inconsistent (residual $residual, tolerance $(solver.init_tol_qr))")
                solver.status = :DualInconsistent
                return point
            end
            solver.verbose && println("removed $(n - AG_rank) out of $n dual equality constraints")

            @timeit solver.timer "qr_solve" begin
                # point.x = AG_R \ ((AG_fact.Q' * vcat(b, orig_model.h - point.s))[1:AG_rank])
                tmp = vcat(b, orig_model.h - point.s)
                lmul!(AG_fact.Q', tmp)
                point.x = tmp[1:AG_rank]
                ldiv!(AG_R, point.x)
            end

            c = c_sub
            A = A[:, x_keep_idxs]
            G = G[:, x_keep_idxs]
            n = AG_rank
        end
    else
        x_keep_idxs = Int[]
    end

    # solve for y as least squares solution to A'y = -c - G'z
    @timeit solver.timer "preproc_y" if !iszero(p)
        if issparse(A) && !(T <: sparse_QR_reals)
            if solver.init_use_fallback
                @warn("using dense factorization of A' in preprocessing and initial point finding because sparse factorization for number type $T is not supported by SuiteSparse packages")
                @timeit solver.timer "qr_fact" Ap_fact = qr!(Matrix(A'), Val(true))
            else
                error("sparse factorization for number type $T is not supported by SuiteSparse packages, so Hypatia cannot preprocess and find an initial point")
            end
        else
            @timeit solver.timer "qr_fact" Ap_fact = issparse(A) ? qr(sparse(A'), tol = solver.init_tol_qr) : qr(A', Val(true))
        end

        Ap_rank = get_rank_est(Ap_fact, solver.init_tol_qr)

        Ap_R = UpperTriangular(Ap_fact.R[1:Ap_rank, 1:Ap_rank])
        col_piv = (Ap_fact isa QRPivoted{T, Matrix{T}}) ? Ap_fact.p : Ap_fact.pcol
        y_keep_idxs = col_piv[1:Ap_rank]
        Ap_Q = Ap_fact.Q

        b_sub = b[y_keep_idxs]
        if Ap_rank < p
            # some dependent primal equalities, so check if they are consistent
            @timeit solver.timer "residual" begin
                # x_sub = Ap_Q * vcat((Ap_R' \ b_sub), zeros(n - Ap_rank))
                x_sub = zeros(n)
                x_sub1 = view(x_sub, 1:Ap_rank)
                copyto!(x_sub1, b_sub)
                ldiv!(Ap_R', x_sub1)
                lmul!(Ap_Q, x_sub)
            end

            if !(Ap_fact isa QRPivoted{T, Matrix{T}})
                x_sub = x_sub[Ap_fact.rpivinv]
            end
            residual = norm(A * x_sub - b, Inf)
            if residual > solver.init_tol_qr
                solver.verbose && println("some primal equality constraints are inconsistent (residual $residual, tolerance $(solver.init_tol_qr))")
                solver.status = :PrimalInconsistent
            end
            solver.verbose && println("removed $(p - Ap_rank) out of $p primal equality constraints")
        end

        @timeit solver.timer "qr_solve" begin
            # point.y = Ap_R \ ((Ap_fact.Q' * (-c - G' * point.z))[1:Ap_rank])
            tmp = -c
            mul!(tmp, G', point.z, -1, true)
            lmul!(Ap_fact.Q', tmp)
            point.y = tmp[1:Ap_rank]
            ldiv!(Ap_R, point.y)
        end

        if !(Ap_fact isa QRPivoted{T, Matrix{T}})
            row_piv = Ap_fact.prow
            A = A[y_keep_idxs, row_piv]
            c = c[row_piv]
            G = G[:, row_piv]
            x_keep_idxs = x_keep_idxs[row_piv]
        else
            A = A[y_keep_idxs, :]
        end
        b = b_sub
        p = Ap_rank
        solver.Ap_R = Ap_R
        solver.Ap_Q = Ap_Q
    else
        y_keep_idxs = Int[]
        solver.Ap_R = UpperTriangular(zeros(T, 0, 0))
        solver.Ap_Q = I
    end

    solver.x_keep_idxs = x_keep_idxs
    solver.y_keep_idxs = y_keep_idxs
    solver.model = Models.Model{T}(c, A, b, G, orig_model.h, orig_model.cones, obj_offset = orig_model.obj_offset)

    return point
end

function calc_mu(solver::Solver{T}) where {T <: Real}
    solver.mu = (dot(solver.point.z, solver.point.s) + solver.tau * solver.kap) /
        (one(T) + solver.model.nu)
    return solver.mu
end

# NOTE (pivoted) QR factorizations are usually rank-revealing but may be unreliable, see http://www.math.sjsu.edu/~foster/rankrevealingcode.html
# TODO could replace this with rank(Ap_fact) when available for both dense and sparse
function get_rank_est(qr_fact, init_tol_qr::Real)
    R = qr_fact.R
    rank_est = 0
    for i in 1:size(R, 1) # TODO could replace this with rank(AG_fact) when available for both dense and sparse
        if abs(R[i, i]) > init_tol_qr
            rank_est += 1
        end
    end
    return rank_est
end

function calc_residual(solver::Solver{T}) where {T <: Real}
    model = solver.model
    point = solver.point

    # x_residual = -A'*y - G'*z - c*tau
    x_residual = solver.x_residual
    mul!(x_residual, model.G', point.z)
    mul!(x_residual, model.A', point.y, true, true)
    solver.x_norm_res_t = norm(x_residual)
    @. x_residual += model.c * solver.tau
    solver.x_norm_res = norm(x_residual) / solver.tau
    @. x_residual *= -1

    # y_residual = A*x - b*tau
    y_residual = solver.y_residual
    mul!(y_residual, model.A, point.x)
    solver.y_norm_res_t = norm(y_residual)
    @. y_residual -= model.b * solver.tau
    solver.y_norm_res = norm(y_residual) / solver.tau

    # z_residual = s + G*x - h*tau
    z_residual = solver.z_residual
    mul!(z_residual, model.G, point.x)
    @. z_residual += point.s
    solver.z_norm_res_t = norm(z_residual)
    @. z_residual -= model.h * solver.tau
    solver.z_norm_res = norm(z_residual) / solver.tau

    return
end

function calc_convergence_params(solver::Solver{T}) where {T <: Real}
    model = solver.model
    point = solver.point

    solver.prev_gap = solver.gap
    solver.prev_rel_gap = solver.rel_gap
    solver.prev_x_feas = solver.x_feas
    solver.prev_y_feas = solver.y_feas
    solver.prev_z_feas = solver.z_feas

    solver.primal_obj_t = dot(model.c, point.x)
    solver.dual_obj_t = -dot(model.b, point.y) - dot(model.h, point.z)
    solver.primal_obj = solver.primal_obj_t / solver.tau + model.obj_offset
    solver.dual_obj = solver.dual_obj_t / solver.tau + model.obj_offset
    solver.gap = dot(point.z, point.s)
    if solver.primal_obj < zero(T)
        solver.rel_gap = solver.gap / -solver.primal_obj
    elseif solver.dual_obj > zero(T)
        solver.rel_gap = solver.gap / solver.dual_obj
    else
        solver.rel_gap = NaN
    end

    solver.x_feas = solver.x_norm_res * solver.x_conv_tol
    solver.y_feas = solver.y_norm_res * solver.y_conv_tol
    solver.z_feas = solver.z_norm_res * solver.z_conv_tol

    return
end

function print_iteration_stats(solver::Solver{T}) where {T <: Real}
    if iszero(solver.num_iters)
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
            "iter", "p_obj", "d_obj", "abs_gap", "rel_gap",
            "x_feas", "y_feas", "z_feas", "tau", "kap", "mu",
            "gamma", "alpha",
            )
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap, solver.rel_gap,
            solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu
            )
    else
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap, solver.rel_gap,
            solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu,
            solver.prev_gamma, solver.prev_alpha,
            )
    end
    flush(stdout)
    return
end

function check_convergence(solver::Solver{T}) where {T <: Real}
    # check convergence criteria
    # TODO nearly primal or dual infeasible or nearly optimal cases?
    if max(solver.x_feas, solver.y_feas, solver.z_feas) <= solver.tol_feas &&
        (solver.gap <= solver.tol_abs_opt || (!isnan(solver.rel_gap) && solver.rel_gap <= solver.tol_rel_opt))
        solver.verbose && println("optimal solution found; terminating")
        solver.status = :Optimal
        return true
    end
    if solver.dual_obj_t > zero(T)
        infres_pr = solver.x_norm_res_t * solver.x_conv_tol / solver.dual_obj_t
        if infres_pr <= solver.tol_feas
            solver.verbose && println("primal infeasibility detected; terminating")
            solver.status = :PrimalInfeasible
            return true
        end
    end
    if solver.primal_obj_t < zero(T)
        infres_du = -max(solver.y_norm_res_t * solver.y_conv_tol, solver.z_norm_res_t * solver.z_conv_tol) / solver.primal_obj_t
        if infres_du <= solver.tol_feas
            solver.verbose && println("dual infeasibility detected; terminating")
            solver.status = :DualInfeasible
            return true
        end
    end
    if solver.mu <= solver.tol_feas * T(1e-2) && solver.tau <= solver.tol_feas * T(1e-2) * min(one(T), solver.kap)
        solver.verbose && println("ill-posedness detected; terminating")
        solver.status = :IllPosed
        return true
    end

    max_improve = zero(T)
    for (curr, prev) in ((solver.gap, solver.prev_gap), (solver.rel_gap, solver.prev_rel_gap),
        (solver.x_feas, solver.prev_x_feas), (solver.y_feas, solver.prev_y_feas), (solver.z_feas, solver.prev_z_feas))
        if isnan(prev) || isnan(curr)
            continue
        end
        max_improve = max(max_improve, (prev - curr) / (abs(prev) + eps(T)))
    end
    if max_improve < solver.tol_slow
        if solver.prev_is_slow && solver.prev2_is_slow
            solver.verbose && println("slow progress in consecutive iterations; terminating")
            solver.status = :SlowProgress
            return true
        else
            solver.prev2_is_slow = solver.prev_is_slow
            solver.prev_is_slow = true
        end
    else
        solver.prev2_is_slow = solver.prev_is_slow
        solver.prev_is_slow = false
    end

    return false
end
