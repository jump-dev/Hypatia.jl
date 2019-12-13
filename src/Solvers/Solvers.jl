#=
Copyright 2018, Chris Coey and contributors

functions and caches for interior point algorithms
=#

module Solvers

using Printf
using LinearAlgebra
using SparseArrays
import SuiteSparse
import IterativeSolvers
using Test
using TimerOutputs
import Hypatia.Cones
import Hypatia.Models
import Hypatia.BlockMatrix
import Hypatia.SparseNonSymCache
import Hypatia.SparseSymCache
import Hypatia.update_fact
import Hypatia.solve_system
import Hypatia.free_memory
import Hypatia.int_type
import Hypatia.DenseNonSymCache
import Hypatia.DenseSymCache
import Hypatia.DensePosDefCache
import Hypatia.load_matrix
import Hypatia.update_fact
import Hypatia.solve_system
import Hypatia.invert

abstract type Stepper{T <: Real} end

abstract type SystemSolver{T <: Real} end

mutable struct Solver{T <: Real}
    # main options
    verbose::Bool
    iter_limit::Int
    time_limit::Float64
    tol_rel_opt::T
    tol_abs_opt::T
    tol_feas::T
    tol_slow::T
    preprocess::Bool
    reduce::Bool
    init_use_indirect::Bool
    init_tol_qr::T
    init_use_fallback::Bool
    max_nbhd::T
    use_infty_nbhd::Bool
    stepper::Stepper{T}
    system_solver::SystemSolver{T}

    # current status of the solver object
    status::Symbol

    # solve info and timers
    solve_time::Float64
    timer::TimerOutput
    num_iters::Int

    # model and preprocessed model data
    orig_model::Models.Model{T}
    model::Models.Model{T}
    x_keep_idxs::AbstractVector{Int}
    y_keep_idxs::AbstractVector{Int}
    Ap_R::UpperTriangular{T, <:AbstractMatrix{T}}
    Ap_Q::Union{UniformScaling, AbstractMatrix{T}}
    reduce_cQ1
    reduce_Rpib0
    reduce_GQ1
    reduce_Ap_R
    reduce_Ap_Q
    reduce_y_keep_idxs
    reduce_row_piv_inv

    # current iterate
    point::Models.Point{T}
    tau::T
    kap::T
    mu::T

    # residuals
    x_residual::Vector{T}
    y_residual::Vector{T}
    z_residual::Vector{T}
    x_norm_res_t::T
    y_norm_res_t::T
    z_norm_res_t::T
    x_norm_res::T
    y_norm_res::T
    z_norm_res::T

    # convergence parameters
    primal_obj_t::T
    dual_obj_t::T
    primal_obj::T
    dual_obj::T
    gap::T
    rel_gap::T
    x_feas::T
    y_feas::T
    z_feas::T

    # termination condition helpers
    x_conv_tol::T
    y_conv_tol::T
    z_conv_tol::T
    prev_is_slow::Bool
    prev2_is_slow::Bool
    prev_gap::T
    prev_rel_gap::T
    prev_x_feas::T
    prev_y_feas::T
    prev_z_feas::T

    # step helpers
    keep_iterating::Bool
    prev_aff_alpha::T
    prev_gamma::T
    prev_alpha::T
    z_temp::Vector{T}
    s_temp::Vector{T}
    primal_views
    dual_views
    nbhd_temp
    cones_infeas::Vector{Bool}
    cones_loaded::Vector{Bool}

    function Solver{T}(;
        verbose::Bool = true,
        iter_limit::Int = 1000,
        time_limit::Real = Inf,
        tol_rel_opt::Real = sqrt(eps(T)),
        tol_abs_opt::Real = sqrt(eps(T)),
        tol_feas::Real = sqrt(eps(T)),
        tol_slow::Real = 1e-3,
        preprocess::Bool = true,
        reduce::Bool = true,
        init_use_indirect::Bool = false,
        init_tol_qr::Real = 100 * eps(T),
        init_use_fallback::Bool = true,
        max_nbhd::Real = 0.7,
        use_infty_nbhd::Bool = false,
        stepper::Stepper{T} = CombinedStepper{T}(),
        system_solver::SystemSolver{T} = QRCholDenseSystemSolver{T}(),
        ) where {T <: Real}
        if isa(system_solver, QRCholSystemSolver{T})
            @assert preprocess # require preprocessing for QRCholSystemSolver # TODO only need primal eq preprocessing or reduction
        end
        if reduce
            @assert preprocess # cannot use reduction without preprocessing # TODO only need primal eq preprocessing
        end
        @assert !(init_use_indirect && preprocess) # cannot use preprocessing and indirect methods for initial point

        solver = new{T}()

        solver.verbose = verbose
        solver.iter_limit = iter_limit
        solver.time_limit = time_limit
        solver.tol_rel_opt = tol_rel_opt
        solver.tol_abs_opt = tol_abs_opt
        solver.tol_feas = tol_feas
        solver.tol_slow = tol_slow
        solver.preprocess = preprocess
        solver.reduce = reduce
        solver.init_use_indirect = init_use_indirect
        solver.init_tol_qr = init_tol_qr
        solver.init_use_fallback = init_use_fallback
        solver.max_nbhd = max_nbhd
        solver.use_infty_nbhd = use_infty_nbhd
        solver.stepper = stepper
        solver.system_solver = system_solver
        solver.status = :NotLoaded

        return solver
    end
end

function solve(solver::Solver{T}) where {T <: Real}
    @assert solver.status == :Loaded
    solver.status = :SolveCalled
    start_time = time()
    solver.num_iters = 0
    solver.solve_time = NaN
    solver.timer = TimerOutput()

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

    # preprocess and find initial point
    @timeit solver.timer "initialize" begin
        orig_model = solver.orig_model
        model = solver.model = Models.Model{T}(orig_model.c, orig_model.A, orig_model.b, orig_model.G, orig_model.h, orig_model.cones, obj_offset = orig_model.obj_offset) # copy original model to solver.model, which may be modified

        @timeit solver.timer "init_cone" point = solver.point = initialize_cone_point(solver.orig_model.cones, solver.orig_model.cone_idxs)

        if solver.reduce
            # TODO don't find point / unnecessary stuff before reduce
            @timeit solver.timer "remove_prim_eq" point.y = find_initial_y(solver, true)
            @timeit solver.timer "preproc_init_x" point.x = find_initial_x(solver)
        else
            @timeit solver.timer "preproc_init_x" point.x = find_initial_x(solver)
            @timeit solver.timer "preproc_init_y" point.y = find_initial_y(solver, false)
        end

        if solver.status != :SolveCalled
            point.x = fill(NaN, orig_model.n)
            point.y = fill(NaN, orig_model.p)
            return solver
        end

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

# optionally preprocess dual equalities and solve for x as least squares solution to Ax = b, Gx = h - s
function find_initial_x(solver::Solver{T}) where {T <: Real}
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

    rhs = vcat(model.b, model.h - solver.point.s)

    # indirect method
    if solver.init_use_indirect
        # TODO pick lsqr or lsmr
        if iszero(p)
            AG = G
        else
            # TODO use LinearMaps.jl
            AG = BlockMatrix{T}(p + q, n, [A, G], [1:p, (p + 1):(p + q)], [1:n, 1:n])
        end
        @timeit solver.timer "lsqr_solve" init_x = IterativeSolvers.lsqr(AG, rhs)
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
    @timeit solver.timer "qr_fact" if issparse(AG)
        if !(T <: Float64)
            if solver.init_use_fallback
                @warn("using dense factorization of [A; G] in preprocessing and initial point finding because sparse factorization for number type $T is not supported by SuiteSparse packages")
                AG_fact = qr!(Matrix(AG), Val(true))
            else
                error("sparse factorization for number type $T is not supported by SuiteSparse packages, so Hypatia cannot preprocess and find an initial point")
            end
        else
            AG_fact = qr(AG, tol = solver.init_tol_qr)
        end
    else
        AG_fact = qr!(AG, Val(true))
    end
    AG_rank = get_rank_est(AG_fact, solver.init_tol_qr)

    if !solver.preprocess || (AG_rank == n)
        AG_rank < n && @warn("some dual equalities appear to be dependent (possibly inconsistent); try using preprocess = true")
        @timeit solver.timer "qr_solve" init_x = AG_fact \ rhs
        return init_x
    end

    # preprocess dual equalities
    col_piv = (AG_fact isa QRPivoted{T, Matrix{T}}) ? AG_fact.p : AG_fact.pcol
    x_keep_idxs = col_piv[1:AG_rank]
    AG_R = UpperTriangular(AG_fact.R[1:AG_rank, 1:AG_rank])

    c_sub = model.c[x_keep_idxs]
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
    @views residual = norm(A' * yz_sub[1:p] + G' * yz_sub[(p + 1):end] - model.c, Inf)
    if residual > solver.init_tol_qr
        solver.verbose && println("some dual equality constraints are inconsistent (residual $residual, tolerance $(solver.init_tol_qr))")
        solver.status = :DualInconsistent
        return zeros(T, 0)
    end
    solver.verbose && println("$(n - AG_rank) out of $n dual equality constraints are inconsistent")

    # modify solver.model to remove/reorder some primal variables x
    model.c = c_sub
    model.A = A[:, x_keep_idxs]
    model.G = G[:, x_keep_idxs]
    model.n = AG_rank
    solver.x_keep_idxs = x_keep_idxs

    @timeit solver.timer "qr_solve" begin
        # init_x = AG_R \ ((AG_fact.Q' * vcat(b, h - point.s))[1:AG_rank])
        tmp = vcat(model.b, model.h - solver.point.s)
        lmul!(AG_fact.Q', tmp)
        init_x = tmp[1:model.n]
        ldiv!(AG_R, init_x)
    end

    return init_x
end

# optionally preprocess primal equalities and solve for y as least squares solution to A'y = -c - G'z
function find_initial_y(solver::Solver{T}, reducing::Bool) where {T <: Real}
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

    if !reducing
        # rhs = -c - G' * point.z
        rhs = copy(model.c)
        mul!(rhs, model.G', solver.point.z, -1, -1)

        # indirect method
        if solver.init_use_indirect
            # TODO pick lsqr or lsmr
            @timeit solver.timer "lsqr_solve" init_y = IterativeSolvers.lsqr(A', rhs)
            return init_y
        end
    end

    # factorize A'
    @timeit solver.timer "qr_fact" if issparse(A)
        if !(T <: Float64)
            if solver.init_use_fallback
                @warn("using dense factorization of A' in preprocessing and initial point finding because sparse factorization for number type $T is not supported by SuiteSparse packages")
                Ap_fact = qr!(Matrix(A'), Val(true))
            else
                error("sparse factorization for number type $T is not supported by SuiteSparse packages, so Hypatia cannot preprocess and find an initial point")
            end
        else
            Ap_fact = qr(sparse(A'), tol = solver.init_tol_qr)
        end
    else
        Ap_fact = qr!(Matrix(A'), Val(true))
    end
    Ap_rank = get_rank_est(Ap_fact, solver.init_tol_qr)

    if !reducing && !solver.preprocess
        Ap_rank < p && @warn("some primal equalities appear to be dependent (possibly inconsistent); try using preprocess = true")
        @timeit solver.timer "qr_solve" init_y = Ap_fact \ rhs
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
        residual = norm(A * x_sub - model.b, Inf)
        if residual > solver.init_tol_qr
            solver.verbose && println("some primal equality constraints are inconsistent (residual $residual, tolerance $(solver.init_tol_qr))")
            solver.status = :PrimalInconsistent
            return zeros(T, 0)
        end
        solver.verbose && println("$(p - Ap_rank) out of $p primal equality constraints are inconsistent")
    end

    if reducing
        # remove all primal equalities by making A and b empty with n = n0 - p0 and p = 0
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
        GQ = model.G * Ap_Q
        GQ1 = solver.reduce_GQ1 = GQ[:, Q1_idxs]
        GQ2 = GQ[:, Q2_idxs]
        # h = h0 - GQ1 * (R' \ b0)
        model.h -= GQ1 * Rpib0 # TODO replace with below when working
        # mul!(model.h, GQ1, Rpib0, -1, true)

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

    @timeit solver.timer "qr_solve" begin
        # init_y = Ap_R \ ((Ap_fact.Q' * (-c - G' * point.z))[1:Ap_rank])
        tmp = copy(model.c)
        mul!(tmp, model.G', solver.point.z, true, true)
        lmul!(Ap_fact.Q', tmp)
        init_y = tmp[1:Ap_rank]
        init_y .*= -1
        ldiv!(Ap_R, init_y)
    end

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

get_timer(solver::Solver) = solver.timer

get_status(solver::Solver) = solver.status
get_solve_time(solver::Solver) = solver.solve_time
get_num_iters(solver::Solver) = solver.num_iters

get_primal_obj(solver::Solver) = solver.primal_obj
get_dual_obj(solver::Solver) = solver.dual_obj

get_s(solver::Solver) = copy(solver.point.s)
get_z(solver::Solver) = copy(solver.point.z)

function get_x(solver::Solver{T}) where {T <: Real}
    if solver.preprocess && !iszero(solver.orig_model.n) && !any(isnan, solver.point.x)
        # unpreprocess solver's solution
        if solver.reduce && !iszero(solver.orig_model.p)
            # unreduce solver's solution
            # x0 = Q * [(R' \ b0), x]
            x = zeros(T, solver.orig_model.n - length(solver.reduce_Rpib0))
            x[solver.x_keep_idxs] = solver.point.x
            x = vcat(solver.reduce_Rpib0, x)
            lmul!(solver.reduce_Ap_Q, x)
            if !isempty(solver.reduce_row_piv_inv)
                x = x[solver.reduce_row_piv_inv]
            end
        else
            x = zeros(T, solver.orig_model.n)
            x[solver.x_keep_idxs] = solver.point.x
        end
    else
        x = copy(solver.point.x)
    end

    return x
end

function get_y(solver::Solver{T}) where {T <: Real}
    if solver.preprocess && !iszero(solver.orig_model.p) && !any(isnan, solver.point.y)
        # unpreprocess solver's solution
        y = zeros(T, solver.orig_model.p)
        if solver.reduce
            # unreduce solver's solution
            # y0 = R \ (-cQ1' - GQ1' * z0)
            y0 = solver.reduce_cQ1
            mul!(y0, solver.reduce_GQ1', solver.point.z, -1, -1)
            y[solver.reduce_y_keep_idxs] = ldiv!(solver.reduce_Ap_R, y0[1:length(solver.reduce_y_keep_idxs)])
        else
            y[solver.y_keep_idxs] = solver.point.y
        end
    else
        y = copy(solver.point.y)
    end

    return y
end

get_tau(solver::Solver) = solver.tau
get_kappa(solver::Solver) = solver.kap
get_mu(solver::Solver) = solver.mu

function load(solver::Solver{T}, model::Models.Model{T}) where {T <: Real}
    # @assert solver.status == :NotLoaded # TODO maybe want a reset function that just keeps options
    solver.orig_model = model
    solver.status = :Loaded
    return solver
end

# solve, optionally test conic certificates, and return solve information
function solve_check(
    model::Models.Model{T};
    solver::Solver{T} = Solver{T}(),
    test::Bool = true,
    atol::Real = sqrt(sqrt(eps(T))),
    rtol::Real = sqrt(sqrt(eps(T))),
    ) where {T <: Real}
    load(solver, model)
    solve(solver)

    status = get_status(solver)
    primal_obj = get_primal_obj(solver)
    dual_obj = get_dual_obj(solver)
    x = get_x(solver)
    y = get_y(solver)
    s = get_s(solver)
    z = get_z(solver)

    if test
        (c, A, b, G, h, cones, obj_offset) = (model.c, model.A, model.b, model.G, model.h, model.cones, model.obj_offset)
        if status == :Optimal
            @test primal_obj ≈ dual_obj atol=atol rtol=rtol
            @test A * x ≈ b atol=atol rtol=rtol
            @test G * x + s ≈ h atol=atol rtol=rtol
            @test G' * z + A' * y ≈ -c atol=atol rtol=rtol
            @test dot(c, x) + obj_offset ≈ primal_obj atol=atol^2 rtol=rtol^2
            @test -dot(b, y) - dot(h, z) + obj_offset ≈ dual_obj atol=atol^2 rtol=rtol^2
            @test dot(s, z) ≈ zero(T) atol=10atol rtol=10rtol
        elseif status == :PrimalInfeasible
            @test dual_obj > obj_offset
            @test -dot(b, y) - dot(h, z) + obj_offset ≈ dual_obj atol=atol^2 rtol=rtol^2
            # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
            # @test G' * z ≈ -A' * y atol=atol rtol=rtol
        elseif status == :DualInfeasible
            @test primal_obj < obj_offset
            @test dot(c, x) + obj_offset ≈ primal_obj atol=atol^2 rtol=rtol^2
            # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
            # @test G * x ≈ -s atol=atol rtol=rtol
            # @test A * x ≈ zeros(T, length(y)) atol=atol rtol=rtol
        elseif status == :IllPosed
            # TODO primal vs dual ill-posed statuses and conditions
        end
    end

    solve_time = get_solve_time(solver)

    return (solver = solver, model = model, solve_time = solve_time, primal_obj = primal_obj, dual_obj = dual_obj, status = status, x = x, y = y, s = s, z = z)
end

# build model and call solve_check
function build_solve_check(
    c::Vector{T},
    A,
    b::Vector{T},
    G,
    h::Vector{T},
    cones::Vector{Cones.Cone{T}};
    obj_offset::T = zero(T),
    other_options...
    ) where {T <: Real}
    model = Models.Model{T}(c, A, b, G, h, cones, obj_offset = obj_offset)
    return solve_check(model; other_options...)
end

include("stepper.jl")
include("naive.jl")
include("naiveelim.jl")
include("symindef.jl")
include("qrchol.jl")

# release memory used by sparse system solvers
free_memory(::SystemSolver) = nothing
free_memory(system_solver::Union{NaiveSparseSystemSolver, SymIndefSparseSystemSolver}) = free_memory(system_solver.fact_cache)

end
