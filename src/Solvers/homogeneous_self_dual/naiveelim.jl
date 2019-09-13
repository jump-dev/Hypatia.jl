#=
Copyright 2018, Chris Coey and contributors

naive+elimination linear system solver
solves linear system in naive.jl via the following procedure

eliminate s
-G*x + h*tau - s = zrhs
so if using primal barrier
z_k + mu*H_k*s_k = srhs_k --> s_k = (mu*H_k)\(srhs_k - z_k)
-->
-G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
-->
-mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
or if using dual barrier
mu*H_k*z_k + s_k = srhs_k --> s_k = srhs_k - mu*H_k*z_k
-->
-G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k

eliminate kap
-c'x - b'y - h'z - kap = taurhs
so
mu/(taubar^2)*tau + kap = kaprhs --> kap = kaprhs - mu/(taubar^2)*tau
-->
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs

TODO add iterative method
=#

mutable struct NaiveElimSystemSolver{T <: Real} <: SystemSolver{T}
    use_sparse::Bool

    solver::Solver{T}

    x1
    x2
    y1
    y2
    z1
    z2
    z1_k
    z2_k
    s1::Vector{T}
    s2::Vector{T}
    s1_k
    s2_k

    lhs_copy
    lhs
    rhs::Matrix{T}

    solvesol
    solvecache

    function NaiveElimSystemSolver{T}(; use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_sparse = use_sparse
        return system_solver
    end
end

function load(system_solver::NaiveElimSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.solver = solver

    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    rhs = zeros(T, n + p + q + 1, 2)
    system_solver.rhs = rhs
    rows = 1:n
    system_solver.x1 = view(rhs, rows, 1)
    system_solver.x2 = view(rhs, rows, 2)
    rows = (n + 1):(n + p)
    system_solver.y1 = view(rhs, rows, 1)
    system_solver.y2 = view(rhs, rows, 2)
    rows = (n + p + 1):(n + p + q)
    system_solver.z1 = view(rhs, rows, 1)
    system_solver.z2 = view(rhs, rows, 2)
    z_start = n + p
    system_solver.z1_k = [view(rhs, z_start .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
    system_solver.z2_k = [view(rhs, z_start .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]
    system_solver.s1 = zeros(T, q)
    system_solver.s2 = zeros(T, q)
    system_solver.s1_k = [view(system_solver.s1, model.cone_idxs[k]) for k in eachindex(model.cones)]
    system_solver.s2_k = [view(system_solver.s2, model.cone_idxs[k]) for k in eachindex(model.cones)]

    if system_solver.use_sparse
        system_solver.lhs_copy = T[
            spzeros(T,n,n)  model.A'        model.G'              model.c;
            -model.A        spzeros(T,p,p)  spzeros(T,p,q)        model.b;
            -model.G        spzeros(T,q,p)  sparse(one(T)*I,q,q)  model.h;
            -model.c'       -model.b'       -model.h'             one(T);
            ]
        dropzeros!(system_solver.lhs_copy)
        @assert issparse(system_solver.lhs_copy)
    else
        system_solver.lhs_copy = T[
            zeros(T,n,n)  model.A'      model.G'              model.c;
            -model.A      zeros(T,p,p)  zeros(T,p,q)          model.b;
            -model.G      zeros(T,q,p)  Matrix(one(T)*I,q,q)  model.h;
            -model.c'     -model.b'     -model.h'             one(T);
            ]
    end

    system_solver.lhs = similar(system_solver.lhs_copy)

    if !system_solver.use_sparse
        system_solver.solvesol = Matrix{T}(undef, size(system_solver.lhs, 1), 2)
        system_solver.solvecache = HypLUSolveCache(system_solver.solvesol, system_solver.lhs, rhs)
    end

    return system_solver
end

function get_combined_directions(system_solver::NaiveElimSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    x1 = system_solver.x1
    x2 = system_solver.x2
    y1 = system_solver.y1
    y2 = system_solver.y2
    z1 = system_solver.z1
    z2 = system_solver.z2
    z1_k = system_solver.z1_k
    z2_k = system_solver.z2_k
    s1 = system_solver.s1
    s2 = system_solver.s2
    s1_k = system_solver.s1_k
    s2_k = system_solver.s2_k

    sqrtmu = sqrt(solver.mu)

    # update rhs and lhs matrices
    x1 .= solver.x_residual
    x2 .= zero(T)
    y1 .= solver.y_residual
    y2 .= zero(T)
    for k in eachindex(cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cones[k])
        @. s1_k[k] = -duals_k
        @. s2_k[k] = -duals_k - grad_k * sqrtmu
    end

    # -c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs
    tau_rhs1 = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    kap_rhs1 = -solver.kap
    kap_rhs2 = -solver.kap + solver.mu / solver.tau
    rhs[end, 1] = tau_rhs1 + kap_rhs1
    rhs[end, 2] = kap_rhs2

    copyto!(lhs, system_solver.lhs_copy)
    lhs[end, end] = solver.mu / solver.tau / solver.tau
    for k in eachindex(cones)
        cone_k = cones[k]
        idxs_k = model.cone_idxs[k]
        rows_k = (n + p) .+ idxs_k
        # TODO optimize by preallocing the views
        if Cones.use_dual(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            lhs[rows_k, rows_k] .= Cones.hess(cone_k)
            @views copyto!(z1_k[k], solver.z_residual[idxs_k])
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(lhs[rows_k, 1:n], model.G[idxs_k, :], cone_k)
            @. lhs[rows_k, 1:n] *= -1
            @views Cones.hess_prod!(lhs[rows_k, end], model.h[idxs_k], cone_k)
            @views Cones.hess_prod!(z1_k[k], solver.z_residual[idxs_k], cone_k)
        end
    end
    z1 .+= s1
    z2 .= s2

    # solve system
    if system_solver.use_sparse
        rhs .= lu(lhs) \ rhs
    else
        if !hyp_lu_solve!(system_solver.solvecache, system_solver.solvesol, lhs, rhs)
            @warn("numerical failure: could not fix linear solve failure (mu is $mu)")
        end
        copyto!(rhs, system_solver.solvesol)
    end

    # lift to get s and kap
    tau1 = rhs[end, 1]
    tau2 = rhs[end, 2]

    # s = -G*x - zrhs + h*tau
    @. s1 = model.h * tau1 - solver.z_residual
    mul!(s1, model.G, x1, -one(T), true)
    s2 .= model.h
    mul!(s2, model.G, x2, -one(T), tau2)

    # kap = kaprhs - mu/(taubar^2)*tau
    kap1 = kap_rhs1 - solver.mu / solver.tau * tau1 / solver.tau
    kap2 = kap_rhs2 - solver.mu / solver.tau * tau2 / solver.tau

    return (x1, x2, y1, y2, z1, z2, tau1, tau2, s1, s2, kap1, kap2)
end
